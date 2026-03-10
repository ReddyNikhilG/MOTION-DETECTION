import cv2
import csv
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

from ai_detector import AIDetector
from vision_utils import CentroidTracker, PredictionSmoother

try:
    import mediapipe as mp
except Exception:
    mp = None


class AICamera(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("AI Age & Mood Detection System")

        self.cap = cv2.VideoCapture(0)
        self.face_detector_mode = "MediaPipe" if mp is not None else "Haar"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.mp_face = None
        if mp is not None:
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5,
            )

        self.last_faces = []
        self.last_predictions = []
        self.last_track_ids = []
        self.frame_count = 0
        self.analysis_interval = 12
        self.detection_interval = 2
        self.max_faces = 3
        self.camera_index = 0
        self.confidence_threshold = 55.0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_future = None
        self.analysis_running = False

        self.detector = AIDetector()
        self.tracker = CentroidTracker(max_distance=90)
        self.smoother = PredictionSmoother(window_size=5)

        self.video_label = QLabel()
        self.status_label = QLabel("Status: Ready")
        self.privacy_label = QLabel("Privacy: Processing stays on this device. Logs are stored locally in logs/.")

        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.export_btn = QPushButton("Export Summary")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Fast", "Balanced", "Quality"])
        self.detector_combo = QComboBox()
        self.detector_combo.addItems(["Auto", "MediaPipe", "Haar"])
        self.analysis_spin = QSpinBox()
        self.analysis_spin.setRange(5, 60)
        self.analysis_spin.setValue(self.analysis_interval)
        self.max_faces_spin = QSpinBox()
        self.max_faces_spin.setRange(1, 6)
        self.max_faces_spin.setValue(self.max_faces)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 100.0)
        self.conf_spin.setValue(self.confidence_threshold)
        self.conf_spin.setSingleStep(1.0)
        self.cam_index_spin = QSpinBox()
        self.cam_index_spin.setRange(0, 5)
        self.cam_index_spin.setValue(self.camera_index)

        layout = QVBoxLayout()
        controls_row = QHBoxLayout()
        controls_form = QFormLayout()
        controls_form.addRow("Profile", self.mode_combo)
        controls_form.addRow("Detector", self.detector_combo)
        controls_form.addRow("Analysis Every N Frames", self.analysis_spin)
        controls_form.addRow("Max Faces", self.max_faces_spin)
        controls_form.addRow("Confidence %", self.conf_spin)
        controls_form.addRow("Camera Index", self.cam_index_spin)
        controls_row.addLayout(controls_form)

        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addLayout(controls_row)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.export_btn)
        layout.addWidget(self.privacy_label)

        self.setLayout(layout)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.export_btn.clicked.connect(self.export_summary)
        self.mode_combo.currentTextChanged.connect(self.apply_mode)
        self.analysis_spin.valueChanged.connect(self.on_settings_changed)
        self.max_faces_spin.valueChanged.connect(self.on_settings_changed)
        self.conf_spin.valueChanged.connect(self.on_settings_changed)
        self.cam_index_spin.valueChanged.connect(self.on_settings_changed)
        self.detector_combo.currentTextChanged.connect(self.on_settings_changed)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self._prepare_logging()
        self.apply_mode("Balanced")

    def _prepare_logging(self):

        self.logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.csv_path = os.path.join(self.logs_dir, "detections.csv")
        self.jsonl_path = os.path.join(self.logs_dir, "detections.jsonl")
        self.last_logged_second = {}

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["timestamp", "track_id", "age", "emotion", "confidence"])

    def apply_mode(self, mode):

        if mode == "Fast":
            self.analysis_interval = 24
            self.detection_interval = 3
            self.max_faces = 2
            width, height = 480, 360
        elif mode == "Quality":
            self.analysis_interval = 8
            self.detection_interval = 1
            self.max_faces = 4
            width, height = 960, 720
        else:
            self.analysis_interval = 12
            self.detection_interval = 2
            self.max_faces = 3
            width, height = 640, 480

        self.analysis_spin.setValue(self.analysis_interval)
        self.max_faces_spin.setValue(self.max_faces)
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def on_settings_changed(self):

        self.analysis_interval = int(self.analysis_spin.value())
        self.max_faces = int(self.max_faces_spin.value())
        self.confidence_threshold = float(self.conf_spin.value())
        self.camera_index = int(self.cam_index_spin.value())

    def start_camera(self):

        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.video_label.setText("Unable to open camera. Check permissions or camera index.")
            self.status_label.setText("Status: Camera unavailable")
            return

        self.apply_mode(self.mode_combo.currentText())
        self.status_label.setText("Status: Camera running")

        self.timer.start(30)

    def stop_camera(self):

        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.status_label.setText("Status: Camera stopped")

    def _detect_faces_haar(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if faces is None or len(faces) == 0:
            return []

        return sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[: self.max_faces]

    def _detect_faces_mediapipe(self, frame):

        if self.mp_face is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mp_face.process(rgb)
        if not result.detections:
            return []

        h, w = frame.shape[:2]
        boxes = []
        for det in result.detections[: self.max_faces]:
            bbox = det.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = max(1, int(bbox.width * w))
            bh = max(1, int(bbox.height * h))
            boxes.append((x, y, bw, bh))

        return boxes

    def _detect_faces(self, frame):

        selected = self.detector_combo.currentText()

        if selected == "Haar":
            return self._detect_faces_haar(frame)

        if selected == "MediaPipe":
            return self._detect_faces_mediapipe(frame)

        if self.face_detector_mode == "MediaPipe":
            mp_boxes = self._detect_faces_mediapipe(frame)
            if mp_boxes:
                return mp_boxes

        return self._detect_faces_haar(frame)

    @staticmethod
    def _analyze_faces(detector, frame, faces):

        results = []
        for x, y, w, h in faces:
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                results.append(None)
                continue

            results.append(detector.analyze(crop))

        return results

    def _submit_analysis(self, frame):

        if not self.last_faces:
            return

        if self.pending_future and not self.pending_future.done():
            return

        faces_snapshot = [tuple(map(int, face)) for face in self.last_faces]
        frame_snapshot = frame.copy()
        self.analysis_running = True
        self.pending_future = self.executor.submit(
            self._analyze_faces,
            self.detector,
            frame_snapshot,
            faces_snapshot
        )

    def _collect_analysis_result(self):

        if not self.pending_future or not self.pending_future.done():
            return

        try:
            result = self.pending_future.result()
            self.last_predictions = result if isinstance(result, list) else []
        except Exception as exc:
            print("Detection worker error:", exc)
            self.last_predictions = []
        finally:
            self.pending_future = None
            self.analysis_running = False

    def _log_prediction(self, track_id, prediction):

        conf = prediction.get("confidence")
        if not isinstance(conf, (int, float)) or conf < self.confidence_threshold:
            return

        now = datetime.now()
        second_key = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.last_logged_second.get(track_id) == second_key:
            return

        self.last_logged_second[track_id] = second_key
        age = prediction.get("age", "N/A")
        emotion = prediction.get("dominant_emotion", "N/A")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow([now.isoformat(), track_id, age, emotion, round(conf, 2)])

        record = {
            "timestamp": now.isoformat(),
            "track_id": track_id,
            "age": age,
            "emotion": emotion,
            "confidence": round(conf, 2),
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")

    def export_summary(self):

        summary_path = os.path.join(self.logs_dir, "summary.json")
        totals = {}
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8") as fp:
                rows = list(csv.DictReader(fp))
            totals["detections"] = len(rows)
            totals["last_timestamp"] = rows[-1]["timestamp"] if rows else None
        else:
            totals["detections"] = 0
            totals["last_timestamp"] = None

        totals["generated_at"] = datetime.now().isoformat()
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(totals, fp, indent=2)

        self.status_label.setText(f"Status: Summary exported to {summary_path}")

    def update_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            self.status_label.setText("Status: No camera frame")
            return

        self.frame_count += 1

        if self.frame_count % self.detection_interval == 0:
            self.last_faces = self._detect_faces(frame)
            tracked = self.tracker.update(self.last_faces)
            self.last_track_ids = [tid for tid, _ in tracked]
            self.last_faces = [box for _, box in tracked]

        self._collect_analysis_result()

        if self.frame_count % self.analysis_interval == 0:
            self._submit_analysis(frame)

        self.smoother.cleanup(self.last_track_ids)

        for index, (x, y, w, h) in enumerate(self.last_faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            track_id = self.last_track_ids[index] if index < len(self.last_track_ids) else index + 1

            prediction = None
            if index < len(self.last_predictions):
                prediction = self.smoother.update(track_id, self.last_predictions[index])

            if prediction:
                age = prediction.get("age", "N/A")
                emotion = prediction.get("dominant_emotion", "N/A")
                confidence = prediction.get("confidence")

                label = f"ID:{track_id} Age:{age} Mood:{emotion}"
                if isinstance(confidence, (int, float)) and confidence >= self.confidence_threshold:
                    label += f" ({confidence:.0f}%)"
                    self._log_prediction(track_id, prediction)

                text_y = y - 10 if y > 25 else y + h + 20
                cv2.putText(
                    frame,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

        status = "Analyzing" if self.analysis_running else "Idle"
        self.status_label.setText(
            f"Status: {status} | Faces: {len(self.last_faces)} | Detector: {self.detector_combo.currentText()}"
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):

        self.stop_camera()
        if self.mp_face is not None:
            self.mp_face.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = AICamera()

    window.show()

    sys.exit(app.exec())   