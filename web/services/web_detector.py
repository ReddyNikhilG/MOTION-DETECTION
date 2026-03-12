import base64
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
try:
    import mediapipe as mp
except Exception:
    mp = None

from ai_detector import AIDetector


class WebFaceAnalyzer:
    def __init__(self, logs_dir):
        self.detector = AIDetector()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        self.log_path = os.path.join(self.logs_dir, "web_detections.jsonl")
        self.pose_detector = None
        if mp is not None and getattr(mp, "solutions", None) and getattr(mp.solutions, "pose", None):
            self.pose_detector = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
            )

    def detect_motion(self, frame, face_count):
        if self.pose_detector is None:
            if face_count > 0:
                return ["Face detected (pose unavailable)"]
            return ["No person detected (pose unavailable)"]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose_detector.process(rgb)
        if not result.pose_landmarks:
            if face_count > 0:
                return ["Person detected"]
            return ["No person detected"]

        lm = result.pose_landmarks.landmark
        pose = mp.solutions.pose.PoseLandmark
        nose = lm[pose.NOSE]
        l_wrist = lm[pose.LEFT_WRIST]
        r_wrist = lm[pose.RIGHT_WRIST]
        l_shoulder = lm[pose.LEFT_SHOULDER]
        r_shoulder = lm[pose.RIGHT_SHOULDER]
        l_hip = lm[pose.LEFT_HIP]
        r_hip = lm[pose.RIGHT_HIP]
        l_knee = lm[pose.LEFT_KNEE]
        r_knee = lm[pose.RIGHT_KNEE]

        motions = []
        if l_wrist.y < l_shoulder.y - 0.05 and r_wrist.y < r_shoulder.y - 0.05:
            motions.append("Both Hands Raised")
        elif l_wrist.y < l_shoulder.y - 0.05:
            motions.append("Left Hand Raised")
        elif r_wrist.y < r_shoulder.y - 0.05:
            motions.append("Right Hand Raised")

        torso_len = abs(((l_hip.y + r_hip.y) / 2) - ((l_shoulder.y + r_shoulder.y) / 2))
        thigh_len = abs(((l_knee.y + r_knee.y) / 2) - ((l_hip.y + r_hip.y) / 2))
        if torso_len > 0 and thigh_len / torso_len < 0.5:
            motions.append("Sitting")
        else:
            motions.append("Standing")

        mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
        mid_hip_x = (l_hip.x + r_hip.x) / 2
        lean = mid_shoulder_x - mid_hip_x
        if lean > 0.06:
            motions.append("Leaning Right")
        elif lean < -0.06:
            motions.append("Leaning Left")

        if l_wrist.y < nose.y - 0.10 or r_wrist.y < nose.y - 0.10:
            motions.append("Waving / Hand Above Head")

        return motions if motions else ["Idle"]

    @staticmethod
    def decode_image(data_url):
        if "," in data_url:
            _, data = data_url.split(",", 1)
        else:
            data = data_url

        img_bytes = base64.b64decode(data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if faces is None:
            return []
        return [tuple(map(int, box)) for box in faces]

    def analyze_frame(self, frame):
        started = time.time()
        faces = self.detect_faces(frame)

        items = []
        for x, y, w, h in faces:
            crop = frame[y : y + h, x : x + w]
            prediction = self.detector.analyze(crop) if crop.size else None
            if not prediction:
                prediction = {
                    "dominant_emotion": "N/A",
                    "confidence": None,
                }

            record = {
                "box": {"x": x, "y": y, "w": w, "h": h},
                "emotion": prediction.get("dominant_emotion", "N/A"),
                "confidence": prediction.get("confidence"),
            }
            items.append(record)

        latency_ms = int((time.time() - started) * 1000)
        motions = self.detect_motion(frame, len(items))
        payload = {
            "timestamp": datetime.now().isoformat(),
            "faces": items,
            "face_count": len(items),
            "motions": motions,
            "latency_ms": latency_ms,
        }
        self._write_log(payload)
        return payload

    def _write_log(self, payload):
        with open(self.log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")
