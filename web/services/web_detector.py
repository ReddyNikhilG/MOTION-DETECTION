import base64
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

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
                    "age": "N/A",
                    "dominant_emotion": "N/A",
                    "confidence": None,
                }

            record = {
                "box": {"x": x, "y": y, "w": w, "h": h},
                "age": prediction.get("age", "N/A"),
                "emotion": prediction.get("dominant_emotion", "N/A"),
                "confidence": prediction.get("confidence"),
            }
            items.append(record)

        latency_ms = int((time.time() - started) * 1000)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "faces": items,
            "face_count": len(items),
            "latency_ms": latency_ms,
        }
        self._write_log(payload)
        return payload

    def _write_log(self, payload):
        with open(self.log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\\n")
