import json
import os
import sys
from collections import Counter
from datetime import datetime

from flask import Flask, jsonify, render_template, request


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.web_detector import WebFaceAnalyzer

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

app = Flask(__name__, template_folder="templates", static_folder="static")
analyzer = WebFaceAnalyzer(LOGS_DIR)


def create_app():
    return app


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/analytics")
def analytics_page():
    return render_template("analytics.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


@app.post("/api/analyze")
def analyze():
    body = request.get_json(silent=True) or {}
    image_data = body.get("image")
    if not image_data:
        return jsonify({"error": "Missing image field"}), 400

    frame = analyzer.decode_image(image_data)
    if frame is None:
        return jsonify({"error": "Unable to decode image"}), 400

    result = analyzer.analyze_frame(frame)
    return jsonify(result)


@app.get("/api/analytics")
def analytics_api():
    log_path = os.path.join(LOGS_DIR, "web_detections.jsonl")
    if not os.path.exists(log_path):
        return jsonify(
            {
                "samples": 0,
                "avg_latency_ms": 0,
                "peak_faces": 0,
                "top_emotions": [],
            }
        )

    total_samples = 0
    total_latency = 0
    peak_faces = 0
    emotions = []

    with open(log_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_samples += 1
            total_latency += int(row.get("latency_ms", 0))
            face_count = int(row.get("face_count", 0))
            peak_faces = max(peak_faces, face_count)

            for face in row.get("faces", []):
                emotion = face.get("emotion")
                if emotion and emotion != "N/A":
                    emotions.append(emotion)

    avg_latency = int(total_latency / total_samples) if total_samples else 0
    top = Counter(emotions).most_common(6)

    return jsonify(
        {
            "samples": total_samples,
            "avg_latency_ms": avg_latency,
            "peak_faces": peak_faces,
            "top_emotions": [{"emotion": k, "count": v} for k, v in top],
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
