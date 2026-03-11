import os
import json
import time
from datetime import datetime, timedelta
from collections import Counter

import cv2
import numpy as np
import mediapipe as mp
import streamlit as st

from ai_detector import AIDetector

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOGS_DIR, "streamlit_detections.jsonl")

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


@st.cache_resource
def get_detector():
    return AIDetector()


@st.cache_resource
def get_pose_detector():
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )


def classify_motion(landmarks, h, w):
    """Classify the person's motion/pose from MediaPipe landmarks."""
    motions = []

    nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]
    l_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    r_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    l_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    l_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
    r_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
    l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    l_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
    r_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
    l_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
    r_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]

    mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
    mid_hip_y = (l_hip.y + r_hip.y) / 2
    mid_knee_y = (l_knee.y + r_knee.y) / 2

    # Hands raised: wrists above shoulders
    if l_wrist.y < l_shoulder.y - 0.05 and r_wrist.y < r_shoulder.y - 0.05:
        motions.append("Both Hands Raised")
    elif l_wrist.y < l_shoulder.y - 0.05:
        motions.append("Left Hand Raised")
    elif r_wrist.y < r_shoulder.y - 0.05:
        motions.append("Right Hand Raised")

    # Sitting vs standing: compare hip-to-knee vs shoulder-to-hip ratio
    torso_len = abs(mid_hip_y - mid_shoulder_y)
    thigh_len = abs(mid_knee_y - mid_hip_y)
    if torso_len > 0 and thigh_len / torso_len < 0.5:
        motions.append("Sitting")
    else:
        motions.append("Standing")

    # Leaning: shoulder midpoint offset from hip midpoint
    mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
    mid_hip_x = (l_hip.x + r_hip.x) / 2
    lean = mid_shoulder_x - mid_hip_x
    if lean > 0.06:
        motions.append("Leaning Right")
    elif lean < -0.06:
        motions.append("Leaning Left")

    # Arms crossed: wrists near opposite shoulders
    if (abs(l_wrist.x - r_shoulder.x) < 0.08 and abs(l_wrist.y - r_shoulder.y) < 0.10
            and abs(r_wrist.x - l_shoulder.x) < 0.08 and abs(r_wrist.y - l_shoulder.y) < 0.10):
        motions.append("Arms Crossed")

    # Waving: one wrist above head
    if l_wrist.y < nose.y - 0.10 or r_wrist.y < nose.y - 0.10:
        motions.append("Waving / Hand Above Head")

    return motions if motions else ["Idle"]


def detect_pose(frame, pose_detector):
    """Run MediaPipe Pose on the frame and return detected motions."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb)
    if not results.pose_landmarks:
        return None, None
    h, w = frame.shape[:2]
    motions = classify_motion(results.pose_landmarks.landmark, h, w)
    return motions, results.pose_landmarks


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return [tuple(map(int, box)) for box in faces] if faces is not None and len(faces) else []


def analyze_frame(frame, detector, pose_detector):
    started = time.time()
    faces = detect_faces(frame)
    items = []

    for x, y, w, h in faces:
        crop = frame[y : y + h, x : x + w]
        prediction = detector.analyze(crop) if crop.size else None
        if not prediction:
            prediction = {"dominant_emotion": "N/A", "confidence": None}

        items.append({
            "box": {"x": x, "y": y, "w": w, "h": h},
            "emotion": prediction.get("dominant_emotion", "N/A"),
            "confidence": prediction.get("confidence"),
        })

    # Detect body motion / pose
    motions, pose_lm = detect_pose(frame, pose_detector)
    motion_list = motions if motions else ["No person detected"]

    latency_ms = int((time.time() - started) * 1000)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "faces": items,
        "face_count": len(items),
        "motions": motion_list,
        "latency_ms": latency_ms,
    }
    _write_log(payload)
    return payload


def _write_log(payload):
    with open(LOG_PATH, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload) + "\n")


def draw_detections(frame, faces, motions):
    annotated = frame.copy()
    h_frame, w_frame = annotated.shape[:2]

    for face in faces:
        b = face["box"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = face["emotion"]
        if face["confidence"] is not None:
            label += f" ({face['confidence']:.0f}%)"
        cv2.putText(
            annotated, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    # Draw motion labels at top-left
    for i, motion in enumerate(motions):
        cv2.putText(
            annotated, f"Motion: {motion}", (10, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )
    return annotated


def load_logs():
    records = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


# ── Streamlit UI ─────────────────────────────────────────────

st.set_page_config(page_title="FacePulse Live", page_icon="🎯", layout="wide")
st.title("🎯 FacePulse Live — Face & Motion Detection")

mode = st.sidebar.radio("Mode", ["📷 Camera Capture", "🖼️ Upload Image", "📊 Analytics"])

detector = get_detector()
pose_detector = get_pose_detector()

# ── Camera Capture ───────────────────────────────────────────
if mode == "📷 Camera Capture":
    st.subheader("Capture from Webcam")
    st.info("Click the camera button to take a photo, then it will be analyzed automatically.")

    img_data = st.camera_input("Take a photo")

    if img_data is not None:
        file_bytes = np.frombuffer(img_data.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Analyzing..."):
            result = analyze_frame(frame, detector, pose_detector)

        annotated = draw_detections(frame, result["faces"], result["motions"])
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detection Results", width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("Faces Detected", result["face_count"])
        col2.metric("Latency", f"{result['latency_ms']} ms")
        col3.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))

        st.subheader("Detected Motions")
        for motion in result["motions"]:
            st.write(f"- **{motion}**")

        if result["faces"]:
            st.subheader("Face Details")
            for i, face in enumerate(result["faces"], 1):
                with st.expander(f"Face {i}"):
                    fc1, fc2 = st.columns(2)
                    fc1.write(f"**Emotion:** {face['emotion']}")
                    conf = f"{face['confidence']:.1f}%" if face['confidence'] else "N/A"
                    fc2.write(f"**Confidence:** {conf}")

# ── Upload Image ─────────────────────────────────────────────
elif mode == "🖼️ Upload Image":
    st.subheader("Upload an Image for Analysis")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Could not decode the image. Please try a different file.")
        else:
            with st.spinner("Analyzing..."):
                result = analyze_frame(frame, detector, pose_detector)

            annotated = draw_detections(frame, result["faces"], result["motions"])
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(annotated_rgb, caption="Detection Results", width="stretch")
            with c2:
                st.metric("Faces Detected", result["face_count"])
                st.metric("Latency", f"{result['latency_ms']} ms")

                st.markdown("---\n**Motions**")
                for motion in result["motions"]:
                    st.write(f"- {motion}")

                if result["faces"]:
                    for i, face in enumerate(result["faces"], 1):
                        st.markdown(f"---\n**Face {i}**")
                        st.write(f"Emotion: {face['emotion']}")
                        conf = f"{face['confidence']:.1f}%" if face['confidence'] else "N/A"
                        st.write(f"Confidence: {conf}")

# ── Analytics ────────────────────────────────────────────────
elif mode == "📊 Analytics":
    st.subheader("Detection Analytics")

    records = load_logs()

    if not records:
        st.info("No detection data yet. Analyze some images first!")
    else:
        # Time filter
        filter_opt = st.sidebar.selectbox("Time Range", ["All", "Today", "Last 7 Days", "Last 30 Days"])
        now = datetime.now()
        if filter_opt == "Today":
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif filter_opt == "Last 7 Days":
            cutoff = now - timedelta(days=7)
        elif filter_opt == "Last 30 Days":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = None

        filtered = []
        for r in records:
            try:
                ts = datetime.fromisoformat(r["timestamp"])
                if cutoff is None or ts >= cutoff:
                    filtered.append(r)
            except (ValueError, KeyError):
                continue

        st.write(f"**{len(filtered)}** detections in selected range")

        # Summary metrics
        total_faces = sum(r.get("face_count", 0) for r in filtered)
        avg_latency = (
            sum(r.get("latency_ms", 0) for r in filtered) / len(filtered)
            if filtered else 0
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Analyses", len(filtered))
        m2.metric("Total Faces Detected", total_faces)
        m3.metric("Avg Latency", f"{avg_latency:.0f} ms")

        # Emotion distribution
        emotions = []
        for r in filtered:
            for face in r.get("faces", []):
                emo = face.get("emotion")
                if emo and emo != "N/A":
                    emotions.append(emo)

        if emotions:
            st.subheader("Emotion Distribution")
            counts = Counter(emotions)
            st.bar_chart(counts)

        # Motion distribution
        all_motions = []
        for r in filtered:
            for m in r.get("motions", []):
                if m and m != "No person detected":
                    all_motions.append(m)

        if all_motions:
            st.subheader("Motion Distribution")
            motion_counts = Counter(all_motions)
            st.bar_chart(motion_counts)

        # Detections over time
        if filtered:
            st.subheader("Detections Over Time")
            import pandas as pd
            time_data = []
            for r in filtered:
                try:
                    ts = datetime.fromisoformat(r["timestamp"])
                    time_data.append({"date": ts.date(), "faces": r.get("face_count", 0)})
                except (ValueError, KeyError):
                    continue
            if time_data:
                tdf = pd.DataFrame(time_data)
                daily = tdf.groupby("date")["faces"].sum()
                st.line_chart(daily)
