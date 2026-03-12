import os
import io
import json
import time
import tempfile
import urllib.request
from datetime import datetime, timedelta
from collections import Counter

try:
    import cv2
    CV2_IMPORT_ERROR = None
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)
import numpy as np
import pandas as pd
import streamlit as st

try:
    import mediapipe as mp
    MP_IMPORT_ERROR = None
except Exception as exc:
    mp = None
    MP_IMPORT_ERROR = str(exc)

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_TASKS_IMPORT_ERROR = None
except Exception as exc:
    mp_python = None
    mp_vision = None
    MP_TASKS_IMPORT_ERROR = str(exc)

from ai_detector import AIDetector

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOGS_DIR, "streamlit_detections.jsonl")
MODELS_DIR = os.path.join(os.path.dirname(__file__), ".models")
os.makedirs(MODELS_DIR, exist_ok=True)
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

FACE_CASCADE = (
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cv2 is not None
    else None
)
EYE_CASCADE = (
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    if cv2 is not None
    else None
)
MP_POSE_AVAILABLE = bool(mp is not None and mp_python is not None and mp_vision is not None)

NOSE_IDX = 0
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_ELBOW_IDX = 13
RIGHT_ELBOW_IDX = 14
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
LEFT_KNEE_IDX = 25
RIGHT_KNEE_IDX = 26
LEFT_ANKLE_IDX = 27
RIGHT_ANKLE_IDX = 28


@st.cache_resource
def get_detector():
    return AIDetector()


def get_pose_status_message():
    if MP_IMPORT_ERROR:
        return f"MediaPipe import failed: {MP_IMPORT_ERROR}"
    if MP_TASKS_IMPORT_ERROR:
        return f"MediaPipe Tasks import failed: {MP_TASKS_IMPORT_ERROR}"
    return "Pose model unavailable in this deployment."


def ensure_pose_model():
    if os.path.exists(POSE_MODEL_PATH):
        return POSE_MODEL_PATH

    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    return POSE_MODEL_PATH


@st.cache_resource
def get_pose_detector():
    if not MP_POSE_AVAILABLE:
        return None
    try:
        model_path = ensure_pose_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)
    except Exception:
        return None


def get_pose_runtime_message(pose_detector):
    if pose_detector is not None:
        return None
    return get_pose_status_message()


def format_emotion_label(face):
    emotion = face.get("emotion")
    if emotion and emotion != "N/A":
        return emotion
    return "Emotion unavailable"


def format_confidence_label(face):
    confidence = face.get("confidence")
    if isinstance(confidence, (int, float)):
        return f"{confidence:.1f}%"
    return "Unavailable"


def render_runtime_banner(detector, pose_detector):
    emotion_state = "Ready" if detector.enabled else "Limited"
    motion_state = "Ready" if pose_detector is not None else "Limited"
    emotion_detail = "DeepFace emotion model online" if detector.enabled else (detector.disabled_reason or "Emotion model not installed in this deployment")
    motion_detail = "Pose landmarker online" if pose_detector is not None else get_pose_status_message()
    st.markdown(
        f"""
        <div class="runtime-strip">
            <div class="runtime-card">
                <div class="runtime-label">Emotion Engine</div>
                <div class="runtime-value">{emotion_state}</div>
                <div class="runtime-detail">{emotion_detail}</div>
            </div>
            <div class="runtime-card">
                <div class="runtime-label">Motion Engine</div>
                <div class="runtime-value">{motion_state}</div>
                <div class="runtime-detail">{motion_detail}</div>
            </div>
            <div class="runtime-card">
                <div class="runtime-label">Deployment</div>
                <div class="runtime-value">Streamlit</div>
                <div class="runtime-detail">Camera, uploads, video analytics, and export tools</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classify_motion(landmarks, h, w):
    """Classify the person's motion/pose from MediaPipe landmarks."""
    motions = []

    nose = landmarks[NOSE_IDX]
    l_wrist = landmarks[LEFT_WRIST_IDX]
    r_wrist = landmarks[RIGHT_WRIST_IDX]
    l_shoulder = landmarks[LEFT_SHOULDER_IDX]
    r_shoulder = landmarks[RIGHT_SHOULDER_IDX]
    l_elbow = landmarks[LEFT_ELBOW_IDX]
    r_elbow = landmarks[RIGHT_ELBOW_IDX]
    l_hip = landmarks[LEFT_HIP_IDX]
    r_hip = landmarks[RIGHT_HIP_IDX]
    l_knee = landmarks[LEFT_KNEE_IDX]
    r_knee = landmarks[RIGHT_KNEE_IDX]
    l_ankle = landmarks[LEFT_ANKLE_IDX]
    r_ankle = landmarks[RIGHT_ANKLE_IDX]

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
    if pose_detector is None or cv2 is None or not MP_POSE_AVAILABLE:
        return None, None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = pose_detector.detect(mp_image)
    if not results.pose_landmarks:
        return None, None
    h, w = frame.shape[:2]
    first_pose = results.pose_landmarks[0]
    motions = classify_motion(first_pose, h, w)
    return motions, first_pose


def detect_faces(frame, min_face_size=60):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=7, minSize=(int(min_face_size), int(min_face_size))
    )
    if faces is None or not len(faces):
        return []

    frame_area = gray.shape[0] * gray.shape[1]
    validated_faces = []

    for x, y, w, h in faces:
        box_area = w * h
        if box_area < max(int(frame_area * 0.015), int(min_face_size) * int(min_face_size)):
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.65 or aspect_ratio > 1.45:
            continue

        if EYE_CASCADE is not None:
            face_roi = gray[y:y + h, x:x + w]
            upper_face = face_roi[: max(1, h // 2), :]
            eyes = EYE_CASCADE.detectMultiScale(
                upper_face,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(max(8, w // 10), max(8, h // 10)),
            )
            if eyes is None or len(eyes) < 1:
                continue

        validated_faces.append((int(x), int(y), int(w), int(h)))

    return validated_faces


def analyze_frame(frame, detector, pose_detector, min_confidence=0, min_face_size=60):
    started = time.time()
    faces = detect_faces(frame, min_face_size=min_face_size)
    items = []

    for x, y, w, h in faces:
        crop = frame[y : y + h, x : x + w]
        prediction = detector.analyze(crop) if crop.size else None
        if not prediction:
            prediction = {"dominant_emotion": "N/A", "confidence": None}

        conf = prediction.get("confidence")
        # Filter by minimum confidence threshold
        if min_confidence > 0 and conf is not None and conf < min_confidence:
            continue

        items.append({
            "box": {"x": x, "y": y, "w": w, "h": h},
            "emotion": prediction.get("dominant_emotion", "N/A"),
            "confidence": conf,
        })

    # Detect body motion / pose
    motions, pose_lm = detect_pose(frame, pose_detector)
    if motions:
        motion_list = motions
    elif items:
        if pose_detector is None:
            motion_list = ["Face detected (pose unavailable)"]
        else:
            motion_list = ["Person detected"]
    else:
        if pose_detector is None:
            motion_list = ["No face detected (pose unavailable)"]
        else:
            motion_list = ["No person detected"]

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

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(71, 208, 255, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(90, 162, 255, 0.14), transparent 22%),
            linear-gradient(180deg, #06101d 0%, #0b1727 52%, #09111e 100%);
    }
    .hero-shell {
        background: linear-gradient(135deg, rgba(8, 25, 42, 0.92), rgba(16, 44, 73, 0.86));
        border: 1px solid rgba(90, 162, 255, 0.22);
        border-radius: 22px;
        padding: 22px 24px;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.24);
        margin-bottom: 16px;
    }
    .hero-title {
        color: #eff7ff;
        font-size: 34px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .hero-copy {
        color: #9eb9d4;
        font-size: 15px;
        max-width: 720px;
    }
    .runtime-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 14px 0 18px;
    }
    .runtime-card {
        background: linear-gradient(145deg, rgba(10, 25, 42, 0.96), rgba(16, 35, 58, 0.92));
        border: 1px solid rgba(90, 162, 255, 0.2);
        border-radius: 18px;
        padding: 16px 18px;
        min-height: 112px;
    }
    .runtime-label {
        color: #7f9abb;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .runtime-value {
        color: #f3fbff;
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .runtime-detail {
        color: #9eb9d4;
        font-size: 13px;
        line-height: 1.45;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(90, 162, 255, 0.3);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetric"] label { color: #8fa8cc !important; font-size: 14px !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #47d0ff !important; font-size: 28px !important;
    }
    .face-card {
        background: linear-gradient(135deg, #0b1929 0%, #132744 100%);
        border: 1px solid rgba(90, 162, 255, 0.25);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 8px;
    }
    .motion-badge {
        display: inline-block;
        background: linear-gradient(120deg, #5aa2ff, #47d0ff);
        color: #031025;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        margin: 3px 4px;
    }
    .section-header {
        background: linear-gradient(90deg, rgba(90,162,255,0.15), transparent);
        border-left: 3px solid #5aa2ff;
        padding: 8px 16px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0 12px;
    }
    .success-box {
        background: linear-gradient(135deg, #0a2a1a 0%, #0d3320 100%);
        border: 1px solid rgba(50, 205, 50, 0.3);
        border-radius: 10px;
        padding: 12px 16px;
    }
    .compact-note {
        background: rgba(9, 20, 34, 0.88);
        border: 1px solid rgba(90, 162, 255, 0.16);
        border-radius: 14px;
        padding: 14px 16px;
        color: #aac1d9;
        margin: 10px 0 18px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-title">FacePulse Live</div>
        <div class="hero-copy">
            Analyze faces, posture cues, and sampled video frames from a single Streamlit workspace.
            The app adapts to the deployment environment and keeps face analytics available even when optional ML backends are limited.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if cv2 is None:
    st.error(
        "OpenCV could not be loaded in this deployment. "
        f"Underlying error: {CV2_IMPORT_ERROR}"
    )
    st.info("The server is missing native OpenCV runtime libraries. Please redeploy after updating system packages.")
    st.stop()

# ── Sidebar settings ─────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Mode", ["📷 Camera Capture", "🖼️ Upload Image", "🎬 Video Analysis", "📊 Analytics"])

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Settings")
min_confidence = st.sidebar.slider(
    "Min Confidence (%)", 0, 100, 0, 5,
    help="Filter out detections below this confidence threshold"
)
min_face_size = st.sidebar.slider(
    "Min Face Size (px)", 30, 200, 60, 10,
    help="Minimum pixel size for face detection"
)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Detection Logs"):
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    st.sidebar.success("Logs cleared!")

detector = get_detector()
pose_detector = get_pose_detector()
render_runtime_banner(detector, pose_detector)

if not detector.enabled or pose_detector is None:
    fallback_messages = []
    if not detector.enabled:
        fallback_messages.append(
            "Emotion analysis is currently limited in this deployment. "
            "Select Streamlit Cloud Python 3.12 to enable the DeepFace backend."
        )
    if pose_detector is None:
        fallback_messages.append(
            "Pose analysis is currently limited. The app will still render face detection and analytics, "
            "and will use generic motion labels where posture landmarks are unavailable."
        )
    st.markdown(
        f"<div class='compact-note'>{' '.join(fallback_messages)}</div>",
        unsafe_allow_html=True,
    )


def _render_result(frame, result):
    """Shared result renderer for camera/upload/video modes."""
    annotated = draw_detections(frame, result["faces"], result["motions"])
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(annotated_rgb, caption="Detection Results", use_container_width=True)

        # Download annotated image
        success, buf = cv2.imencode(".png", annotated)
        if success:
            st.download_button(
                "⬇️ Download Annotated Image",
                data=buf.tobytes(),
                file_name=f"facepulse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
            )

    with c2:
        st.metric("Faces Detected", result["face_count"])
        st.metric("Latency", f"{result['latency_ms']} ms")
        st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))

        st.markdown('<div class="section-header"><b>🏃 Detected Motions</b></div>', unsafe_allow_html=True)
        motion_html = "".join(f'<span class="motion-badge">{m}</span>' for m in result["motions"])
        st.markdown(motion_html, unsafe_allow_html=True)

        if result["faces"]:
            st.markdown('<div class="section-header"><b>😊 Face Details</b></div>', unsafe_allow_html=True)
            for i, face in enumerate(result["faces"], 1):
                conf = format_confidence_label(face)
                emotion_label = format_emotion_label(face)
                st.markdown(f"""
                <div class="face-card">
                    <b>Face {i}</b><br>
                    🎭 Emotion: <b>{emotion_label}</b><br>
                    📊 Confidence: <b>{conf}</b>
                </div>
                """, unsafe_allow_html=True)


# ── Camera Capture ───────────────────────────────────────────
if mode == "📷 Camera Capture":
    st.markdown('<div class="section-header"><b>📷 Capture from Webcam</b></div>', unsafe_allow_html=True)
    st.info("Click the camera button to take a photo, then it will be analyzed automatically.")

    img_data = st.camera_input("Take a photo")

    if img_data is not None:
        file_bytes = np.frombuffer(img_data.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Could not decode the camera image. Please try again.")
        else:
            with st.spinner("🔍 Analyzing faces and motion..."):
                result = analyze_frame(frame, detector, pose_detector, min_confidence, min_face_size)
            _render_result(frame, result)

# ── Upload Image ─────────────────────────────────────────────
elif mode == "🖼️ Upload Image":
    st.markdown('<div class="section-header"><b>🖼️ Upload an Image for Analysis</b></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose one or more images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for idx, uploaded in enumerate(uploaded_files):
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                st.error(f"Could not decode **{uploaded.name}**. Skipping.")
                continue

            if len(uploaded_files) > 1:
                st.markdown(f"---\n### Image {idx + 1}: {uploaded.name}")

            with st.spinner(f"🔍 Analyzing {uploaded.name}..."):
                result = analyze_frame(frame, detector, pose_detector, min_confidence, min_face_size)
            _render_result(frame, result)

# ── Video Analysis ───────────────────────────────────────────
elif mode == "🎬 Video Analysis":
    st.markdown('<div class="section-header"><b>🎬 Upload a Video for Analysis</b></div>', unsafe_allow_html=True)
    st.info("Upload a video file to analyze faces and motion in sampled frames.")

    video_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv", "webm"])

    frame_interval = st.slider("Analyze every N-th frame", 5, 60, 15, 5,
                                help="Lower = more detail but slower processing")

    if video_file is not None:
        # Write to temp file for OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()
        tfile_path = tfile.name
        tfile.close()

        cap = cv2.VideoCapture(tfile_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        if total_frames <= 0:
            st.error("Could not read the video file.")
        else:
            frames_to_analyze = list(range(0, total_frames, frame_interval))
            st.write(f"**{total_frames}** total frames at **{fps:.0f}** FPS — analyzing **{len(frames_to_analyze)}** sampled frames")

            progress = st.progress(0, text="Analyzing video...")
            video_results = []

            for i, frame_no in enumerate(frames_to_analyze):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                if not ret:
                    continue

                result = analyze_frame(frame, detector, pose_detector, min_confidence, min_face_size)
                result["frame_no"] = frame_no
                result["time_sec"] = round(frame_no / fps, 2)
                video_results.append((frame, result))

                progress.progress((i + 1) / len(frames_to_analyze),
                                  text=f"Analyzing frame {frame_no}/{total_frames}...")

            cap.release()
            try:
                os.unlink(tfile_path)
            except OSError:
                pass

            progress.empty()

            if not video_results:
                st.warning("No frames could be analyzed from this video.")
            else:
                # Summary metrics
                st.markdown('<div class="section-header"><b>📊 Video Summary</b></div>', unsafe_allow_html=True)

                all_face_counts = [r["face_count"] for _, r in video_results]
                all_latencies = [r["latency_ms"] for _, r in video_results]
                all_emotions = []
                for _, r in video_results:
                    for face in r["faces"]:
                        emo = face.get("emotion")
                        if emo and emo != "N/A":
                            all_emotions.append(emo)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Frames Analyzed", len(video_results))
                m2.metric("Max Faces", max(all_face_counts) if all_face_counts else 0)
                m3.metric("Avg Latency", f"{sum(all_latencies)/len(all_latencies):.0f} ms" if all_latencies else "N/A")
                m4.metric("Top Emotion", Counter(all_emotions).most_common(1)[0][0] if all_emotions else "N/A")

                # Face count over time chart
                if video_results:
                    chart_df = pd.DataFrame([
                        {"Time (s)": r["time_sec"], "Faces": r["face_count"], "Latency (ms)": r["latency_ms"]}
                        for _, r in video_results
                    ])
                    st.subheader("Faces Over Time")
                    st.line_chart(chart_df.set_index("Time (s)")["Faces"])
                    st.subheader("Latency Over Time")
                    st.line_chart(chart_df.set_index("Time (s)")["Latency (ms)"])

                # Emotion distribution
                if all_emotions:
                    st.subheader("Emotion Distribution")
                    emo_counts = Counter(all_emotions)
                    st.bar_chart(emo_counts)

                # Browse individual frames
                st.markdown('<div class="section-header"><b>🔎 Browse Frames</b></div>', unsafe_allow_html=True)
                frame_idx = st.slider("Select frame", 0, len(video_results) - 1, 0)
                selected_frame, selected_result = video_results[frame_idx]
                st.caption(f"Frame {selected_result['frame_no']} — {selected_result['time_sec']}s")
                _render_result(selected_frame, selected_result)

# ── Analytics ────────────────────────────────────────────────
elif mode == "📊 Analytics":
    st.markdown('<div class="section-header"><b>📊 Detection Analytics</b></div>', unsafe_allow_html=True)

    records = load_logs()

    if not records:
        st.info("No detection data yet. Analyze some images or videos first!")
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
        max_faces = max((r.get("face_count", 0) for r in filtered), default=0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Analyses", len(filtered))
        m2.metric("Total Faces", total_faces)
        m3.metric("Peak Faces", max_faces)
        m4.metric("Avg Latency", f"{avg_latency:.0f} ms")

        # Emotion and motion data
        emotions = []
        all_motions = []
        for r in filtered:
            for face in r.get("faces", []):
                emo = face.get("emotion")
                if emo and emo != "N/A":
                    emotions.append(emo)
            for m in r.get("motions", []):
                if m and m != "No person detected":
                    all_motions.append(m)

        # Two-column charts
        col_a, col_b = st.columns(2)

        with col_a:
            if emotions:
                st.subheader("Emotion Distribution")
                counts = Counter(emotions)
                st.bar_chart(counts)
            else:
                st.info("No emotion data in this range")

        with col_b:
            if all_motions:
                st.subheader("Motion Distribution")
                motion_counts = Counter(all_motions)
                st.bar_chart(motion_counts)
            else:
                st.info("No motion data in this range")

        # Detections over time
        if filtered:
            st.subheader("Detections Over Time")
            time_data = []
            latency_data = []
            for r in filtered:
                try:
                    ts = datetime.fromisoformat(r["timestamp"])
                    time_data.append({"date": ts.date(), "faces": r.get("face_count", 0)})
                    latency_data.append({"date": ts.date(), "latency": r.get("latency_ms", 0)})
                except (ValueError, KeyError):
                    continue

            if time_data:
                tc1, tc2 = st.columns(2)
                with tc1:
                    tdf = pd.DataFrame(time_data)
                    daily = tdf.groupby("date")["faces"].sum()
                    st.caption("Daily Face Count")
                    st.line_chart(daily)
                with tc2:
                    ldf = pd.DataFrame(latency_data)
                    daily_lat = ldf.groupby("date")["latency"].mean()
                    st.caption("Daily Avg Latency (ms)")
                    st.line_chart(daily_lat)

        # Export analytics data
        st.markdown("---")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            csv_data = pd.DataFrame(filtered).to_csv(index=False)
            st.download_button(
                "⬇️ Export as CSV",
                data=csv_data,
                file_name=f"facepulse_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        with export_col2:
            json_data = json.dumps(filtered, indent=2)
            st.download_button(
                "⬇️ Export as JSON",
                data=json_data,
                file_name=f"facepulse_analytics_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )
