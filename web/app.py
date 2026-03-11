import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from flask_socketio import SocketIO, emit
from werkzeug.security import check_password_hash, generate_password_hash


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web.models import DetectionEvent, User, WorkspaceSetting, db
from web.services.inference_service import InferenceService

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(ROOT_DIR, 'webapp.db')}")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

inference_service = InferenceService(LOGS_DIR, workers=2)
metrics = {
    "requests_total": 0,
    "analyze_total": 0,
    "analyze_errors": 0,
    "latency_total": 0,
}

RATE_LIMIT_WINDOW_SEC = 3
RATE_LIMIT_MAX_ANALYZE = 6
_analyze_hits = defaultdict(deque)
_analyze_hits_lock = threading.Lock()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.before_request
def count_requests():
    metrics["requests_total"] += 1


def create_app():
    with app.app_context():
        db.create_all()
    return app


def _save_detection_event(user_id, payload):
    row = DetectionEvent(
        user_id=user_id,
        face_count=int(payload.get("face_count", 0)),
        latency_ms=int(payload.get("latency_ms", 0)),
        payload_json=json.dumps(payload),
    )
    db.session.add(row)
    db.session.commit()


def _consume_analyze_slot(user_id):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC

    with _analyze_hits_lock:
        q = _analyze_hits[user_id]
        while q and q[0] < window_start:
            q.popleft()

        if len(q) >= RATE_LIMIT_MAX_ANALYZE:
            retry_after_ms = max(50, int((q[0] + RATE_LIMIT_WINDOW_SEC - now) * 1000))
            return False, retry_after_ms

        q.append(now)

    return True, 0


@app.get("/")
@login_required
def index():
    return render_template("index.html", username=current_user.username)


@app.get("/analytics")
@login_required
def analytics_page():
    return render_template("analytics.html", username=current_user.username)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("index"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html", error=None)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not username or not email or len(password) < 6:
            return render_template("register.html", error="Provide valid username, email and password (>=6 chars)")

        if User.query.filter((User.username == username) | (User.email == email)).first():
            return render_template("register.html", error="Username or email already exists")

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("index"))

    return render_template("register.html", error=None)


@app.get("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


@app.get("/api/metrics")
def metrics_api():
    avg_latency = int(metrics["latency_total"] / metrics["analyze_total"]) if metrics["analyze_total"] else 0
    return jsonify(
        {
            "requests_total": metrics["requests_total"],
            "analyze_total": metrics["analyze_total"],
            "analyze_errors": metrics["analyze_errors"],
            "avg_latency_ms": avg_latency,
        }
    )


@app.post("/api/analyze")
@login_required
def analyze():
    allowed, retry_after_ms = _consume_analyze_slot(current_user.id)
    if not allowed:
        return jsonify({"error": "Rate limit exceeded", "retry_after_ms": retry_after_ms}), 429

    body = request.get_json(silent=True) or {}
    image_data = body.get("image")
    if not image_data:
        return jsonify({"error": "Missing image field"}), 400

    started = time.time()
    result = inference_service.analyze_image_data(image_data)
    if result is None:
        metrics["analyze_errors"] += 1
        return jsonify({"error": "Unable to decode image"}), 400

    elapsed = int((time.time() - started) * 1000)
    result["latency_ms"] = elapsed
    metrics["analyze_total"] += 1
    metrics["latency_total"] += elapsed
    _save_detection_event(current_user.id, result)
    return jsonify(result)


@app.get("/api/workspace")
@login_required
def get_workspace():
    rows = WorkspaceSetting.query.filter_by(user_id=current_user.id).all()
    data = {row.setting_key: json.loads(row.setting_value) for row in rows}
    return jsonify(data)


@app.post("/api/workspace")
@login_required
def set_workspace():
    body = request.get_json(silent=True) or {}
    for key, value in body.items():
        row = WorkspaceSetting.query.filter_by(user_id=current_user.id, setting_key=key).first()
        if not row:
            row = WorkspaceSetting(user_id=current_user.id, setting_key=key, setting_value=json.dumps(value))
            db.session.add(row)
        else:
            row.setting_value = json.dumps(value)

    db.session.commit()
    return jsonify({"status": "saved"})


@app.get("/api/analytics")
@login_required
def analytics_api():
    range_key = (request.args.get("range") or "all").strip().lower()
    start_date_raw = (request.args.get("start") or "").strip()
    end_date_raw = (request.args.get("end") or "").strip()

    now = datetime.utcnow()
    query = DetectionEvent.query.filter_by(user_id=current_user.id)

    if range_key == "today":
        start = datetime(now.year, now.month, now.day)
        query = query.filter(DetectionEvent.created_at >= start)
    elif range_key == "week":
        query = query.filter(DetectionEvent.created_at >= now - timedelta(days=7))
    elif range_key == "custom":
        try:
            if start_date_raw:
                start = datetime.fromisoformat(start_date_raw)
                query = query.filter(DetectionEvent.created_at >= start)
            if end_date_raw:
                end = datetime.fromisoformat(end_date_raw) + timedelta(days=1)
                query = query.filter(DetectionEvent.created_at < end)
        except ValueError:
            return jsonify({"error": "Invalid start/end date format (expected YYYY-MM-DD)"}), 400

    rows = query.order_by(DetectionEvent.id.desc()).limit(1500).all()
    if not rows:
        return jsonify(
            {
                "samples": 0,
                "avg_latency_ms": 0,
                "peak_faces": 0,
                "top_emotions": [],
                "trend": [],
                "applied_range": range_key,
            }
        )

    emotions = []
    total_latency = 0
    peak_faces = 0
    for row in rows:
        total_latency += row.latency_ms
        peak_faces = max(peak_faces, row.face_count)
        payload = json.loads(row.payload_json)
        for face in payload.get("faces", []):
            emotion = face.get("emotion")
            if emotion and emotion != "N/A":
                emotions.append(emotion)

    top = Counter(emotions).most_common(6)
    trend_rows = list(reversed(rows[:180]))
    trend = [
        {
            "timestamp": row.created_at.isoformat(),
            "latency_ms": row.latency_ms,
            "face_count": row.face_count,
        }
        for row in trend_rows
    ]
    return jsonify(
        {
            "samples": len(rows),
            "avg_latency_ms": int(total_latency / len(rows)),
            "peak_faces": peak_faces,
            "top_emotions": [{"emotion": k, "count": v} for k, v in top],
            "trend": trend,
            "applied_range": range_key,
        }
    )


@socketio.on("connect")
def socket_connect():
    if not current_user.is_authenticated:
        emit("analyze_result", {"error": "Authentication required"})


@socketio.on("analyze_frame")
def analyze_frame_ws(payload):
    if not current_user.is_authenticated:
        emit("analyze_result", {"error": "Authentication required"})
        return

    allowed, retry_after_ms = _consume_analyze_slot(current_user.id)
    if not allowed:
        emit("analyze_result", {"error": "Rate limit exceeded", "retry_after_ms": retry_after_ms})
        return

    image_data = (payload or {}).get("image")
    if not image_data:
        emit("analyze_result", {"error": "Missing image"})
        return

    started = time.time()
    result = inference_service.analyze_image_data(image_data)
    if result is None:
        metrics["analyze_errors"] += 1
        emit("analyze_result", {"error": "Unable to decode image"})
        return

    elapsed = int((time.time() - started) * 1000)
    result["latency_ms"] = elapsed
    metrics["analyze_total"] += 1
    metrics["latency_total"] += elapsed
    _save_detection_event(current_user.id, result)
    emit("analyze_result", result)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=debug,
        allow_unsafe_werkzeug=True,
    )
