# 🎥 Motion Detection & Face Analytics Platform

Real-time **motion detection and face analytics system** built using **Python, OpenCV, DeepFace, Streamlit, and Flask**.

The project provides multiple interfaces:

- 🌐 **FacePulse Live Web Application**
- 📊 **Streamlit Analytics Dashboard**
- 🎥 **Real-time Motion & Face Detection Engine**
- 🐳 **Docker-ready backend using Gunicorn**

The system detects faces from webcam streams, performs emotion analysis, tracks motion activity, and stores analytics data for visualization.

---

# 🚀 Features

### 🎥 Computer Vision
- Real-time webcam motion detection
- Multi-face detection with bounding boxes
- Face tracking across frames
- Emotion analysis using DeepFace
- Motion state detection (moving / idle)
- Confidence threshold filtering

### 🌐 Web Application (FacePulse Live)

- Browser webcam capture
- Real-time analysis via API
- Analytics dashboard
- Database-backed analytics storage
- Workspace settings per user

### 👤 Authentication

- User registration
- User login / logout
- Session management
- Workspace preference storage

### ⚡ Real-Time Communication

- WebSocket streaming for live analysis
- API-based image processing
- Live analytics updates

### 📊 Monitoring & Analytics

- Request counters
- Latency monitoring
- Aggregate analytics API
- Exportable detection logs

### ⚙️ CI/CD

- Automated testing workflows
- Deploy hooks
- Health monitoring endpoint

---

# 🛠️ Tech Stack

| Technology | Purpose |
|-----------|--------|
| Python | Core programming language |
| OpenCV | Computer vision processing |
| DeepFace | Face emotion analysis |
| NumPy | Image processing operations |
| Flask | Backend API & web server |
| Streamlit | Interactive analytics dashboard |
| WebSockets | Real-time streaming |
| SQLite / SQL DB | Analytics storage |
| Gunicorn | Production WSGI server |
| Eventlet | Async workers |
| Docker | Containerized deployment |

---

# 📂 Project Structure

```
MOTION-DETECTION
│
├── web/
│   ├── app.py
│   ├── routes/
│   ├── templates/
│   └── static/
│
├── streamlit_app.py
├── motion.py
├── wsgi.py
├── tests/
├── logs/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# ⚙️ Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/ReddyNikhilG/MOTION-DETECTION.git
cd MOTION-DETECTION
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Applications

## 1️⃣ Run Motion Detection Engine

```bash
python motion.py
```

This launches the **real-time webcam motion detection system**.

---

## 2️⃣ Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Then open:

```
http://localhost:8501
```

This dashboard provides:

- Live analytics
- Detection summaries
- Visualization of logs

---

## 3️⃣ Run Flask Web Application

```bash
python web/app.py
```

Open:

```
http://127.0.0.1:5000
```

---

# 🌐 Website Routes

| Route | Description |
|------|-------------|
| `/` | Live webcam analyzer |
| `/analytics` | Analytics dashboard |
| `/login` | User login |
| `/register` | User registration |

### API Endpoints

| Endpoint | Description |
|---------|-------------|
| `/api/health` | Service health check |
| `/api/analyze` | Analyze base64 image |
| `/api/analytics` | Aggregated analytics |
| `/api/workspace` | Workspace settings |
| `/api/metrics` | Monitoring metrics |

---

# 🐳 Docker Deployment

This repository includes a **production-ready Docker configuration**.

## Build Docker Image

```bash
docker build -t face-monitor .
```

## Run Container

```bash
docker run -p 5000:5000 face-monitor
```

The app will be available at:

```
http://localhost:5000
```

The container runs using:

```
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 wsgi:application
```

---

# 📊 Logs

Logs are generated inside the **logs/** directory.

Files include:

```
logs/
 ├── detections.csv
 ├── detections.jsonl
 ├── summary.json
 └── web_detections.jsonl
```

These logs store:

- face detection results
- emotion predictions
- motion state
- timestamps

---

# 🧪 Tests

Run unit tests using:

```bash
python -m unittest discover -s tests
```

---

# 📦 Packaging

To build a **Windows executable**:

```
build_exe.bat
```

To run the website quickly:

```
run_web.bat
```

---

# ⚙️ Environment Variables

| Variable | Description |
|--------|-------------|
| `SECRET_KEY` | Flask session secret |
| `DATABASE_URL` | Database connection string |
| `PORT` | Web server port (default 5000) |
| `FLASK_DEBUG` | Enable debug mode |

---

# ⚠️ Notes

- TensorFlow CUDA warnings can be ignored on systems without NVIDIA GPUs.
- The first run may take longer due to **DeepFace model downloads**.
- Ensure webcam permissions are enabled.

---

# 👨‍💻 Author

**Nikhil Reddy**

GitHub  
https://github.com/ReddyNikhilG

---

# ⭐ Support

If you like this project:

⭐ Star the repository  
🍴 Fork the project  
🚀 Contribute improvements
