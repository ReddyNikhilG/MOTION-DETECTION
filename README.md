# AI Face Monitor

Real-time webcam face analytics app built with PyQt5, OpenCV, and DeepFace.

Now includes a full website called FacePulse Live with browser webcam capture, API-based analysis, and analytics dashboard.

It now also includes:

- User authentication (register/login/logout)
- Per-user workspace settings persistence
- Real-time WebSocket analysis stream
- Database-backed analytics storage
- Monitoring endpoint for request and latency counters
- CI/CD workflows for testing and deploy hooks

## Features

- Multi-face detection with green bounding boxes
- Face tracking IDs across frames
- Smoothed age and emotion labels to reduce flicker
- Confidence threshold filtering
- Detector selection: Auto, MediaPipe, or Haar Cascade
- Mode presets: Fast, Balanced, Quality
- Local CSV and JSONL logging
- Summary export button
- Local-only privacy notice in UI

## Setup

1. Install dependencies:

```powershell
C:/Users/reddy/AppData/Local/Programs/Python/Python310/python.exe -m pip install -r requirements.txt
```

2. Run the app:

```powershell
C:/Users/reddy/AppData/Local/Programs/Python/Python310/python.exe motion.py
```

3. Run the website:

```powershell
C:/Users/reddy/AppData/Local/Programs/Python/Python310/python.exe web/app.py
```

Then open http://127.0.0.1:5000

## UI Controls

- Profile: performance preset
- Detector: Auto, MediaPipe, or Haar
- Analysis Every N Frames: DeepFace frequency
- Max Faces: maximum tracked faces
- Confidence %: minimum confidence to display and log
- Camera Index: camera source index

## Logs

Generated in `logs/`:

- `detections.csv`
- `detections.jsonl`
- `summary.json` (when Export Summary is clicked)
- `web_detections.jsonl` (website live analyzer logs)

## Website

- Live page: `/`
- Analytics page: `/analytics`
- Login page: `/login`
- Register page: `/register`
- Health endpoint: `/api/health`
- Analyze endpoint: `/api/analyze` (POST with base64 image)
- Aggregate analytics endpoint: `/api/analytics`
- Workspace endpoint: `/api/workspace`
- Monitoring endpoint: `/api/metrics`

## Tests

Run unit tests:

```powershell
C:/Users/reddy/AppData/Local/Programs/Python/Python310/python.exe -m unittest discover -s tests
```

## Packaging

Build a Windows executable:

```powershell
build_exe.bat
```

Run desktop website quickly using:

```powershell
run_web.bat
```

## Notes

- TensorFlow CUDA warnings can be ignored on non-NVIDIA systems.
- First run may take longer due to model weight downloads.

## Environment Variables

- `SECRET_KEY`: Flask session secret (set in production)
- `DATABASE_URL`: database DSN. Default is local SQLite file `webapp.db`
- `PORT`: web port (default 5000)
- `FLASK_DEBUG`: set `1` to enable debug mode
