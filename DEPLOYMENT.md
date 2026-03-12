# Streamlit Deployment Guide

This project supports direct deployment of the Streamlit app in `streamlit_app.py`.

Use this path if you want the simplest hosted version with camera capture, image upload, video analysis, analytics, and real-time motion labels.

## Option 1: Run Locally With Streamlit

1. Install dependencies:

```powershell
C:/Users/reddy/AppData/Local/Programs/Python/Python310/python.exe -m pip install -r requirements.txt
```

2. Start the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

3. Open the local URL printed in the terminal, usually:

- http://127.0.0.1:8501

## Option 2: Deploy To Streamlit Community Cloud

1. Push this project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from your repository.
4. Use these settings:
- Branch: `main` (or your deployment branch)
- Main file path: `streamlit_app.py`
- Python version: `3.12` or `3.11`
5. Deploy.

Streamlit Community Cloud will install dependencies from `requirements.txt` and native Linux packages from `packages.txt`.

If you leave the app on a newer Python version where TensorFlow wheels are unavailable, the deployment will still succeed, but emotion inference will be disabled and the app will run with face and motion detection only.

## Streamlit Files Used By Deployment

- App entrypoint: `streamlit_app.py`
- Python version: `runtime.txt`
- Python dependencies: `requirements.txt`
- Linux system packages: `packages.txt`
- Streamlit config: `.streamlit/config.toml`

## Option 3: Run Flask App Locally In Browser

1. Install dependencies:

```powershell
C:/Users/reddy/AppData/Local/Programs/Python/Python310/python.exe -m pip install -r requirements.txt
```

2. Start web app:

```powershell
C:/Users/reddy/AppData/Local/Programs/Python/Python310/python.exe web/app.py
```

3. Open:

- http://127.0.0.1:5000

## Option 4: Deploy Flask App To Render

1. Push this project to GitHub.
2. In Render: New + -> Web Service.
3. Connect your GitHub repo.
4. Use these settings:
- Environment: Docker
- Branch: main (or your branch)
- Region: nearest to you
- Add environment variables:
	- `SECRET_KEY=<strong-random-secret>`
	- `DATABASE_URL=<your-postgres-url>` (optional; if omitted, SQLite is used)
	- `PORT=5000`
5. Deploy.

Render will build from the Dockerfile and start Gunicorn automatically.

## Option 5: Deploy Flask App To Railway

1. Push this project to GitHub.
2. In Railway: New Project -> Deploy from GitHub repo.
3. Railway detects Dockerfile automatically.
4. Deploy and open generated domain.

## Notes

- Streamlit is the recommended deployment target for this repository if you do not need login, websocket streaming, or the separate Flask UI.
- For full DeepFace emotion inference on Streamlit Community Cloud, select Python `3.12` or `3.11` in Advanced settings during deployment.
- This app uses webcam capture in browser via `getUserMedia`.
- Browser camera access requires HTTPS on public deployments.
- TensorFlow may run on CPU if CUDA is unavailable.
- First requests can be slower while models warm up.

## CI/CD

- CI workflow: `.github/workflows/ci.yml`
- Render deploy workflow: `.github/workflows/deploy-render.yml`

To enable auto deploy hook from GitHub Actions:

1. Create deploy hook URL in Render service settings.
2. Add repository secret `RENDER_DEPLOY_HOOK_URL` in GitHub.
3. Push to `main` or manually trigger workflow.

## Production Command (Non-Docker)

If your host does not use Docker and supports Python processes:

```bash
gunicorn --bind 0.0.0.0:$PORT wsgi:application
```
