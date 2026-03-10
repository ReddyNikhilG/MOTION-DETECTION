# Browser Deployment Guide

This project now supports browser deployment with Flask + Gunicorn.

It includes authentication, websocket live analysis, and database-backed analytics.

## Option 1: Run Locally In Browser

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

## Option 2: Deploy To Render (Recommended)

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

## Option 3: Deploy To Railway

1. Push this project to GitHub.
2. In Railway: New Project -> Deploy from GitHub repo.
3. Railway detects Dockerfile automatically.
4. Deploy and open generated domain.

## Notes

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
