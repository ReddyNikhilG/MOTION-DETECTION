const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const statusText = document.getElementById('statusText');
const faceCount = document.getElementById('faceCount');
const latency = document.getElementById('latency');
const topEmotion = document.getElementById('emotion');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const intervalRange = document.getElementById('intervalRange');

const snapshot = document.createElement('canvas');
const sctx = snapshot.getContext('2d');
const octx = overlay.getContext('2d');

let stream = null;
let timer = null;
let busy = false;

function resizeOverlay() {
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
}

function drawFaces(faces) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.strokeStyle = '#3bff8f';
  octx.lineWidth = 2;
  octx.font = '16px Segoe UI';
  octx.fillStyle = '#3bff8f';

  faces.forEach((f) => {
    const b = f.box;
    octx.strokeRect(b.x, b.y, b.w, b.h);
    let label = `Age ${f.age} | ${f.emotion}`;
    if (typeof f.confidence === 'number') {
      label += ` (${Math.round(f.confidence)}%)`;
    }
    const y = b.y > 20 ? b.y - 6 : b.y + b.h + 16;
    octx.fillText(label, b.x, y);
  });
}

async function captureAndAnalyze() {
  if (!stream || busy) {
    return;
  }

  busy = true;
  try {
    snapshot.width = video.videoWidth;
    snapshot.height = video.videoHeight;
    sctx.drawImage(video, 0, 0, snapshot.width, snapshot.height);
    const image = snapshot.toDataURL('image/jpeg', 0.82);

    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image }),
    });

    const data = await res.json();
    if (!res.ok) {
      statusText.textContent = 'Status: ' + (data.error || 'analyze error');
      return;
    }

    statusText.textContent = 'Status: running';
    faceCount.textContent = String(data.face_count);
    latency.textContent = data.latency_ms + ' ms';

    const emotions = data.faces.map((f) => f.emotion).filter((e) => e && e !== 'N/A');
    topEmotion.textContent = emotions[0] || 'N/A';

    drawFaces(data.faces);
  } catch (err) {
    statusText.textContent = 'Status: request failed';
  } finally {
    busy = false;
  }
}

async function startCamera() {
  if (stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();
    resizeOverlay();
    statusText.textContent = 'Status: camera started';

    const intervalMs = Number(intervalRange.value);
    timer = setInterval(captureAndAnalyze, intervalMs);
  } catch (err) {
    statusText.textContent = 'Status: camera permission denied or unavailable';
  }
}

function stopCamera() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  octx.clearRect(0, 0, overlay.width, overlay.height);
  statusText.textContent = 'Status: stopped';
}

intervalRange.addEventListener('input', () => {
  if (timer) {
    clearInterval(timer);
    timer = setInterval(captureAndAnalyze, Number(intervalRange.value));
  }
});

window.addEventListener('resize', resizeOverlay);
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
