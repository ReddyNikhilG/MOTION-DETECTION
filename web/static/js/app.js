const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const statusText = document.getElementById('statusText');
const faceCount = document.getElementById('faceCount');
const latency = document.getElementById('latency');
const topEmotion = document.getElementById('emotion');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const intervalRange = document.getElementById('intervalRange');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');

const snapshot = document.createElement('canvas');
const sctx = snapshot.getContext('2d');
const octx = overlay.getContext('2d');
const socket = io();

let stream = null;
let timer = null;
let busy = false;

function resizeOverlay() {
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
}

function drawFaces(faces) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.strokeStyle = '#4eb1ff';
  octx.lineWidth = 2;
  octx.font = '16px Segoe UI';
  octx.fillStyle = '#4eb1ff';

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

    socket.emit('analyze_frame', { image });
  } catch (err) {
    statusText.textContent = 'Status: request failed';
  } finally {
    busy = false;
  }
}

socket.on('analyze_result', (data) => {
  if (data.error) {
    statusText.textContent = 'Status: ' + data.error;
    return;
  }

  statusText.textContent = 'Status: running';
  faceCount.textContent = String(data.face_count);
  latency.textContent = data.latency_ms + ' ms';

  const emotions = data.faces.map((f) => f.emotion).filter((e) => e && e !== 'N/A');
  topEmotion.textContent = emotions[0] || 'N/A';

  drawFaces(data.faces);
});

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

async function loadWorkspaceSettings() {
  try {
    const res = await fetch('/api/workspace');
    if (!res.ok) return;
    const data = await res.json();
    if (data.intervalMs) {
      intervalRange.value = String(data.intervalMs);
    }
  } catch (err) {
    statusText.textContent = 'Status: failed to load settings';
  }
}

async function saveWorkspaceSettings() {
  try {
    const payload = {
      intervalMs: Number(intervalRange.value),
    };
    const res = await fetch('/api/workspace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      statusText.textContent = 'Status: save failed';
      return;
    }
    statusText.textContent = 'Status: workspace saved';
  } catch (err) {
    statusText.textContent = 'Status: save failed';
  }
}

window.addEventListener('resize', resizeOverlay);
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
if (saveSettingsBtn) {
  saveSettingsBtn.addEventListener('click', saveWorkspaceSettings);
}
loadWorkspaceSettings();
