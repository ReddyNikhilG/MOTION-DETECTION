const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const statusText = document.getElementById('statusText');
const faceCount = document.getElementById('faceCount');
const latency = document.getElementById('latency');
const topEmotion = document.getElementById('emotion');
const confidenceValue = document.getElementById('confidenceValue');
const qualityBadge = document.getElementById('qualityBadge');
const cadenceText = document.getElementById('cadenceText');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const cameraMode = document.getElementById('cameraMode');
const cameraDevice = document.getElementById('cameraDevice');
const refreshDevicesBtn = document.getElementById('refreshDevicesBtn');
const intervalRange = document.getElementById('intervalRange');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');
const screenshotBtn = document.getElementById('screenshotBtn');
const analyzeSpinner = document.getElementById('analyzeSpinner');
const connectionStatus = document.getElementById('connectionStatus');
const faceDetailsPanel = document.getElementById('faceDetailsPanel');
const faceDetailsList = document.getElementById('faceDetailsList');
const toastContainer = document.getElementById('toastContainer');

const snapshot = document.createElement('canvas');
const sctx = snapshot.getContext('2d');
const octx = overlay.getContext('2d');
const socket = io();

let stream = null;
let timer = null;
let busy = false;
let currentIntervalMs = Number(intervalRange.value);
let lastResult = null;
let lastResultTs = 0;

const MIN_INTERVAL = Number(intervalRange.min || 600);
const MAX_INTERVAL = Number(intervalRange.max || 3000);

// ── Toast notifications ────────────────────────────────────
function showToast(message, type = 'info', duration = 3000) {
  if (!toastContainer) return;
  const toast = document.createElement('div');
  toast.className = 'toast toast-' + type;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('toast-visible'));
  setTimeout(() => {
    toast.classList.remove('toast-visible');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Connection status indicator ────────────────────────────
socket.on('connect', () => {
  if (connectionStatus) {
    connectionStatus.className = 'conn-badge conn-connected';
    connectionStatus.innerHTML = '&#x25CF; Online';
  }
});

socket.on('disconnect', () => {
  if (connectionStatus) {
    connectionStatus.className = 'conn-badge conn-disconnected';
    connectionStatus.innerHTML = '&#x25CF; Offline';
  }
  showToast('Connection lost — retrying...', 'error');
});

function getVideoConstraints() {
  const selectedDevice = cameraDevice ? cameraDevice.value : '';
  if (selectedDevice) {
    return { deviceId: { exact: selectedDevice } };
  }

  const mode = cameraMode ? cameraMode.value : 'auto';
  if (mode === 'user' || mode === 'environment') {
    return { facingMode: { ideal: mode } };
  }

  return true;
}

function buildDeviceLabel(device, index) {
  if (device.label && device.label.trim()) {
    return device.label;
  }
  return `Camera ${index + 1}`;
}

async function refreshCameraDevices() {
  if (!cameraDevice || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    return;
  }

  const prev = cameraDevice.value;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videos = devices.filter((d) => d.kind === 'videoinput');

  cameraDevice.innerHTML = '';

  const autoOption = document.createElement('option');
  autoOption.value = '';
  autoOption.textContent = 'Auto Device';
  cameraDevice.appendChild(autoOption);

  videos.forEach((d, idx) => {
    const option = document.createElement('option');
    option.value = d.deviceId;
    option.textContent = buildDeviceLabel(d, idx);
    cameraDevice.appendChild(option);
  });

  if (prev && videos.some((d) => d.deviceId === prev)) {
    cameraDevice.value = prev;
  }
}

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
  if (analyzeSpinner) analyzeSpinner.style.display = 'flex';
  try {
    snapshot.width = video.videoWidth;
    snapshot.height = video.videoHeight;
    sctx.drawImage(video, 0, 0, snapshot.width, snapshot.height);
    const image = snapshot.toDataURL('image/jpeg', 0.82);

    socket.emit('analyze_frame', { image, requested_interval_ms: currentIntervalMs });
  } catch (err) {
    statusText.textContent = 'Status: request failed';
    showToast('Frame capture failed', 'error');
  } finally {
    busy = false;
  }
}

function getTopConfidence(faces) {
  if (!Array.isArray(faces) || !faces.length) {
    return null;
  }

  const vals = faces
    .map((f) => f.confidence)
    .filter((v) => typeof v === 'number');
  if (!vals.length) {
    return null;
  }
  return Math.max(...vals);
}

function getQualityLabel(latencyMs) {
  if (latencyMs <= 320) return 'Excellent';
  if (latencyMs <= 650) return 'Good';
  if (latencyMs <= 1000) return 'Fair';
  return 'Heavy';
}

function updateCadenceText() {
  if (!cadenceText) return;
  cadenceText.textContent = `${currentIntervalMs} ms`;
}

function scheduleNextCapture() {
  if (!stream) return;
  if (timer) {
    clearTimeout(timer);
  }
  timer = setTimeout(captureLoop, currentIntervalMs);
}

async function captureLoop() {
  await captureAndAnalyze();
  scheduleNextCapture();
}

function applyAdaptiveInterval(latencyMs) {
  const baseInterval = Number(intervalRange.value);
  let nextInterval = baseInterval;

  if (latencyMs > 900) {
    nextInterval = Math.min(MAX_INTERVAL, Math.max(baseInterval, latencyMs + 220));
  } else if (latencyMs < 320) {
    nextInterval = Math.max(MIN_INTERVAL, baseInterval);
  }

  currentIntervalMs = Math.round(nextInterval / 50) * 50;
  updateCadenceText();
}

function updateFaceDetails(faces) {
  if (!faceDetailsPanel || !faceDetailsList) return;
  if (!faces || !faces.length) {
    faceDetailsPanel.style.display = 'none';
    return;
  }
  faceDetailsPanel.style.display = 'block';
  faceDetailsList.innerHTML = '';
  faces.forEach((f, i) => {
    const card = document.createElement('div');
    card.className = 'face-detail-card';
    const age = f.age !== undefined && f.age !== 'N/A' ? f.age : '?';
    const conf = typeof f.confidence === 'number' ? Math.round(f.confidence) + '%' : 'N/A';
    card.innerHTML =
      '<strong>Face ' + (i + 1) + '</strong>' +
      '<span class="fd-row">Age: <b>' + age + '</b></span>' +
      '<span class="fd-row">Emotion: <b>' + (f.emotion || 'N/A') + '</b></span>' +
      '<span class="fd-row">Confidence: <b>' + conf + '</b></span>';
    faceDetailsList.appendChild(card);
  });
}

socket.on('analyze_result', (data) => {
  if (analyzeSpinner) analyzeSpinner.style.display = 'none';

  if (data.error) {
    statusText.textContent = 'Status: ' + data.error;
    if (lastResult && Date.now() - lastResultTs < 1200) {
      drawFaces(lastResult.faces || []);
    }
    return;
  }

  statusText.textContent = 'Status: running';
  faceCount.textContent = String(data.face_count);
  latency.textContent = data.latency_ms + ' ms';

  const emotions = data.faces.map((f) => f.emotion).filter((e) => e && e !== 'N/A');
  topEmotion.textContent = emotions[0] || 'N/A';

  const confidence = getTopConfidence(data.faces);
  confidenceValue.textContent = confidence === null ? 'N/A' : Math.round(confidence) + '%';

  const quality = getQualityLabel(data.latency_ms);
  qualityBadge.textContent = quality;

  applyAdaptiveInterval(Number(data.latency_ms) || currentIntervalMs);
  lastResult = data;
  lastResultTs = Date.now();

  drawFaces(data.faces);
  updateFaceDetails(data.faces);
});

async function startCamera() {
  if (stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: getVideoConstraints(), audio: false });
    video.srcObject = stream;
    await video.play();
    resizeOverlay();
    statusText.textContent = 'Status: camera started';
    showToast('Camera started', 'success');

    await refreshCameraDevices();

    currentIntervalMs = Number(intervalRange.value);
    updateCadenceText();
    scheduleNextCapture();
  } catch (err) {
    statusText.textContent = 'Status: camera permission denied or unavailable';
    showToast('Camera access denied', 'error');
  }
}

function stopCamera() {
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  octx.clearRect(0, 0, overlay.width, overlay.height);
  lastResult = null;
  lastResultTs = 0;
  confidenceValue.textContent = 'N/A';
  qualityBadge.textContent = 'N/A';
  cadenceText.textContent = '-';
  statusText.textContent = 'Status: stopped';
  if (faceDetailsPanel) faceDetailsPanel.style.display = 'none';
  if (analyzeSpinner) analyzeSpinner.style.display = 'none';
  showToast('Camera stopped', 'info');
}

async function restartCameraIfRunning() {
  if (!stream) return;
  stopCamera();
  await startCamera();
}

intervalRange.addEventListener('input', () => {
  currentIntervalMs = Number(intervalRange.value);
  updateCadenceText();
  if (stream) {
    scheduleNextCapture();
  }
});

async function loadWorkspaceSettings() {
  try {
    const res = await fetch('/api/workspace');
    if (!res.ok) return;
    const data = await res.json();
    if (data.intervalMs) {
      intervalRange.value = String(data.intervalMs);
      currentIntervalMs = Number(data.intervalMs);
      updateCadenceText();
    }
    if (cameraMode && data.cameraMode) {
      cameraMode.value = data.cameraMode;
    }
    await refreshCameraDevices();
    if (cameraDevice && typeof data.cameraDeviceId === 'string') {
      cameraDevice.value = data.cameraDeviceId;
    }
  } catch (err) {
    statusText.textContent = 'Status: failed to load settings';
  }
}

async function saveWorkspaceSettings() {
  try {
    const payload = {
      intervalMs: Number(intervalRange.value),
      cameraMode: cameraMode ? cameraMode.value : 'auto',
      cameraDeviceId: cameraDevice ? cameraDevice.value : '',
    };
    const res = await fetch('/api/workspace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      statusText.textContent = 'Status: save failed';
      showToast('Failed to save settings', 'error');
      return;
    }
    statusText.textContent = 'Status: workspace saved';
    showToast('Settings saved', 'success');
  } catch (err) {
    statusText.textContent = 'Status: save failed';
    showToast('Failed to save settings', 'error');
  }
}

function takeScreenshot() {
  if (!stream) {
    showToast('Start the camera first', 'error');
    return;
  }
  const c = document.createElement('canvas');
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  const ctx = c.getContext('2d');
  ctx.drawImage(video, 0, 0);
  // Also draw the overlay (face boxes)
  ctx.drawImage(overlay, 0, 0);

  const link = document.createElement('a');
  link.download = 'facepulse_' + new Date().toISOString().replace(/[:.]/g, '-') + '.png';
  link.href = c.toDataURL('image/png');
  link.click();
  showToast('Screenshot saved', 'success');
}

window.addEventListener('resize', resizeOverlay);
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
if (cameraMode) {
  cameraMode.addEventListener('change', restartCameraIfRunning);
}
if (cameraDevice) {
  cameraDevice.addEventListener('change', restartCameraIfRunning);
}
if (refreshDevicesBtn) {
  refreshDevicesBtn.addEventListener('click', refreshCameraDevices);
}
if (saveSettingsBtn) {
  saveSettingsBtn.addEventListener('click', saveWorkspaceSettings);
}
if (screenshotBtn) {
  screenshotBtn.addEventListener('click', takeScreenshot);
}
updateCadenceText();
refreshCameraDevices();
loadWorkspaceSettings();
