const logEl = document.getElementById("log");
const textEl = document.getElementById("text");
const wsStateEl = document.getElementById("wsState");

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function appendText(delta) {
  textEl.textContent += delta;
}

let ws;
let audioCtx;
let micStream;
let sourceNode;
let processorNode;
let isRecording = false;

const TARGET_SR = window.TARGET_SAMPLE_RATE || 24000;

// ---- Playback queue (PCM16 @ TARGET_SR) ----
let playTime = 0;
function playPcm16Chunk(pcmBytes) {
  if (!audioCtx) return;

  const int16 = new Int16Array(pcmBytes.buffer, pcmBytes.byteOffset, Math.floor(pcmBytes.byteLength / 2));
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

  const buffer = audioCtx.createBuffer(1, float32.length, TARGET_SR);
  buffer.copyToChannel(float32, 0);

  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioCtx.destination);

  const now = audioCtx.currentTime;
  if (playTime < now) playTime = now;
  src.start(playTime);
  playTime += buffer.duration;
}

// ---- Downsample Float32 -> Int16 PCM ----
function downsampleToInt16(float32, inSampleRate, outSampleRate) {
  if (outSampleRate === inSampleRate) {
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  const ratio = inSampleRate / outSampleRate;
  const newLen = Math.floor(float32.length / ratio);
  const out = new Int16Array(newLen);

  let offset = 0;
  for (let i = 0; i < newLen; i++) {
    const idx = Math.floor(offset);
    const s = float32[idx] || 0;
    const clamped = Math.max(-1, Math.min(1, s));
    out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    offset += ratio;
  }
  return out;
}

function ensureWebSocket() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    wsStateEl.textContent = "WS: connected";
    log("WebSocket connected");
  };

  ws.onclose = () => {
    wsStateEl.textContent = "WS: disconnected";
    log("WebSocket closed");
  };

  ws.onerror = (e) => {
    log("WebSocket error: " + e);
  };

  ws.onmessage = (evt) => {
    if (typeof evt.data === "string") {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "status") log("[status] " + msg.message);
        if (msg.type === "error") log("[error] " + msg.message);
        if (msg.type === "text_delta") appendText(msg.delta);
        if (msg.type === "tts_transcript") log("[tts_transcript] " + msg.text);
        if (msg.type === "done") log("[done]");
      } catch {
        log("Text msg: " + evt.data);
      }
    } else {
      // binary audio chunk from model (PCM16 @ TARGET_SR)
      playPcm16Chunk(new Uint8Array(evt.data));
    }
  };
}

async function startRecording() {
  ensureWebSocket();
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    log("WS not ready yet. Try again in a second.");
    return;
  }
  if (isRecording) return;

  // reset UI text
  textEl.textContent = "";
  playTime = 0;

  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();

  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  sourceNode = audioCtx.createMediaStreamSource(micStream);

  // ScriptProcessorNode is deprecated but still widely supported and simplest for a PoC.
  // bufferSize 4096 gives stable callbacks. We'll downsample to 24k and send.
  processorNode = audioCtx.createScriptProcessor(4096, 1, 1);

  processorNode.onaudioprocess = (e) => {
    if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;

    const input = e.inputBuffer.getChannelData(0);
    const inRate = audioCtx.sampleRate;

    const pcm16 = downsampleToInt16(input, inRate, TARGET_SR);

    // send as binary
    ws.send(pcm16.buffer);
  };

  sourceNode.connect(processorNode);
  // Do NOT connect processorNode to destination (avoids echo), but some browsers require it connected:
  processorNode.connect(audioCtx.destination);

  ws.send(JSON.stringify({ type: "start" }));
  isRecording = true;
  log("Recording started (streaming mic audio)");
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;

  try {
    ws.send(JSON.stringify({ type: "stop" }));
  } catch {}

  try {
    if (processorNode) processorNode.disconnect();
    if (sourceNode) sourceNode.disconnect();
  } catch {}

  try {
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
    }
  } catch {}

  processorNode = null;
  sourceNode = null;
  micStream = null;

  log("Recording stopped (waiting for model response)");
}

document.getElementById("btnStart").onclick = async () => {
  document.getElementById("btnStart").disabled = true;
  document.getElementById("btnStop").disabled = false;
  try {
    await startRecording();
  } finally {
    // keep Start disabled while recording
  }
};

document.getElementById("btnStop").onclick = () => {
  document.getElementById("btnStop").disabled = true;
  document.getElementById("btnStart").disabled = false;
  stopRecording();
};

// Connect early
ensureWebSocket();
