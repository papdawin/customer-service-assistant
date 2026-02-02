import { streamUrl } from "./api.js";
import { ensureMic, startProcessor, toPCM16, currentMicStream } from "./audio.js";
import { API_BASE, DEFAULT_SAMPLE_RATE } from "./config.js";
import {
  beamInput,
  greetingInput,
  languageInput,
  recordBtn,
  startBtn,
  stopBtn,
  vadAggInput,
  vadFrameInput,
  vadSilenceInput,
} from "./dom.js";
import { postVoice, checkHealth } from "./api.js";
import {
  logMessage,
  renderResult,
  setHealth,
  setStreamState,
  setVadState,
} from "./ui.js";

let ws;
let recorder;
let recorderChunks = [];
let processorHandle;

async function startStream() {
  await ensureMic();
  const stream = currentMicStream();
  processorHandle = startProcessor(stream, handleFrame);

  ws = new WebSocket(streamUrl());
  ws.binaryType = "arraybuffer";
  setStreamState("connecting…");
  logMessage("opening websocket");

  ws.onopen = () => {
    setStreamState("live");
    const settings = {
      language: languageInput.value || "hu",
      beam_size: Number(beamInput.value) || 5,
      sample_rate: DEFAULT_SAMPLE_RATE,
      greeting_text: greetingInput.value || "",
      vad: {
        aggressiveness: Number(vadAggInput.value) || 3,
        frame_ms: Number(vadFrameInput.value) || 10,
        silence_ms: Number(vadSilenceInput.value) || 250,
      },
    };
    ws.send(JSON.stringify(settings));
    logMessage("sent stream settings: " + JSON.stringify(settings));
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "ready") {
        logMessage(`session ready (${msg.session_id})`);
        setVadState("ready");
      } else if (msg.type === "greeting") {
        logMessage("playing greeting...");
        renderResult({ audio_b64: msg.audio_b64, audio_mime: msg.audio_mime });
      } else if (msg.type === "status") {
        setVadState(msg.message);
        logMessage(`vad status: ${msg.message}`);
      } else if (msg.type === "result") {
        renderResult(msg);
        logMessage("result delivered");
      }
    } catch (err) {
      logMessage("message error: " + err);
    }
  };

  ws.onclose = () => {
    setStreamState("closed");
    setVadState("idle");
  };

  startBtn.disabled = true;
  stopBtn.disabled = false;
}

function handleFrame(floatData, sampleRate) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const pcm = toPCM16(floatData, sampleRate);
  ws.send(pcm.buffer);
}

async function stopStream() {
  try {
    if (processorHandle) await processorHandle.stop();
    if (ws && ws.readyState === WebSocket.OPEN) ws.send("stop");
    if (ws) ws.close();
  } catch (_) {}
  ws = null;
  processorHandle = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStreamState("idle");
  setVadState("idle");
  logMessage("stream stopped");
}

async function sendOneShot() {
  await ensureMic();
  recorderChunks = [];
  const stream = currentMicStream();
  recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recorderChunks.push(e.data);
  };
  recorder.onstop = async () => {
    const blob = new Blob(recorderChunks, { type: recorder.mimeType || "audio/webm" });
    recorderChunks = [];
    const form = new FormData();
    form.append("audio", blob, "clip.webm");
    form.append("language", languageInput.value || "hu");
    form.append("beam_size", beamInput.value || "5");
    form.append("use_vad", "false");
    form.append("pcm_sample_rate", String(DEFAULT_SAMPLE_RATE));
    form.append("vad_aggressiveness", vadAggInput.value || "2");
    form.append("vad_silence_ms", vadSilenceInput.value || "800");

    const data = await postVoice(form);
    renderResult(data);
    logMessage("one-shot request completed");
  };
  recorder.start();
  setTimeout(() => recorder.stop(), 3000);
  logMessage("recording 3s clip…");
}

async function bootstrap() {
  try {
    const health = await checkHealth();
    setHealth(health.ok ? "health: ok" : "health: down");
    logMessage("health check ok");
  } catch (err) {
    setHealth("health: unreachable");
    logMessage("health check failed: " + err);
  }
}

startBtn.onclick = startStream;
stopBtn.onclick = stopStream;
recordBtn.onclick = sendOneShot;

bootstrap();
