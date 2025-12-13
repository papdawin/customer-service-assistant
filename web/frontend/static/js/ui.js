import {
  answerEl,
  healthPill,
  logEl,
  ttsAudio,
  transcriptEl,
  vadPill,
  wsPill,
} from "./dom.js";

export function logMessage(message) {
  const now = new Date().toLocaleTimeString();
  logEl.textContent = `[${now}] ${message}\n` + logEl.textContent;
}

export function setHealth(status) {
  healthPill.textContent = status;
}

export function setStreamState(state) {
  wsPill.textContent = `stream: ${state}`;
}

export function setVadState(state) {
  vadPill.textContent = `vad: ${state}`;
}

export function renderResult(data) {
  transcriptEl.textContent = data.transcript || "(empty)";
  answerEl.textContent = data.answer || "(no answer)";
  if (data.audio_b64 && data.audio_mime) {
    ttsAudio.src = `data:${data.audio_mime};base64,${data.audio_b64}`;
    ttsAudio.play().catch(() => {});
  }
}
