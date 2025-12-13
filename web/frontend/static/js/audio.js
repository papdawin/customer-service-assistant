import { DEFAULT_SAMPLE_RATE, FRAME_SIZE } from "./config.js";
import { logMessage } from "./ui.js";

let micStream;

export async function ensureMic() {
  if (micStream) return micStream;
  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  return micStream;
}

export function currentMicStream() {
  return micStream;
}

export function toPCM16(buffer, inSampleRate, targetRate = DEFAULT_SAMPLE_RATE) {
  if (inSampleRate !== targetRate) {
    const ratio = inSampleRate / targetRate;
    const newLength = Math.round(buffer.length / ratio);
    const resampled = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      resampled[i] = buffer[Math.min(buffer.length - 1, Math.round(i * ratio))];
    }
    return toPCM16(resampled, targetRate, targetRate);
  }
  const pcm = new Int16Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) {
    const s = Math.max(-1, Math.min(1, buffer[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm;
}

export function startProcessor(stream, onFrame) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: DEFAULT_SAMPLE_RATE,
  });
  const input = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(FRAME_SIZE, 1, 1);
  input.connect(processor);
  processor.connect(audioCtx.destination);

  processor.onaudioprocess = (event) => {
    const floatData = event.inputBuffer.getChannelData(0);
    onFrame(floatData, audioCtx.sampleRate);
  };

  logMessage("audio processor started");

  return {
    audioCtx,
    processor,
    stop: async () => {
      try {
        processor.disconnect();
        await audioCtx.close();
      } catch (_) {}
      logMessage("audio processor stopped");
    },
  };
}
