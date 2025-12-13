const root = document.documentElement;

export const API_BASE = root.dataset.apiBase || window.location.origin;
export const DEFAULT_SAMPLE_RATE = 16000;
// Smaller frame for faster VAD reaction.
export const FRAME_SIZE = 1024;
