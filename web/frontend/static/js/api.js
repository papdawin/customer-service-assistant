import { API_BASE } from "./config.js";

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

export async function postVoice(form) {
  const res = await fetch(`${API_BASE}/api/voice`, { method: "POST", body: form });
  return res.json();
}

export function streamUrl() {
  return (API_BASE || window.location.origin).replace(/^http/, "ws") + "/api/stream";
}
