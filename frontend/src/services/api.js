import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 5000,
});

/* ── Orchestrator endpoints (port 8001) ── */

export async function fetchState() {
  const { data } = await api.get("/state");
  return data;
}

export async function fetchHistory(limit = 120) {
  const { data } = await api.get(`/history?limit=${limit}`);
  return data;
}

export async function fetchComparison() {
  const { data } = await api.get("/comparison");
  return data;
}

export async function fetchAIStatus() {
  const { data } = await api.get("/ai/status");
  return data;
}

export async function fetchSafety() {
  const { data } = await api.get("/safety");
  return data;
}

export async function toggleAI(enable) {
  const { data } = await api.post("/ai/toggle", { enable });
  return data;
}

export async function fetchHealth() {
  const { data } = await api.get("/health");
  return data;
}

export default api;
