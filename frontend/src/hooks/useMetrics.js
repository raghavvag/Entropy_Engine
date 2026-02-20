import { useState, useEffect, useRef, useCallback } from "react";
import { fetchState, fetchHistory, fetchComparison } from "../services/api";

/* ────────────────────────────────────────────
   useMetrics — polls /api/state every second
   ──────────────────────────────────────────── */
export function useMetrics(intervalMs = 1000) {
  const [state, setState] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const data = await fetchState();
        if (active) {
          setState(data);
          setConnected(true);
        }
      } catch {
        if (active) setConnected(false);
      }
    };
    poll();
    const id = setInterval(poll, intervalMs);
    return () => { active = false; clearInterval(id); };
  }, [intervalMs]);

  return { state, connected };
}

/* ────────────────────────────────────────────
   useHistory — polls /api/history
   ──────────────────────────────────────────── */
export function useHistory(intervalMs = 2000, limit = 120) {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const data = await fetchHistory(limit);
        if (active) setHistory(data);
      } catch { /* skip */ }
    };
    poll();
    const id = setInterval(poll, intervalMs);
    return () => { active = false; clearInterval(id); };
  }, [intervalMs, limit]);

  return history;
}

/* ────────────────────────────────────────────
   useComparison — polls /api/comparison
   ──────────────────────────────────────────── */
export function useComparison(intervalMs = 3000) {
  const [comparison, setComparison] = useState(null);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const data = await fetchComparison();
        if (active) setComparison(data);
      } catch { /* skip */ }
    };
    poll();
    const id = setInterval(poll, intervalMs);
    return () => { active = false; clearInterval(id); };
  }, [intervalMs]);

  return comparison;
}
