// Thin fetch wrapper for the FastAPI pipeline backend (src/api/server.py).
// In dev, Vite proxies /api → http://127.0.0.1:8000.

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: options.body instanceof FormData
      ? undefined
      : { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail ?? JSON.stringify(data);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),
  reference: () => request("/api/reference"),
  state: () => request("/api/state"),

  uploadFile: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/upload", { method: "POST", body: form });
  },
  uploadSample: () => request("/api/upload-sample", { method: "POST" }),

  annotate: (mode) =>
    request("/api/annotate", { method: "POST", body: JSON.stringify({ mode }) }),

  queue: () => request("/api/queue"),
  review: (payload) =>
    request("/api/review", { method: "POST", body: JSON.stringify(payload) }),

  finalize: (fullAuto) =>
    request("/api/finalize", {
      method: "POST",
      body: JSON.stringify({ full_auto: fullAuto }),
    }),

  activeLearning: (dimension) =>
    request(`/api/active-learning?dimension=${encodeURIComponent(dimension)}`),
  fairness: () => request("/api/fairness"),
  eloRun: (dimension, budget) =>
    request("/api/elo/run", {
      method: "POST",
      body: JSON.stringify({ dimension, budget }),
    }),

  exportUrl: "/api/export.csv",
};

export const ALL_DIMENSIONS = [
  "Linguistic_Feature",
  "Relevance",
  "Clarity",
  "Semantic_Plausibility",
  "Semantic_Uniqueness",
];

export const DIM_LABELS = {
  Linguistic_Feature: "Linguistic Feature",
  Relevance: "Topic Relevance",
  Clarity: "Writing Clarity",
  Semantic_Plausibility: "Semantic Plausibility",
  Semantic_Uniqueness: "Semantic Uniqueness",
};
