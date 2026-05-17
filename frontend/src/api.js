const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8080";

export async function listVideos() {
  const r = await fetch(`${BASE}/metadata/videos?status=ready`);
  return r.json();
}

export async function getVideo(id) {
  const r = await fetch(`${BASE}/metadata/videos/${id}`);
  return r.json();
}

export async function getAiOutputs(id) {
  const r = await fetch(`${BASE}/metadata/ai-outputs/${id}`);
  return r.json();
}

export async function semanticSearch(q, top_k = 10) {
  const r = await fetch(`${BASE}/search/search`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q, top_k }),
  });
  return r.json();
}

export async function recommendFor(userId) {
  const r = await fetch(`${BASE}/recommend/recommend/${userId}`);
  return r.json();
}

export async function uploadVideo({ file, title, description, owner_id }) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("title", title);
  fd.append("description", description || "");
  if (owner_id) fd.append("owner_id", owner_id);
  const r = await fetch(`${BASE}/video/videos/upload`, { method: "POST", body: fd });
  return r.json();
}
