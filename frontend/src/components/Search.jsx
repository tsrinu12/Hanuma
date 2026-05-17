import React, { useState } from "react";
import { Link } from "react-router-dom";
import { semanticSearch } from "../api.js";

export default function Search() {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState([]);
  const [busy, setBusy] = useState(false);

  async function go(e) {
    e.preventDefault();
    setBusy(true);
    try {
      const r = await semanticSearch(q, 15);
      setHits(r || []);
    } finally { setBusy(false); }
  }

  return (
    <div style={{ maxWidth: 800 }}>
      <h2>Semantic search</h2>
      <p style={{ color: "#888" }}>Ask in plain English — "videos that explain Postgres indexes", etc.</p>
      <form onSubmit={go} style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input value={q} onChange={(e) => setQ(e.target.value)}
               placeholder="What are you looking for?"
               style={{ flex: 1, padding: "10px 14px", background: "#15151c",
                        color: "#fff", border: "1px solid #2a2a35", borderRadius: 6 }} />
        <button disabled={busy || !q} style={{
          background: "#7c5cff", color: "#fff", padding: "10px 16px",
          border: 0, borderRadius: 6, fontWeight: 600 }}>Search</button>
      </form>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {hits.map((h, i) => (
          <li key={i} style={{ background: "#15151c", padding: 12, marginBottom: 10, borderRadius: 6 }}>
            <Link to={`/watch/${h.video_id}`} style={{ color: "#7c5cff" }}>
              ▶ {h.video_id} @ {Math.round(h.start)}s
            </Link>
            <div style={{ color: "#ddd", marginTop: 4 }}>{h.text}</div>
            <div style={{ color: "#666", fontSize: 12 }}>score {h.score.toFixed(3)}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
