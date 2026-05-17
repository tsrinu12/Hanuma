import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listVideos } from "../api.js";

export default function Home() {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listVideos().then((v) => { setVideos(v || []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <div>Loading…</div>;
  if (!videos.length) return <div>No videos yet. <Link to="/upload">Upload one</Link>.</div>;

  return (
    <div>
      <h2>Trending</h2>
      <div style={{ display: "grid", gap: 16, gridTemplateColumns: "repeat(auto-fill,minmax(260px,1fr))" }}>
        {videos.map((v) => (
          <Link key={v.id} to={`/watch/${v.id}`} style={{ color: "#eee", textDecoration: "none" }}>
            <div style={{ background: "#15151c", borderRadius: 8, overflow: "hidden" }}>
              {v.thumbnail_url
                ? <img src={v.thumbnail_url} alt="" style={{ width: "100%", aspectRatio: "16/9", objectFit: "cover" }} />
                : <div style={{ aspectRatio: "16/9", background: "#222" }} />}
              <div style={{ padding: 12 }}>
                <div style={{ fontWeight: 600 }}>{v.title}</div>
                <div style={{ color: "#888", fontSize: 12 }}>{Math.round(v.duration_seconds || 0)}s</div>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
