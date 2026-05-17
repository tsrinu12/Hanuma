import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import VideoPlayer from "./VideoPlayer.jsx";
import { getVideo, getAiOutputs } from "../api.js";

export default function Watch() {
  const { id } = useParams();
  const [video, setVideo] = useState(null);
  const [ai, setAi] = useState([]);

  useEffect(() => {
    getVideo(id).then(setVideo);
    getAiOutputs(id).then(setAi).catch(() => {});
  }, [id]);

  if (!video) return <div>Loading…</div>;
  const summary = ai.find((a) => a.kind === "summary")?.payload?.summary;
  const highlights = ai.find((a) => a.kind === "highlights")?.payload || [];

  return (
    <div style={{ display: "grid", gap: 24, gridTemplateColumns: "minmax(0, 2fr) 1fr" }}>
      <div>
        <VideoPlayer src={video.hls_master_url} poster={video.thumbnail_url} />
        <h2 style={{ marginTop: 12 }}>{video.title}</h2>
        <p style={{ color: "#aaa" }}>{video.description}</p>
      </div>
      <aside style={{ background: "#15151c", padding: 16, borderRadius: 8 }}>
        <h3>AI Summary</h3>
        <p style={{ color: "#ddd", whiteSpace: "pre-wrap" }}>{summary || "Processing…"}</p>
        <h3 style={{ marginTop: 16 }}>Highlights</h3>
        <ul>
          {highlights.map((h, i) => (
            <li key={i}>
              <strong>{formatTime(h.start)}</strong> — {h.text}
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}

function formatTime(s) {
  const m = Math.floor(s / 60); const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
