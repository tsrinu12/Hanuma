import React, { useState } from "react";
import { uploadVideo } from "../api.js";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    if (!file || !title) return;
    setBusy(true);
    try {
      const r = await uploadVideo({ file, title, description });
      setResult(r);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} style={{ maxWidth: 540, display: "grid", gap: 12 }}>
      <h2>Upload a video</h2>
      <input value={title} onChange={(e) => setTitle(e.target.value)}
             placeholder="Title" required style={input} />
      <textarea value={description} onChange={(e) => setDescription(e.target.value)}
                placeholder="Description" rows={4} style={input} />
      <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} required />
      <button disabled={busy} style={btn}>{busy ? "Uploading…" : "Upload"}</button>
      {result && (
        <pre style={{ background: "#111", padding: 12, borderRadius: 6 }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </form>
  );
}

const input = { background: "#15151c", color: "#fff", border: "1px solid #2a2a35",
                borderRadius: 6, padding: "10px 12px" };
const btn = { background: "#7c5cff", color: "#fff", padding: "10px 16px",
              border: 0, borderRadius: 6, cursor: "pointer", fontWeight: 600 };
