import React, { useEffect, useRef } from "react";
import shaka from "shaka-player/dist/shaka-player.compiled.js";

export default function VideoPlayer({ src, poster }) {
  const videoRef = useRef(null);
  const playerRef = useRef(null);

  useEffect(() => {
    shaka.polyfill.installAll();
    if (!shaka.Player.isBrowserSupported()) return;
    const player = new shaka.Player(videoRef.current);
    playerRef.current = player;
    if (src) {
      player.load(src).catch((e) => console.error("shaka load error", e));
    }
    return () => { player.destroy(); };
  }, [src]);

  return (
    <video
      ref={videoRef}
      controls
      poster={poster}
      style={{ width: "100%", maxHeight: 540, background: "#000", borderRadius: 8 }}
    />
  );
}
