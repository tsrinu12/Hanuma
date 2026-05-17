"""HLS transcoding via FFmpeg. Produces a multi-bitrate ladder + master.m3u8."""
import os
import shutil
import subprocess
import tempfile
from typing import List

LADDER = [
    # (resolution, video_bitrate, audio_bitrate)
    ("640x360", "800k", "96k"),
    ("854x480", "1400k", "128k"),
    ("1280x720", "2800k", "128k"),
]


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def probe_duration(path: str) -> float:
    """Return media duration in seconds, or 0.0 if ffprobe fails."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path]
        )
        return float(out.strip())
    except Exception:
        return 0.0


def transcode_to_hls(input_path: str, out_dir: str) -> str:
    """Generates HLS segments + master.m3u8 inside out_dir. Returns master path."""
    os.makedirs(out_dir, exist_ok=True)
    variant_playlists = []

    for res, vb, ab in LADDER:
        name = res.split("x")[1] + "p"
        variant_dir = os.path.join(out_dir, name)
        os.makedirs(variant_dir, exist_ok=True)
        run([
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"scale={res}",
            "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "main",
            "-b:v", vb, "-maxrate", vb, "-bufsize", vb,
            "-c:a", "aac", "-b:a", ab, "-ac", "2",
            "-hls_time", "6", "-hls_playlist_type", "vod",
            "-hls_segment_filename", os.path.join(variant_dir, "seg_%03d.ts"),
            os.path.join(variant_dir, "index.m3u8"),
        ])
        bandwidth = int(vb.rstrip("k")) * 1000 + int(ab.rstrip("k")) * 1000
        variant_playlists.append((bandwidth, res, name))

    master = os.path.join(out_dir, "master.m3u8")
    with open(master, "w") as f:
        f.write("#EXTM3U\n#EXT-X-VERSION:3\n")
        for bw, res, name in variant_playlists:
            f.write(f"#EXT-X-STREAM-INF:BANDWIDTH={bw},RESOLUTION={res}\n{name}/index.m3u8\n")

    return master


def thumbnail(input_path: str, out_path: str, at_seconds: float = 1.0) -> None:
    run(["ffmpeg", "-y", "-ss", str(at_seconds), "-i", input_path,
         "-vframes", "1", "-q:v", "3", out_path])
