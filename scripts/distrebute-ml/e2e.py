"""End-to-end test harness for the distrebute-ml stack.

Boots each service as a background uvicorn process on localhost, waits for
the /health endpoint to come up, exercises the public API surface with real
HTTP/WebSocket calls, then tears the processes down. Writes a JSON report.

Usage:
    python scripts/e2e.py
"""

from __future__ import annotations

import json
import os
import signal
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from websockets.sync.client import connect as ws_connect


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SERVICES = [
    ("cold_start_bandit", 8001),
    ("semantic_search", 8002),
    ("moderation", 8003),
    ("live_moderation", 8004),
]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    response: Any = None


@dataclass
class ServiceReport:
    service: str
    booted: bool
    checks: list[Check] = field(default_factory=list)

    def passed(self) -> bool:
        return self.booted and all(c.passed for c in self.checks)


def _env_for_service(service: str) -> dict:
    env = os.environ.copy()
    env["MODEL_PROFILE"] = "lite"
    env["SERVICE_NAME"] = service
    env["LOG_LEVEL"] = "WARNING"
    service_root = REPO_ROOT / "services" / service
    env["PYTHONPATH"] = f"{service_root}{os.pathsep}{REPO_ROOT}"
    return env


@contextmanager
def run_service(service: str, port: int):
    env = _env_for_service(service)
    log_path = REPO_ROOT / "scripts" / f"{service}.e2e.log"
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", str(port),
            "--log-level", "warning",
        ],
        cwd=str(REPO_ROOT / "services" / service),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    try:
        deadline = time.time() + 20.0
        booted = False
        while time.time() < deadline:
            try:
                with httpx.Client(trust_env=False, timeout=1.0) as c:
                    r = c.get(f"http://127.0.0.1:{port}/health")
                if r.status_code == 200:
                    booted = True
                    break
            except httpx.RequestError:
                pass
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        yield proc, booted, log_path
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.close()


def _rand_emb(seed: int, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    return v.tolist()


def check_bandit(client: httpx.Client) -> list[Check]:
    out: list[Check] = []
    r = client.get("/health")
    out.append(Check("health", r.status_code == 200 and r.json()["service"] == "cold_start_bandit"))

    indexed = []
    for i in range(8):
        r = client.post("/index_video", json={"video_id": f"v{i}", "content_embedding": _rand_emb(i)})
        indexed.append(r.status_code == 200)
    out.append(Check("index_8_videos", all(indexed)))

    r = client.post("/peer_lookup", json={"content_embedding": _rand_emb(0), "k": 5})
    body = r.json() if r.status_code == 200 else {}
    out.append(Check(
        "peer_lookup_returns_neighbors",
        r.status_code == 200 and "v0" in body.get("neighbors", []),
        detail=f"neighbors={body.get('neighbors', [])[:3]}",
    ))

    candidates = [
        {
            "video_id": f"v{i}", "creator_id": f"c{i}",
            "content_embedding": _rand_emb(i),
            "age_hours": 0.0, "impressions_global": 0,
        }
        for i in range(8)
    ]
    r = client.post("/rank", json={
        "user_id": "u_test", "user_embedding": _rand_emb(0),
        "candidates": candidates, "top_k": 5,
    })
    rb = r.json() if r.status_code == 200 else {}
    out.append(Check(
        "rank_returns_top_k",
        r.status_code == 200 and len(rb.get("results", [])) == 5,
        detail=f"got {len(rb.get('results', []))}",
    ))
    if rb.get("results"):
        first = rb["results"][0]
        out.append(Check(
            "rank_exposes_ucb_decomposition",
            "exploit" in first and "explore" in first and "peer_cluster_id" in first,
            detail=f"keys={list(first.keys())}",
        ))

    ctx = [0.0] * 64
    ctx[0] = 1.0
    r = client.post("/reward", json={
        "user_id": "u_test", "video_id": "v0", "context": ctx, "reward": 1.0,
    })
    rb = r.json() if r.status_code == 200 else {}
    out.append(Check(
        "reward_accepted",
        r.status_code == 200 and rb.get("accepted") is True and rb.get("arm_pulls", 0) >= 1,
        detail=str(rb),
    ))

    r = client.get("/admin/snapshot")
    snap = r.json() if r.status_code == 200 else {}
    out.append(Check(
        "snapshot_reports_pulls",
        snap.get("bandit_pulls", 0) >= 1,
        detail=f"pulls={snap.get('bandit_pulls')} arms={snap.get('bandit_arms')}",
    ))
    return out


def check_search(client: httpx.Client) -> list[Check]:
    out: list[Check] = []
    out.append(Check("health", client.get("/health").status_code == 200))

    payload_a = {
        "video_id": "vidA", "duration_s": 60.0,
        "transcript": [
            {"start_s": 0.0,  "end_s": 5.0,  "text": "Today we discuss pricing strategy for SaaS startups"},
            {"start_s": 5.5,  "end_s": 12.0, "text": "Tiered pricing and freemium are the two main models"},
            {"start_s": 18.0, "end_s": 25.0, "text": "Now switching topics to onboarding flows for new users"},
            {"start_s": 25.5, "end_s": 35.0, "text": "Onboarding determines whether a new signup converts"},
            {"start_s": 50.0, "end_s": 58.0, "text": "Finally lets talk about churn reduction tactics"},
        ],
    }
    payload_b = {
        "video_id": "vidB", "duration_s": 30.0,
        "transcript": [
            {"start_s": 0.0,  "end_s": 10.0, "text": "A football match recap and player statistics"},
            {"start_s": 15.0, "end_s": 25.0, "text": "Discussion of soccer transfers this season"},
        ],
    }
    ra = client.post("/index_video", json=payload_a)
    rb = client.post("/index_video", json=payload_b)
    out.append(Check("index_two_videos", ra.status_code == 200 and rb.status_code == 200))
    if ra.status_code == 200:
        out.append(Check("chapters_returned", len(ra.json()["chapters"]) >= 1,
                         detail=f"{len(ra.json()['chapters'])} chapters"))

    r = client.post("/search", json={"query": "onboarding flows", "top_k": 3})
    hits = r.json().get("hits", []) if r.status_code == 200 else []
    onboarding_hit = next((h for h in hits if h["video_id"] == "vidA"), None)
    out.append(Check(
        "search_finds_onboarding_in_vidA",
        onboarding_hit is not None and onboarding_hit["end_s"] >= 18.0,
        detail=f"top={hits[0] if hits else None}",
    ))

    r = client.post("/search", json={"query": "pricing", "video_id": "vidB", "top_k": 5})
    hits = r.json().get("hits", []) if r.status_code == 200 else []
    out.append(Check(
        "filter_restricts_to_video",
        all(h["video_id"] == "vidB" for h in hits),
        detail=f"vids={[h['video_id'] for h in hits]}",
    ))
    return out


def check_moderation(client: httpx.Client) -> list[Check]:
    out: list[Check] = []
    out.append(Check("health", client.get("/health").status_code == 200))

    clean = {
        "video_id": "clean1", "duration_s": 10.0,
        "transcript": [{"start_s": 0.0, "end_s": 10.0, "text": "a relaxing gardening tutorial"}],
    }
    r = client.post("/moderate", json=clean)
    body = r.json() if r.status_code == 200 else {}
    out.append(Check("clean_is_ALLOW", body.get("decision") == "ALLOW", detail=str(body.get("decision"))))

    harass = {
        "video_id": "h1", "duration_s": 3.0,
        "transcript": [{"start_s": 0.0, "end_s": 3.0, "text": "kys"}],
    }
    r = client.post("/moderate", json=harass)
    body = r.json()
    out.append(Check(
        "harassment_is_BLOCK",
        body.get("decision") == "BLOCK" and body.get("severity", 0) > 0.9,
        detail=f"decision={body.get('decision')} severity={body.get('severity')}",
    ))
    out.append(Check(
        "explanations_have_modality_and_timestamp",
        all("modality" in f and "start_s" in f and "explanation" in f for f in body.get("flags", [])),
    ))

    multi = {
        "video_id": "mm1", "duration_s": 10.0,
        "transcript": [{"start_s": 0.0, "end_s": 10.0, "text": "I will kill them"}],
        "audio_segments": [{
            "start_s": 0.0, "end_s": 2.0,
            "features": [-20.0, 8.0, -1.0, 0.4, 0.05],
        }],
    }
    r = client.post("/moderate", json=multi)
    body = r.json()
    out.append(Check(
        "multimodal_fusion_elevated",
        body.get("severity", 0.0) > 0.7,
        detail=f"severity={body.get('severity'):.3f} decision={body.get('decision')}",
    ))
    out.append(Check(
        "per_modality_breakdown_present",
        set(body.get("per_modality", {}).keys()) == {"text", "visual", "audio"},
    ))
    return out


def check_live_moderation(client: httpx.Client, port: int) -> list[Check]:
    out: list[Check] = []
    out.append(Check("health", client.get("/health").status_code == 200))

    sr = 16000
    n = int(1.5 * sr)
    t = np.arange(n) / sr
    samples = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    samples = np.concatenate([samples, np.zeros(int(1.5 * sr), dtype=np.float32)])
    r = client.post("/moderate_buffer", json={
        "stream_id": "e2e", "sample_rate": sr, "samples": samples.tolist(),
    })
    events = r.json().get("events", []) if r.status_code == 200 else []
    out.append(Check(
        "buffer_emits_transcript",
        any(e["event"] == "TRANSCRIPT" for e in events),
        detail=f"events={[e['event'] for e in events]}",
    ))

    try:
        with ws_connect(f"ws://127.0.0.1:{port}/ws/moderate", open_timeout=5) as ws:
            first = ws.recv(timeout=5)
            got_status = "STATUS" in first
            ws.send(json.dumps({"stream_id": "e2e_ws", "sample_rate": sr}))
            buf = struct.pack(f"<{samples.size}f", *samples)
            ws.send(buf)
            got_transcript = False
            for _ in range(8):
                try:
                    msg = ws.recv(timeout=3)
                except Exception:
                    break
                if "TRANSCRIPT" in msg:
                    got_transcript = True
                    break
            out.append(Check("websocket_status_ready", got_status))
            out.append(Check("websocket_transcript_event", got_transcript))
    except Exception as e:
        out.append(Check("websocket_status_ready", False, detail=f"WS error: {e}"))
        out.append(Check("websocket_transcript_event", False, detail=f"WS error: {e}"))
    return out


CHECK_FNS = {
    "cold_start_bandit": check_bandit,
    "semantic_search": check_search,
    "moderation": check_moderation,
    "live_moderation": check_live_moderation,
}


def main() -> int:
    report: list[ServiceReport] = []
    for service, port in SERVICES:
        with run_service(service, port) as (proc, booted, log_path):
            sr = ServiceReport(service=service, booted=booted)
            if not booted:
                sr.checks.append(Check("boot", False, detail=f"see {log_path}"))
                report.append(sr)
                continue
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0, trust_env=False) as client:
                fn = CHECK_FNS[service]
                if service == "live_moderation":
                    sr.checks.extend(fn(client, port))  # type: ignore[arg-type]
                else:
                    sr.checks.extend(fn(client))
            report.append(sr)

    report_path = REPO_ROOT / "scripts" / "e2e_report.json"
    payload = {
        "services": [
            {
                "service": sr.service,
                "booted": sr.booted,
                "passed": sr.passed(),
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail}
                    for c in sr.checks
                ],
            }
            for sr in report
        ],
        "total_passed": sum(1 for sr in report for c in sr.checks if c.passed),
        "total_failed": sum(1 for sr in report for c in sr.checks if not c.passed),
    }
    report_path.write_text(json.dumps(payload, indent=2))

    print("\n=== e2e summary ===")
    for sr in report:
        status = "PASS" if sr.passed() else "FAIL"
        print(f"  [{status}] {sr.service}  ({sum(1 for c in sr.checks if c.passed)}/{len(sr.checks)} checks)")
        for c in sr.checks:
            if not c.passed:
                print(f"      FAIL: {c.name} :: {c.detail}")
    print(f"\nTotal passed: {payload['total_passed']}  failed: {payload['total_failed']}")
    print(f"Report: {report_path}")

    return 0 if payload["total_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
