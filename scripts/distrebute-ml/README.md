# scripts/

Operational scripts that sit outside any one service.

## `e2e.py`

End-to-end test harness. Boots all four services in lite mode against
localhost, then exercises each with real HTTP/WebSocket traffic. Writes a
JSON report to `scripts/e2e_report.json` and per-service logs to
`scripts/<service>.e2e.log`.

```bash
python scripts/e2e.py
```

Exit code: 0 if everything passed, 1 otherwise. CI-friendly.

What it covers per service:

- **cold_start_bandit**: index 8 videos, peer_lookup, rank (top-5 with UCB
  decomposition), reward feedback, admin snapshot.
- **semantic_search**: index two videos with distinct topics, global
  semantic query ("onboarding flows"), video-id-filtered query.
- **moderation**: clean -> ALLOW, harassment -> BLOCK, multimodal
  fusion elevates severity (text+audio agreement past block threshold),
  per-modality breakdown surfaced.
- **live_moderation**: buffer endpoint emits TRANSCRIPT, WebSocket round
  trip with STATUS handshake and TRANSCRIPT event.

## `push_to_github.sh` / `push_to_github.ps1`

Initializes the local repo (if needed) and pushes to a GitHub remote.
**Does not accept tokens as command-line arguments** — that would put the
token in shell history and `ps` output. It relies on git's credential
helper so when GitHub prompts, the token only lives in memory for the
push duration.

```bash
# macOS / Linux
./scripts/push_to_github.sh tsrinu12/Hanuma

# Windows PowerShell
.\scripts\push_to_github.ps1 -Repo tsrinu12/Hanuma
```

### Recommended PAT shape

When GitHub prompts, paste a **fine-grained personal access token** scoped
to the single target repo with `Contents: read & write`. Classic PATs grant
all-repo access and are riskier.

### If you've already exposed a token

Revoke it at https://github.com/settings/tokens (or
https://github.com/settings/personal-access-tokens/fine-grained for
fine-grained ones), then issue a fresh one before running the push.
