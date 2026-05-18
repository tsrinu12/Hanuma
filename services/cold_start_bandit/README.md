# cold_start_bandit

LinUCB contextual bandit + V-JEPA peer-cluster routing + MMR diversity rerank.

## Why this exists

YouTube's two-tower retrieval is biased against day-1 content — it recommends
videos similar to ones already watched, and brand-new uploads have nothing in
the historical co-watch graph. The result is the "the first 1,000 views are
the hardest" problem.

This service treats every peer cluster as a bandit arm. A new video joins a
cluster (via V-JEPA / content-embedding nearest neighbours) and inherits that
cluster's learned theta the moment it is indexed. Distribution starts on day
one; the bandit explores around it.

## API

| Method | Path                  | Purpose |
| ------ | --------------------- | ------- |
| GET    | `/health`             | liveness probe |
| POST   | `/index_video`        | called by ingest pipeline to register a new upload |
| POST   | `/peer_lookup`        | given an embedding, return nearest neighbours and cluster id |
| POST   | `/rank`               | rank candidates for a user — LinUCB scoring + MMR diversity rerank |
| POST   | `/reward`             | feed back an observed reward (completion, share, etc.) |
| POST   | `/admin/recluster`    | refit k-means and reassign clusters |
| GET    | `/admin/snapshot`     | bandit + index diagnostics |

## How the math hangs together

LinUCB per arm `a`:
```
A_a = I + sum x_t x_t^T,   b_a = sum r_t x_t
theta_a = A_a^-1 b_a
score(a, x) = theta_a . x  +  alpha * sqrt(x^T A_a^-1 x)
                ^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^^
                exploit        explore
```
Arms are peer clusters, not individual videos — that's the cold-start trick.

Context vector `x` per `(user, candidate)`:
```
[ cos(user_emb, content_emb), age buckets, impression buckets, hashed interaction ]
```
See `app/features.py`. Extendable.

MMR rerank over the LinUCB-scored shortlist:
```
MMR(i) = lambda * relevance(i) - (1 - lambda) * max sim(i, selected) - creator_penalty * already_picked
```
See `app/reranker.py`.

## Tuning

- `bandit_alpha`: raise for more exploration. Start at 1.0; the simulator
  (`python -m app.simulator --rounds 5000 --alpha X`) is fast and gives you
  regret ratios in a few seconds.
- `mmr_lambda`: 0.7 is a reasonable default. Lower for more diversity; users
  notice the difference at around 0.5.
- `peer_k_neighbors`: bigger = more averaged warm start but slower per-call.

## Running

```bash
# from repo root
docker build -f services/cold_start_bandit/Dockerfile -t distrebute/cold-start-bandit .
docker run -p 8001:8001 distrebute/cold-start-bandit
```

Or with the whole stack:

```bash
docker compose up cold_start_bandit
```

Tests:

```bash
cd services/cold_start_bandit
pip install -r requirements.txt
pytest
```

## Where this falls short (next iterations)

- Persistence: arm state and index are in-memory. Add a Redis/Postgres
  snapshot loop.
- Sherman-Morrison rank-1 updates would replace `np.linalg.inv` per-update
  for higher `context_dim`.
- LinUCB assumes linear rewards. For very nonlinear interactions, swap
  `bandit.py` for NeuralTS or a deep contextual Thompson sampler — same
  service boundary.
