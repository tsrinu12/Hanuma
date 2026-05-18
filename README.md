# distrebute.com — AI Video Streaming Platform

A microservices video streaming platform with AI features (Whisper transcription,
NSFW moderation, LLM summarisation, semantic search, personalised recommendations),
built from the PRD. Runs locally with Docker Compose; deploys to AWS via ECS Fargate
or EKS.

## Architecture

```
                ┌──────────────────┐
                │  React Frontend  │ (Shaka Player, Vite, Nginx)
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │   API Gateway    │ Nginx → ALB in prod
                └────────┬─────────┘
                         │
   ┌──────────┬──────────┼───────────┬──────────────┬──────────────┐
   │          │          │           │              │              │
┌──▼──┐  ┌────▼────┐  ┌──▼──┐   ┌────▼─────┐  ┌─────▼──────┐  ┌────▼──────┐
│auth │  │metadata │  │video│   │   ai     │  │  search    │  │recommend  │
│KCloak│ │ FastAPI │  │FastAPI│ │FastAPI + │  │ FastAPI    │  │ FastAPI   │
│      │ │+ RDS    │  │+ ffmpeg│ │Whisper   │  │+SentenceTr.│  │+ LightFM  │
└──────┘ └─────────┘  └───┬──┘   │+ Bedrock │  └─────┬──────┘  └─────┬─────┘
                          │      └─────┬────┘        │               │
                          ▼            ▼             ▼               ▼
                        S3 raw   Redis queue     Qdrant         Redis cache
                          │      ai_jobs
                          ▼
                      FFmpeg HLS → S3 public → CloudFront
```

Services (all listening on port 8000 in-container):

| Service | Purpose | Key tech |
|---|---|---|
| **frontend** | React SPA with Shaka Player | Vite + Nginx |
| **auth-service** | OIDC / JWT | Keycloak |
| **metadata-service** | Source of truth: videos, users, transcripts, AI outputs, watch events | FastAPI + SQLAlchemy + RDS Postgres |
| **video-service** | Upload to S3, FFmpeg HLS transcode, enqueue AI | FastAPI + boto3 + FFmpeg |
| **ai-service** | Whisper ASR, NSFW moderation, BART/GPT summary; runs background worker on Redis queue `ai_jobs` | FastAPI + faster-whisper + transformers + opennsfw2 |
| **search-service** | **Multilingual** semantic search over transcript chunks (50+ langs via mpnet) | FastAPI + sentence-transformers + Qdrant |
| **recommendation-service** | Personalised feed, nightly LightFM training | FastAPI + LightFM + Redis |
| **vjepa-service** | V-JEPA 2 video understanding: action recognition + visual embeddings for similar-video retrieval | FastAPI + transformers + decord + Qdrant |
| **translation-service** | 200-language translation + language detection | FastAPI + NLLB-200-3.3B + fastText lid.176 |
| **spam-service** | Multi-head spam/abuse pipeline: toxic-bert + dehatebert + zero-shot + URL/rate/caps heuristics | FastAPI + transformers + Redis |
| **nlp-service** | Sentiment, emotion, NER, keyphrases, topic modeling, summarization | FastAPI + transformers + BERTopic + KeyBERT |
| **gamification-service** | Viewer counts, points (view/complete/like/share/upload + first-of-day + streak bonus), daily streaks, leaderboard, badges | FastAPI + SQLAlchemy + Postgres + Redis |

## Local development (Docker Compose)

```bash
cd ai-video-platform
cp .env.example .env
docker compose build
docker compose up -d
```

URLs:
- Frontend          → http://localhost:3000
- API gateway       → http://localhost:8080  (`/metadata`, `/video`, `/ai`, `/search`, `/recommend`, `/vjepa`, `/translate`, `/spam`, `/nlp`, `/points`)
- Keycloak admin    → http://localhost:8081  (admin/admin)
- MinIO console     → http://localhost:9001  (minioadmin/minioadmin)
- Qdrant            → http://localhost:6333

Upload via curl:
```bash
curl -F "title=Demo" -F "description=hello" -F "file=@/path/to/clip.mp4" \
     http://localhost:8080/video/videos/upload
```
The video is uploaded to MinIO, transcoded to HLS, transcribed, moderated, summarised,
and indexed for semantic search. Watch it at `http://localhost:3000/watch/<id>`.

## AWS production deployment

### 0. Prereqs
- AWS account, IAM admin
- AWS CLI v2 logged in
- Docker buildx (for `--platform linux/amd64`)

### 1. Bootstrap AWS
```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
cd infra/scripts && ./bootstrap-aws.sh
```
Creates: ECS cluster `distrebute-prod`, S3 buckets `distribute-raw-uploads` /
`distribute-hls-public`, ECR repos, CloudWatch log groups.

Manually also provision (with Terraform/CDK/console):
- **RDS Postgres 16** (`db.t3.medium` to start)
- **ElastiCache Redis** (`cache.t3.small`)
- **CloudFront distribution** in front of `distribute-hls-public`
- **Secrets Manager** entries: `distrebute/db-url`, `distrebute/openai-api-key`
- **IAM roles**:
  - `ecsTaskExecutionRole` (managed `AmazonECSTaskExecutionRolePolicy`)
  - `distrebute-video-task` (S3 PutObject/GetObject on the two buckets)
  - `distrebute-ai-task` (S3 GetObject on raw bucket, Bedrock/Transcribe if used)
  - `distrebute-metadata-task` (RDS connect only)
- **Cloud Map namespace** `distrebute.local` (used by ECS service discovery)

### 2. Build & push images to ECR
```bash
TAG=v1 ./build-and-push-ecr.sh
```

### 3. Deploy to ECS Fargate
```bash
CLUSTER=distrebute-prod TAG=v1 ./deploy-ecs.sh
```
Then add an **Application Load Balancer** with these target groups & path rules:
| Path | Target group |
|---|---|
| `/metadata/*` | metadata-service:8000 |
| `/video/*` | video-service:8000 |
| `/ai/*` | ai-service:8000 |
| `/search/*` | search-service:8000 |
| `/recommend/*` | recommendation-service:8000 |
| `/vjepa/*` | vjepa-service:8000 |
| `/translate/*` | translation-service:8000 |
| `/spam/*` | spam-service:8000 |
| `/nlp/*` | nlp-service:8000 |
| `/points/*` | gamification-service:8000 |
| `/auth/*` | keycloak:8080 |
| `/*` (default) | frontend:80 |

### 4. (Alternative) Deploy to EKS
```bash
# Point your kubeconfig at the EKS cluster, then:
export ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
export TAG=v1
envsubst < infra/k8s/20-metadata-service.yaml | kubectl apply -f -
# Or render all at once:
for f in infra/k8s/*.yaml; do envsubst < "$f" | kubectl apply -f -; done
```
The Ingress uses the AWS Load Balancer Controller — install it first
(`helm install aws-load-balancer-controller eks/aws-load-balancer-controller`).
Provision an ACM certificate and uncomment the `certificate-arn` annotation in
`30-ingress.yaml` for HTTPS.

## Tuning & ops notes

- **Whisper model size** is controlled by `WHISPER_MODEL` env (`tiny|base|small|medium|large-v3|turbo`). On Fargate CPU, `base` is the sweet spot; for `large-v3` use an EKS GPU node group.
- **CDN**: in prod, point `CLOUDFRONT_DOMAIN` at your CloudFront distribution; the video-service will rewrite HLS URLs to it.
- **Search**: Qdrant is run as a StatefulSet on EKS. For ECS, run it as a sidecar task or use Qdrant Cloud.
- **Recommendation training**: nightly CronJob (`infra/k8s/24-...`) calls `POST /train`. On ECS, schedule with EventBridge → ECS RunTask.
- **Scaling**: HPA on `ai-service` (CPU 65%). ECS Fargate uses Application Auto Scaling on the same target.
- **Live streaming** (PRD §Scalability): for live, swap the upload path for an AWS MediaLive ingest → MediaPackage origin and pipe the audio to an additional Whisper streaming worker. The frontend already plays HLS, so the viewer path is unchanged.

---

## distrebute-ml — four new ML/DL services (this branch)

The `feat/distrebute-ml-services` branch adds four self-contained FastAPI
services targeting structural weaknesses in major video platforms. They sit
alongside the existing platform services, do not modify them, and use a
host-port range (8011-8014) that doesn't collide with the existing
8001-8010 assignments.

| Service | Host port | Algorithm |
| --- | --- | --- |
| [`services/cold_start_bandit`](services/cold_start_bandit/) | 8011 | LinUCB contextual bandit + V-JEPA peer-cluster routing + MMR diversity rerank |
| [`services/semantic_search`](services/semantic_search/) | 8012 | TransNetV2 shot detection + chapter assembly + ColBERT-style late-interaction retrieval |
| [`services/moderation`](services/moderation/) | 8013 | Cross-modal ensemble (toxic-bert + CLIP + PANNs) with weighted noisy-OR fusion |
| [`services/live_moderation`](services/live_moderation/) | 8014 | faster-whisper + VAD chunking + per-chunk toxicity over WebSocket |

Each service has its own `README.md`, Dockerfile, `requirements.txt`, and
pytest suite. End-to-end harness at [`scripts/distrebute-ml/e2e.py`](scripts/distrebute-ml/e2e.py).

CI: [`.github/workflows/ci-distrebute-ml.yml`](.github/workflows/ci-distrebute-ml.yml)
runs unit tests, e2e, Docker builds, and compose-config validation on every
push.

Quick start:

```bash
make install
make test        # 76 unit tests across all four services
make e2e         # 22 e2e checks against live HTTP/WebSocket
MODEL_PROFILE=lite docker compose up cold_start_bandit semantic_search moderation live_moderation
```

These services are independent of the existing recommendation-service,
search-service, etc. — they ship together and can be wired into your
gateway when you're ready.
