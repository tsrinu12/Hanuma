#!/usr/bin/env bash
# Build all service Docker images and push to ECR.
# Usage:
#   AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789012 TAG=v1 ./build-and-push-ecr.sh
set -euo pipefail

: "${AWS_REGION:?must set AWS_REGION}"
: "${AWS_ACCOUNT_ID:?must set AWS_ACCOUNT_ID}"
TAG="${TAG:-latest}"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

SERVICES=(metadata-service video-service ai-service search-service recommendation-service \
          vjepa-service translation-service spam-service nlp-service gamification-service)

echo "→ Logging into ECR ${REGISTRY} ..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

ensure_repo() {
  local repo="distrebute/$1"
  aws ecr describe-repositories --repository-names "$repo" --region "$AWS_REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$repo" --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true >/dev/null
}

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

for svc in "${SERVICES[@]}"; do
  ensure_repo "$svc"
  echo "→ Building $svc ..."
  docker build --platform linux/amd64 -t "$REGISTRY/distrebute/$svc:$TAG" "$ROOT/services/$svc"
  docker push "$REGISTRY/distrebute/$svc:$TAG"
done

# Frontend
ensure_repo "frontend"
echo "→ Building frontend ..."
docker build --platform linux/amd64 \
  --build-arg "VITE_API_BASE=https://api.distrebute.com" \
  -t "$REGISTRY/distrebute/frontend:$TAG" "$ROOT/frontend"
docker push "$REGISTRY/distrebute/frontend:$TAG"

echo "✓ All images pushed to $REGISTRY (tag: $TAG)"
