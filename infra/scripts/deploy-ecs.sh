#!/usr/bin/env bash
# Register ECS task definitions and update services on a Fargate cluster.
# Usage:
#   AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789012 \
#   CLUSTER=distrebute-prod TAG=v1 ./deploy-ecs.sh
set -euo pipefail

: "${AWS_REGION:?must set AWS_REGION}"
: "${AWS_ACCOUNT_ID:?must set AWS_ACCOUNT_ID}"
: "${CLUSTER:?must set CLUSTER}"
TAG="${TAG:-latest}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TD_DIR="$ROOT/infra/ecs/task-definitions"

SERVICES=(metadata-service video-service ai-service search-service recommendation-service \
          vjepa-service translation-service spam-service nlp-service gamification-service frontend)

for svc in "${SERVICES[@]}"; do
  echo "→ Rendering & registering task def for $svc ..."
  RENDERED="$(mktemp).json"
  sed -e "s/\${AWS_ACCOUNT_ID}/$AWS_ACCOUNT_ID/g" \
      -e "s/\${AWS_REGION}/$AWS_REGION/g" \
      -e "s/\${TAG}/$TAG/g" \
      "$TD_DIR/$svc.json" > "$RENDERED"
  ARN=$(aws ecs register-task-definition \
        --region "$AWS_REGION" \
        --cli-input-json "file://$RENDERED" \
        --query 'taskDefinition.taskDefinitionArn' --output text)
  echo "   Registered: $ARN"

  echo "→ Updating service $svc on cluster $CLUSTER ..."
  aws ecs update-service \
    --region "$AWS_REGION" \
    --cluster "$CLUSTER" \
    --service "distrebute-$svc" \
    --task-definition "$ARN" \
    --force-new-deployment >/dev/null
done

echo "✓ Deploy complete. Tracking rollout:"
echo "    aws ecs describe-services --cluster $CLUSTER --services distrebute-<svc> --region $AWS_REGION"
