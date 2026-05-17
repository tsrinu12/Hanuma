#!/usr/bin/env bash
# One-shot AWS bootstrap: S3 buckets, ECR repos, CloudWatch log groups, ECS cluster.
# Run once per environment. Idempotent.
set -euo pipefail

: "${AWS_REGION:?must set AWS_REGION}"
: "${AWS_ACCOUNT_ID:?must set AWS_ACCOUNT_ID}"
CLUSTER="${CLUSTER:-distrebute-prod}"

echo "→ Creating ECS cluster ${CLUSTER} ..."
aws ecs describe-clusters --clusters "$CLUSTER" --region "$AWS_REGION" \
  --query 'clusters[?status==`ACTIVE`].clusterName' --output text | grep -q "$CLUSTER" \
  || aws ecs create-cluster --cluster-name "$CLUSTER" --region "$AWS_REGION" >/dev/null

echo "→ Creating S3 buckets ..."
for b in distribute-raw-uploads distribute-hls-public; do
  aws s3api head-bucket --bucket "$b" 2>/dev/null \
    || aws s3 mb "s3://$b" --region "$AWS_REGION"
done

echo "→ Public-read policy on HLS bucket (CloudFront in front in prod) ..."
cat > /tmp/hls-policy.json <<JSON
{"Version":"2012-10-17","Statement":[{"Sid":"PublicRead","Effect":"Allow","Principal":"*","Action":"s3:GetObject","Resource":"arn:aws:s3:::distribute-hls-public/*"}]}
JSON
aws s3api put-bucket-policy --bucket distribute-hls-public --policy file:///tmp/hls-policy.json || true

echo "→ Creating CloudWatch log groups ..."
for svc in metadata-service video-service ai-service search-service recommendation-service \
           vjepa-service translation-service spam-service nlp-service gamification-service frontend; do
  aws logs create-log-group --log-group-name "/ecs/distrebute/$svc" --region "$AWS_REGION" 2>/dev/null || true
done

echo "✓ Bootstrap complete. Next:"
echo "    AWS_REGION=$AWS_REGION AWS_ACCOUNT_ID=$AWS_ACCOUNT_ID TAG=v1 ./build-and-push-ecr.sh"
echo "    AWS_REGION=$AWS_REGION AWS_ACCOUNT_ID=$AWS_ACCOUNT_ID CLUSTER=$CLUSTER TAG=v1 ./deploy-ecs.sh"
