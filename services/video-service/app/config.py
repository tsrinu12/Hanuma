from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    s3_endpoint_url: str = ""           # blank => real AWS S3
    s3_bucket_raw: str = "distribute-raw-uploads"
    s3_bucket_hls: str = "distribute-hls-public"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    aws_region: str = "us-east-1"
    cloudfront_domain: str = ""

    redis_url: str = "redis://redis:6379/0"
    metadata_url: str = "http://metadata-service:8000"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
