from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    s3_endpoint_url: str = ""
    s3_bucket_raw: str = "distribute-raw-uploads"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    aws_region: str = "us-east-1"

    redis_url: str = "redis://redis:6379/0"
    metadata_url: str = "http://metadata-service:8000"
    search_url: str = "http://search-service:8000"

    whisper_model: str = "base"          # tiny|base|small|medium|large-v3|turbo
    summary_model: str = "facebook/bart-large-cnn"
    openai_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
