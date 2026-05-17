from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://redis:6379/0"
    metadata_url: str = "http://metadata-service:8000"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
