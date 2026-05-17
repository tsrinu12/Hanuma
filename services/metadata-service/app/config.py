from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://distribute:changeme@postgres:5432/distribute"
    service_name: str = "metadata-service"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
