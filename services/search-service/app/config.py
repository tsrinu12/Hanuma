from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant:6333"
    # Multilingual mpnet covers 50+ languages with strong cross-lingual retrieval
    # (a Hindi query against English transcripts still works). Override with the
    # smaller paraphrase-multilingual-MiniLM-L12-v2 if you need lower latency
    # at the cost of some recall.
    embed_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    collection: str = "videos"
    vector_size: int = 768


settings = Settings()
