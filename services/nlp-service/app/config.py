"""Settings for nlp-service."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Sentiment: CardiffNLP twitter-roberta-base (3-class: negative/neutral/positive)
    sentiment_model: str = os.getenv(
        "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    # NER: BERT large fine-tuned on CoNLL-2003 (PER / ORG / LOC / MISC)
    ner_model: str = os.getenv("NER_MODEL", "dslim/bert-large-NER")
    # Emotion: GoEmotions / DistilRoBERTa (27 emotion categories)
    emotion_model: str = os.getenv(
        "EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"
    )
    # Embedding backbone for KeyBERT + BERTopic — multilingual
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    # Summarization fallback for non-English (Helsinki / mBART)
    summary_model: str = os.getenv("SUMMARY_MODEL", "facebook/bart-large-cnn")

    device: str = os.getenv("DEVICE", "cuda")
    max_topics: int = int(os.getenv("MAX_TOPICS", "10"))


settings = Settings()
