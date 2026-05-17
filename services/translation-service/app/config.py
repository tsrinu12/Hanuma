"""Settings for translation-service."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # NLLB-200: Meta's 200-language translation model. The 3.3B variant is the
    # highest-quality production option; use distilled-1.3B or distilled-600M
    # for smaller GPUs.
    nllb_model: str = os.getenv("NLLB_MODEL", "facebook/nllb-200-3.3B")
    # fastText lid.176: 176-language identification model
    lang_id_model: str = os.getenv("LANG_ID_MODEL", "facebook/fasttext-language-identification")
    device: str = os.getenv("DEVICE", "cuda")
    max_input_tokens: int = int(os.getenv("MAX_INPUT_TOKENS", "1024"))
    batch_size: int = int(os.getenv("TRANSLATE_BATCH_SIZE", "8"))


settings = Settings()
