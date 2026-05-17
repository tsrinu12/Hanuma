"""NLLB-200 translator + fastText language identification.

NLLB uses BCP-47-ish flores codes like `eng_Latn`, `hin_Deva`, `spa_Latn`. We
maintain a mapping from ISO-639-1 short codes to the flores codes for
convenience and surface both forms in the API.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    import fasttext  # type: ignore
except ImportError:  # pragma: no cover
    fasttext = None  # type: ignore

from .config import settings

log = logging.getLogger("translation")

# Short, curated map: ISO-639-1 -> NLLB flores-200 code. Add more as needed.
ISO_TO_FLORES = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl", "pl": "pol_Latn", "tr": "tur_Latn", "ar": "arb_Arab",
    "fa": "pes_Arab", "ur": "urd_Arab", "hi": "hin_Deva", "bn": "ben_Beng",
    "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym",
    "mr": "mar_Deva", "gu": "guj_Gujr", "pa": "pan_Guru", "ne": "npi_Deva",
    "si": "sin_Sinh", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "vi": "vie_Latn", "th": "tha_Thai", "id": "ind_Latn", "ms": "zsm_Latn",
    "tl": "tgl_Latn", "sw": "swh_Latn", "am": "amh_Ethi", "ha": "hau_Latn",
    "yo": "yor_Latn", "ig": "ibo_Latn", "zu": "zul_Latn", "he": "heb_Hebr",
    "el": "ell_Grek", "cs": "ces_Latn", "ro": "ron_Latn", "hu": "hun_Latn",
    "sv": "swe_Latn", "da": "dan_Latn", "no": "nob_Latn", "fi": "fin_Latn",
}


def to_flores(code: str) -> str:
    """Accept either an ISO-639-1 code (`en`) or a flores code (`eng_Latn`)."""
    if "_" in code and len(code.split("_")[0]) >= 2:
        return code
    return ISO_TO_FLORES.get(code.lower(), code)


class LangID:
    """fastText lid.176 wrapper."""

    def __init__(self) -> None:
        if fasttext is None:
            log.warning("fasttext not installed; language detection disabled")
            self.model = None
            return
        path = hf_hub_download(repo_id=settings.lang_id_model, filename="model.bin")
        self.model = fasttext.load_model(path)

    def detect(self, text: str, k: int = 1) -> List[Tuple[str, float]]:
        if self.model is None:
            return [("und", 0.0)]
        # fastText doesn't like newlines
        clean = re.sub(r"\s+", " ", text).strip()
        labels, probs = self.model.predict(clean, k=k)
        out = []
        for lab, p in zip(labels, probs):
            iso = lab.replace("__label__", "")
            out.append((iso, float(p)))
        return out


class NLLBTranslator:
    def __init__(self) -> None:
        self.device = torch.device(
            settings.device if torch.cuda.is_available() or settings.device == "cpu" else "cpu"
        )
        log.info("Loading NLLB %s on %s", settings.nllb_model, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.nllb_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.nllb_model,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device).eval()

    @torch.inference_mode()
    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        src = to_flores(src_lang)
        tgt = to_flores(tgt_lang)
        self.tokenizer.src_lang = src
        outputs: List[str] = []
        for i in range(0, len(texts), settings.batch_size):
            batch = texts[i : i + settings.batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=settings.max_input_tokens,
            ).to(self.device)
            gen = self.model.generate(
                **enc,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt),
                max_new_tokens=settings.max_input_tokens,
                num_beams=4,
            )
            outputs.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return outputs
