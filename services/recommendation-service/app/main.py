"""Recommendation service - personalised feed via LightFM."""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import recommend, train

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rec")

app = FastAPI(title="distrebute.com - Recommendation Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/train")
def train_ep():
    return train()


@app.get("/recommend/{user_id}")
def recommend_ep(user_id: str, top_k: int = 20):
    ids = recommend(user_id, top_k=top_k)
    return {"user_id": user_id, "video_ids": ids}
