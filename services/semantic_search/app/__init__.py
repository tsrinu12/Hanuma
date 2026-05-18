"""Semantic in-video search service.

Pipeline:
  video -> TransNetV2 shot detection -> scene clustering -> chapter titling
        -> per-chapter ColBERT-style embeddings -> late-interaction retrieval
"""
