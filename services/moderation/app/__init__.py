"""Cross-modal moderation service.

Fuses text (toxic-bert), visual (CLIP-based), and audio (PANNs/YAMNet)
classifiers into a single moderation verdict — with per-modality, per-timestamp
explanations so creators see *why* a decision fired.
"""
