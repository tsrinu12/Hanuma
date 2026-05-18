# moderation

Cross-modal moderation: fuses text, visual, and audio classifiers into a
single verdict with per-modality, per-timestamp explanations.

## Why fuse modalities

A single-modality classifier misses *multimodal* harm:

- clean transcript, violent imagery onscreen
- clean visuals, racial slur in the audio track
- benign individual signals, but agreement across all three

YouTube's moderation pipeline is famously opaque from the creator side —
strikes show up with little explanation. Our output surface includes the
*modality*, *category*, and *timestamp* that drove the decision, so a
creator can see exactly what fired.

## Fusion

We use weighted noisy-OR per category:

```
s_cat = 1 - prod_modalities (1 - w_m * max_severity_in_category_for_m)
fused = max over categories
```

Decision:

```
fused < allow_below  -> ALLOW
fused < block_above  -> REVIEW
fused >= block_above -> BLOCK
```

Two modalities each at 0.6 in the same category fuse to ~0.84 — pushes
past `block_above` even though no single modality is "confident". That's
the cross-modal agreement boost we want.

## API

```http
POST /moderate
{
  "video_id": "abc",
  "duration_s": 120.0,
  "transcript": [{"start_s": 0, "end_s": 5, "text": "..."}, ...],
  "keyframes":  [{"t_s": 4.0, "embedding": [...]}, ...],
  "audio_segments": [{"start_s": 0, "end_s": 2,
                       "features": [mean_db, std_db, peak_db, percussive, voicing]},
                      ...]
}
```

Response:

```http
{
  "video_id": "abc",
  "decision": "BLOCK",
  "severity": 0.91,
  "flags": [
    {"modality": "text", "category": "violence", "severity": 0.92,
     "start_s": 14.0, "end_s": 18.0,
     "explanation": "transcript: explicit violence threat"},
    {"modality": "visual", "category": "violence", "severity": 0.71,
     "start_s": 15.0, "end_s": 17.0,
     "explanation": "keyframe @ 16.0s: violent imagery"}
  ],
  "per_modality": {"text": 0.92, "visual": 0.71, "audio": 0.0}
}
```

## Model profile

| Modality | Lite                                     | Prod                                |
| -------- | ---------------------------------------- | ----------------------------------- |
| Text     | regex rules over toxic-bert-style labels | unitary/toxic-bert                  |
| Visual   | hash-text anchors vs. caller embeddings  | CLIP ViT-B/32 + NSFW head           |
| Audio    | feature rules (mean/peak/voicing)        | PANNs CNN14 over raw waveform       |

The visual classifier expects keyframe embeddings (precomputed during
ingest), not raw images, in both modes. That keeps the moderation service
out of the GPU-image-decode hot path. Prod can be wired to accept raw
images via the `image_b64` field on `Keyframe`.

## Tuning

- `allow_below` / `block_above` set decision boundaries. Defaults `0.35` /
  `0.85` are intentionally cautious — overshoot moderation, then loosen
  with data.
- `weight_text` / `weight_visual` / `weight_audio` weight each modality in
  the noisy-OR. Lower a modality's weight when its classifier has a known
  false-positive bias.

## Where this falls short

- Audio prod path is stubbed to fall back to lite rules. Wire in PANNs by
  uncommenting the dep and decoding `wave_b64` to log-mel.
- The visual classifier is anchor-based for explainability. For higher
  recall, add a learned head on top of CLIP features (e.g. NSFW detector
  or violence head).
- No language detection — text rules are English-only. Add a language head
  ahead of dispatch.
