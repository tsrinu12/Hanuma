# semantic_search

In-video semantic search. Given a free-form query like
*"the part where they talk about pricing"*, return the right
`(video_id, start_s, end_s, title)`.

## Pipeline

```
video -> [TransNetV2] -> shots -> [chapter assembler] -> chapters
                                           |
                                           v
                              [embedder] -> pooled vec
                                         -> token vecs (n, d)
                                           |
                                           v
                              [LateInteractionIndex]
                                           ^
   query --[embedder]------------- pooled + token vecs

   search:
     stage 1: pooled @ chapter_pooled  -> top-N coarse
     stage 2: ColBERT MaxSim re-rank   -> top-K
```

## Why ColBERT / late interaction

Pooled-vector retrieval ("two-tower") collapses query and document into a
single point each. That works for "find similar items" but it's
under-expressive for natural-language fragments — *"the segment where they
debunked the rumor"* shares only one or two tokens with the matching chapter.

Late-interaction (MaxSim) keeps token-level vectors on both sides and lets
each query token find its best match independently. The cost is roughly
`m * n` dot products per candidate, which is why we two-stage: pool-vector
prefilter narrows the candidate set first.

## API

| Method | Path                  | Purpose |
| ------ | --------------------- | ------- |
| GET    | `/health`             | liveness |
| POST   | `/index_video`        | ingest a video; optional shot boundaries; returns chapters |
| POST   | `/search`             | semantic query, optional `video_id` filter |
| GET    | `/admin/snapshot`     | index size and profile |

`index_video` accepts either:
- precomputed `shots` (from TransNetV2 in your ingest pipeline), or
- nothing — the service synthesizes shots from transcript pauses

It also accepts optional per-shot keyframe embeddings (e.g. CLIP). When
present, the chapter assembler uses cosine distance over shot embeddings to
decide where to break chapters; otherwise it falls back to length-capped
merging.

## Model profile

| Component         | Lite                          | Prod                                |
| ----------------- | ----------------------------- | ----------------------------------- |
| Embedder          | hash-based, deterministic     | BGE-small (or BGE-large)            |
| Shot detection    | transcript-pause heuristic    | TransNetV2 (in the ingest pipeline) |
| Chapter title     | extractive first sentence     | T5-small generator (drop-in)        |

Lite is enough to exercise the algorithms in CI; prod needs `transformers` +
`torch` installed (commented out in `requirements.txt`).

## Tuning

- `chapter_merge_distance`: smaller -> more chapters, finer-grained search.
- `max_chapter_seconds`: hard upper bound; reasonable default 180s.
- `max_tokens_per_chapter`: bigger = better recall, slower MaxSim. 64 is a
  good starting point for transcript-grounded chapters.

## Where this falls short

- Index is in-memory. For >10M chapters use a PLAID-style on-disk store.
- The hash embedder is great for tests but useless for actual search; flip
  `MODEL_PROFILE=prod` and provide GPU-or-CPU for inference.
- We don't index visual tokens yet — for "find the part where the slide
  shows a graph" we'd add CLIP token embeddings concatenated alongside text.
