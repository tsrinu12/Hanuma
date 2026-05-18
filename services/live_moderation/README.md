# live_moderation

Streaming ASR + per-chunk toxicity for live video. Designed for ~2-3s
end-to-end latency from sound to moderation action.

## Why

Reactive moderation (review after the fact) is too late for live streaming.
You can blur a clip in VOD; you can't unsay it on a livestream that already
hit 100k viewers. This service classifies as the words are spoken.

## Pipeline

```
audio frames -> [VadChunker] -> chunks (~2.5s, closed on silence)
                                   |
                                   v
                              [Whisper] -> text
                                   |
                                   v
                          [StreamingToxicity]
                                   |
                                   v
                           TRANSCRIPT + FLAG events
                                   |
                                   v
                              WebSocket -> client
```

The chunker holds out for either ~2.5s of accumulated speech or a clean
silence break, so most chunks land at meaningful sentence boundaries.

## Protocol (`/ws/moderate`)

1. Server sends a `STATUS` text frame: `{"event":"STATUS","msg":"ready"}`.
2. Client may send a JSON text frame to declare stream params:
   `{"stream_id":"abc","sample_rate":16000}`. Optional; defaults to 16000.
3. Client streams audio as binary frames — little-endian `float32` mono.
4. Server emits text frames as JSON:
   ```
   {"event":"TRANSCRIPT","chunk_id":0,"start_s":0.0,"end_s":2.5,"text":"..."}
   {"event":"FLAG","chunk_id":0,"start_s":0.0,"end_s":2.5,
    "severity":0.92,"category":"violence","action":"BLOCK",
    "explanation":"violence threat"}
   ```
5. Closing the socket flushes any in-progress chunk.

There's also a `POST /moderate_buffer` endpoint that runs the same pipeline
over a single in-memory buffer — useful for tests and offline replays.

## Model profile

| Component | Lite                  | Prod                              |
| --------- | --------------------- | --------------------------------- |
| VAD       | RMS energy threshold  | Silero VAD / WebRTC VAD           |
| ASR       | deterministic mock    | faster-whisper tiny/medium (int8) |
| Toxicity  | regex rule set        | toxic-bert                        |

In lite mode the service boots without downloading anything; you can wire
the WebSocket to a browser or a streaming-RTMP transcoder and watch
TRANSCRIPT events flow even though the transcripts themselves are mock.
Switch `MODEL_PROFILE=prod` to enable Whisper.

## Latency budget

With faster-whisper `tiny.en` on CPU:

  - VAD chunk close: ~chunk_target_s seconds after speech starts (configurable, 2.5s default)
  - ASR: ~250-400ms per 2.5s chunk on a modern laptop CPU
  - Toxicity (lite): O(1) regex
  - Toxicity (prod, toxic-bert): ~30ms per chunk on CPU

End to end: ~2.5s + 300ms + 30ms ≈ 2.8s from "the word was said" to
"the moderation event fired". On GPU with `medium.en`, the ASR portion
drops to ~80ms, getting you under 2.7s reliably.

## Tuning

- `chunk_target_s`: lower it for tighter latency at the cost of ASR
  accuracy on short chunks.
- `silence_break_s`: tighter values mean chunks close more eagerly mid-
  sentence; looser values mean smoother chunk boundaries but higher tail
  latency.
- `block_above` / `warn_above`: decision thresholds.

## Where this falls short

- Speaker diarization isn't done — for multi-speaker streams (e.g. a
  podcast format), wire in Pyannote and key flags by speaker.
- The pipeline is mono. Stereo channels are averaged; if you have a host
  + caller layout, run two pipelines or split first.
- Backpressure: if the client overruns the ASR, the chunker buffers
  indefinitely. Add a high-watermark drop in front of the chunker if
  you expect that.
