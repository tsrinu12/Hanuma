"""Live moderation service.

WebSocket pipe:
    upstream (audio frames) -> VAD chunking -> Whisper -> per-chunk toxicity
                                                     -> moderation event stream
"""
