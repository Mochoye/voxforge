import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from voxforge.pipeline import VoxForgePipeline
from voxforge.streamer import ChunkedStreamer

def test_streaming():
    print("Testing ChunkedStreamer...\n")

    pipeline = VoxForgePipeline()
    pipeline.load(speaker="Ana Florence")

    # Multi-chunk text so we can observe streaming behavior
    text = (
        "VoxForge streaming engine is now active. "
        "The first chunk is synthesized and ready before the second begins. "
        "This is how low latency streaming works in production TTS systems."
    )

    from voxforge.normalizer import normalize
    from voxforge.chunker import chunk

    normalized = normalize(text)
    chunks = chunk(normalized)

    print(f"Chunks to stream: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] {c}")
    print()

    streamer = ChunkedStreamer(pipeline.engine)

    result = streamer.stream_to_file(
        chunks=chunks,
        gpt_cond_latent=pipeline._gpt_cond_latent,
        speaker_embedding=pipeline._speaker_embedding,
        output_path="outputs/streaming_test.wav",
        on_chunk=lambda i, audio: print(f"  → Chunk {i+1} received by consumer: "
                                         f"{len(audio)/pipeline.engine.SAMPLE_RATE:.2f}s audio")
    )

    print(f"\n--- Streaming Results ---")
    print(f"First chunk latency : {result['first_chunk_latency_sec']:.3f}s")
    print(f"Total audio         : {result['total_audio_sec']:.2f}s")
    print(f"Total time          : {result['total_time_sec']:.2f}s")
    print(f"Overall RTF         : {result['rtf']:.2f}x")
    print(f"\nPer-chunk timings:")
    for t in result['chunk_timings']:
        print(f"  Chunk {t['chunk_index']+1}: "
              f"inference={t['inference_sec']:.2f}s "
              f"audio={t['audio_duration_sec']:.2f}s "
              f"RTF={t['rtf']:.2f}x")

    print(f"\nOutput saved: outputs/streaming_test.wav")
    print("\nPASS: Streaming test complete.")

if __name__ == "__main__":
    test_streaming()