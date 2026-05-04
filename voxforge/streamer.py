"""
VoxForge Phase 3 — Chunked Streaming Engine

Implements a producer-consumer pipeline where:
  - Producer: TTS inference generates audio chunks sequentially
  - Consumer: receives chunks as they're ready (simulates streaming playback)

This hides synthesis latency behind playback — by the time the user
finishes hearing chunk N, chunk N+1 is already synthesized.

In Phase 4 this becomes the SSE streaming backend for the FastAPI server.
"""

import time
import queue
import threading
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Generator, Callable


# Sentinel value that signals the consumer the stream is finished
_STREAM_END = object()


class ChunkedStreamer:
    """
    Producer-consumer streaming engine for VoxForge.

    Producer thread: runs TTS inference chunk by chunk, pushes
                     audio arrays into a thread-safe queue.

    Consumer: pulls audio chunks from the queue as they arrive.
              In Phase 4, this becomes an SSE endpoint that pushes
              WAV bytes to the client in real time.

    Usage:
        streamer = ChunkedStreamer(engine)
        for chunk_audio in streamer.stream(chunks, gpt_cond_latent, speaker_embedding):
            # chunk_audio is a numpy float32 array
            # play it, write it, or push it over the network
            pass
    """

    # Silence inserted between chunks (seconds)
    CHUNK_SILENCE_SEC = 0.05

    def __init__(self, engine, queue_maxsize: int = 4):
        """
        Args:
            engine       : loaded TTSEngine instance
            queue_maxsize: max chunks buffered ahead. 4 is safe for RTX 3050.
                           Higher = more memory, lower = more stalling.
        """
        self.engine = engine
        self.queue_maxsize = queue_maxsize
        self._timings = []

    def _producer(
        self,
        chunks: list[str],
        gpt_cond_latent,
        speaker_embedding,
        language: str,
        audio_queue: queue.Queue,
    ):
        """
        Runs in a background thread.
        Synthesizes chunks sequentially and pushes results to the queue.
        """
        silence = np.zeros(
            int(self.CHUNK_SILENCE_SEC * self.engine.SAMPLE_RATE),
            dtype=np.float32
        )

        for i, chunk in enumerate(chunks):
            try:
                t0 = time.perf_counter()

                waveform, timings = self.engine.synthesize_chunk(
                    text=chunk,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    language=language,
                )

                elapsed = time.perf_counter() - t0
                self._timings.append({
                    "chunk_index": i,
                    "text": chunk[:60],
                    "inference_sec": timings["inference_sec"],
                    "audio_duration_sec": timings["audio_duration_sec"],
                    "rtf": timings["realtime_factor"],
                    "queue_wait_sec": round(elapsed - timings["inference_sec"], 3),
                })

                print(f"[Streamer] Chunk {i+1}/{len(chunks)} ready "
                      f"({timings['audio_duration_sec']:.2f}s audio, "
                      f"RTF {timings['realtime_factor']:.2f}x) "
                      f"→ queued")

                # Push audio to queue (blocks if queue is full)
                if i < len(chunks) - 1:
                    audio_queue.put(np.concatenate([waveform, silence]))
                else:
                    audio_queue.put(waveform)

            except Exception as e:
                print(f"[Streamer] Error on chunk {i+1}: {e}")
                audio_queue.put(_STREAM_END)
                return

        # Signal end of stream
        audio_queue.put(_STREAM_END)

    def stream(
        self,
        chunks: list[str],
        gpt_cond_latent,
        speaker_embedding,
        language: str = "en",
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream synthesized audio chunks as they become available.

        Yields numpy float32 arrays, one per synthesized chunk.
        The first chunk is yielded as soon as it's synthesized —
        not after all chunks are done.

        Args:
            chunks           : list of text chunks (from chunker.py)
            gpt_cond_latent  : speaker conditioning tensor
            speaker_embedding: speaker embedding tensor
            language         : language code

        Yields:
            np.ndarray — float32 audio samples at engine.SAMPLE_RATE
        """
        if not chunks:
            return

        self._timings = []
        audio_queue = queue.Queue(maxsize=self.queue_maxsize)

        # Start producer in background thread
        producer_thread = threading.Thread(
            target=self._producer,
            args=(chunks, gpt_cond_latent, speaker_embedding, language, audio_queue),
            daemon=True,
        )
        producer_thread.start()

        # Consume chunks as they arrive
        first_chunk_time = None
        stream_start = time.perf_counter()

        while True:
            chunk_audio = audio_queue.get()

            if chunk_audio is _STREAM_END:
                break

            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - stream_start
                print(f"[Streamer] First chunk latency: {first_chunk_time:.3f}s")

            yield chunk_audio

        producer_thread.join()

    def stream_to_file(
        self,
        chunks: list[str],
        gpt_cond_latent,
        speaker_embedding,
        output_path: str,
        language: str = "en",
        on_chunk: Callable[[int, np.ndarray], None] = None,
    ) -> dict:
        """
        Stream synthesis and write all chunks to a WAV file.

        This is the Phase 3 demo mode — it shows streaming behavior
        (first chunk latency, progressive synthesis) while producing
        a complete output file.

        Args:
            chunks           : text chunks
            gpt_cond_latent  : speaker conditioning
            speaker_embedding: speaker embedding
            output_path      : where to save the final WAV
            language         : language code
            on_chunk         : optional callback(chunk_index, audio_array)
                               called each time a chunk is ready

        Returns:
            dict with timing stats
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_audio = []
        chunk_index = 0
        stream_start = time.perf_counter()
        first_chunk_latency = None

        for chunk_audio in self.stream(
            chunks, gpt_cond_latent, speaker_embedding, language
        ):
            if first_chunk_latency is None:
                first_chunk_latency = time.perf_counter() - stream_start

            all_audio.append(chunk_audio)

            if on_chunk:
                on_chunk(chunk_index, chunk_audio)

            chunk_index += 1

        if not all_audio:
            raise RuntimeError("No audio was generated.")

        # Concatenate and save
        full_audio = np.concatenate(all_audio)
        sf.write(str(output_path), full_audio, self.engine.SAMPLE_RATE)

        total_time = time.perf_counter() - stream_start
        total_audio = len(full_audio) / self.engine.SAMPLE_RATE

        return {
            "output_path": str(output_path),
            "total_audio_sec": round(total_audio, 2),
            "total_time_sec": round(total_time, 2),
            "first_chunk_latency_sec": round(first_chunk_latency, 3),
            "rtf": round(total_audio / total_time, 3),
            "chunk_timings": self._timings,
        }