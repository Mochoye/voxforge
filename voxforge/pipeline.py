import time
import soundfile as sf
import numpy as np
from pathlib import Path

from voxforge.normalizer import normalize
from voxforge.chunker import chunk
from voxforge.engine import TTSEngine

from pathlib import Path

class VoxForgePipeline:
    """
    End-to-end TTS pipeline for VoxForge.

    Usage:
        pipeline = VoxForgePipeline()
        pipeline.load()
        result = pipeline.synthesize("Hello world.", "outputs/hello.wav")
    """

    def __init__(self, device: str = None, use_fp16: bool = True):
        self.engine = TTSEngine(device=device, use_fp16=use_fp16)
        self._gpt_cond_latent = None
        self._speaker_embedding = None
        self._speaker_name = None

    def load(self, speaker: str = "Ana Florence"):
        """
        Load the model and set the default speaker.
        Call this once at startup — it takes ~5–10 seconds.
        """
        self.engine.load()
        self.set_speaker(speaker)

    def set_speaker(self, speaker_name: str):
        """Switch to a different built-in speaker."""
        print(f"[Pipeline] Loading speaker embedding: {speaker_name}")
        self._gpt_cond_latent, self._speaker_embedding = \
            self.engine.get_builtin_speaker_embedding(speaker_name)
        self._speaker_name = speaker_name
        print(f"[Pipeline] Speaker ready: {speaker_name}")

    def set_speaker_from_audio(
        self,
        audio_path: str,
        apply_denoising: bool = True,
        force_reprocess: bool = False,
    ) -> dict:
        """
        Clone a voice from a reference audio file.

        Steps:
            1. Hash the file (cache key)
            2. Check shelve cache — if hit, load and return immediately
            3. If miss: preprocess audio → extract embedding → cache it

        Args:
            audio_path      : Path to reference WAV/MP3 file
            apply_denoising : Run DeepFilterNet before embedding extraction
            force_reprocess : Ignore cache and re-extract even if cached

        Returns:
            report dict with validation info and cache status
        """
        from voxforge.audio_processor import process_reference_audio, file_hash
        from voxforge.speaker_cache import SpeakerCache

        audio_path = str(audio_path)
        cache = SpeakerCache()

        # Step 1: Hash
        audio_hash = file_hash(audio_path)
        print(f"[Pipeline] Reference audio hash: {audio_hash[:12]}...")

        # Step 2: Cache check
        if not force_reprocess:
            cached = cache.get(audio_hash, device=self.engine.device)
            if cached is not None:
                self._gpt_cond_latent = cached["gpt_cond_latent"]
                self._speaker_embedding = cached["speaker_embedding"]
                self._speaker_name = f"cached:{Path(audio_path).stem}"
                print(f"[Pipeline] Cache HIT — loaded embedding from disk.")
                return {
                    "cache_hit": True,
                    "source_file": audio_path,
                    "hash": audio_hash[:12],
                    "duration": cached["duration"],
                    "voice_ratio": cached["voice_ratio"],
                }

        print(f"[Pipeline] Cache MISS — processing reference audio...")

        # Step 3a: Preprocess
        processed_path, report = process_reference_audio(
            audio_path,
            apply_denoising=apply_denoising,
        )

        # Step 3b: Extract embedding
        gpt_cond_latent, speaker_embedding = \
            self.engine.get_speaker_embedding_from_audio(processed_path)

        # Step 3c: Store in cache
        cache.set(
            audio_hash,
            gpt_cond_latent,
            speaker_embedding,
            metadata={
                "source_file": audio_path,
                "duration": report["duration"],
                "voice_ratio": report["voice_ratio"],
            }
        )

        # Step 3d: Set as active speaker
        self._gpt_cond_latent = gpt_cond_latent
        self._speaker_embedding = speaker_embedding
        self._speaker_name = f"cloned:{Path(audio_path).stem}"

        report["cache_hit"] = False
        report["hash"] = audio_hash[:12]
        return report
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        verbose: bool = True,
    ) -> dict:
        """
        Full pipeline: raw text → WAV file on disk.

        Args:
            text:        Raw input text (unnormalized)
            output_path: Where to save the output WAV file
            language:    Language code for synthesis
            verbose:     Print timing breakdown

        Returns:
            result dict with keys:
              normalized_text, chunks, output_path,
              timings (per stage), total_audio_duration_sec
        """
        if self._gpt_cond_latent is None:
            raise RuntimeError("No speaker loaded. Call pipeline.load() first.")

        total_start = time.perf_counter()

        # --- Stage 1: Normalization ---
        t0 = time.perf_counter()
        normalized = normalize(text)
        norm_time = time.perf_counter() - t0

        # --- Stage 2: Chunking ---
        t0 = time.perf_counter()
        chunks = chunk(normalized)
        chunk_time = time.perf_counter() - t0

        if verbose:
            print(f"\n[Pipeline] Input text: '{text[:80]}{'...' if len(text) > 80 else ''}'")
            print(f"[Pipeline] Normalized: '{normalized[:80]}{'...' if len(normalized) > 80 else ''}'")
            print(f"[Pipeline] Chunks ({len(chunks)}): {chunks}")
            print(f"[Pipeline] Norm time: {norm_time*1000:.1f}ms | Chunk time: {chunk_time*1000:.1f}ms")

        # --- Stage 3: TTS Inference ---
        waveform, chunk_timings = self.engine.synthesize_chunks(
            chunks=chunks,
            gpt_cond_latent=self._gpt_cond_latent,
            speaker_embedding=self._speaker_embedding,
            language=language,
        )

        # --- Stage 4: Save audio ---
        t0 = time.perf_counter()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), waveform, self.engine.SAMPLE_RATE)
        save_time = time.perf_counter() - t0

        total_time = time.perf_counter() - total_start
        total_audio = sum(t["audio_duration_sec"] for t in chunk_timings)
        total_inference = sum(t["inference_sec"] for t in chunk_timings)

        result = {
            "normalized_text": normalized,
            "chunks": chunks,
            "output_path": str(output_path),
            "timings": {
                "normalization_ms": round(norm_time * 1000, 2),
                "chunking_ms": round(chunk_time * 1000, 2),
                "inference_sec": round(total_inference, 2),
                "save_ms": round(save_time * 1000, 2),
                "total_sec": round(total_time, 2),
            },
            "total_audio_duration_sec": round(total_audio, 2),
            "speaker": self._speaker_name,
        }

        if verbose:
            print(f"\n[Pipeline] ✓ Done.")
            print(f"[Pipeline]   Audio duration : {total_audio:.2f}s")
            print(f"[Pipeline]   Inference time : {total_inference:.2f}s")
            print(f"[Pipeline]   Total time     : {total_time:.2f}s")
            print(f"[Pipeline]   Saved to       : {output_path}")

        return result