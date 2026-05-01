import time
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class TTSEngine:
    """
    Low-level XTTS-v2 inference engine.

    Responsibilities:
      - Load and hold the model in memory (load once, reuse forever)
      - Accept normalized text chunks and synthesize each one
      - Return raw numpy waveform arrays
      - Track per-stage timing for benchmarking
    """

    # Silence buffer inserted between chunks (seconds)
    CHUNK_SILENCE_SEC = 0.05
    SAMPLE_RATE = 24000  # XTTS-v2 native sample rate

    def __init__(self, device: str = None):
        """
        Initialize the engine. Model is NOT loaded here —
        call .load() explicitly so startup timing is measurable.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.config = None
        self._load_time = None

        print(f"[TTSEngine] Initialized. Device: {self.device}")
    
    def _warmup(self):
        """
        Run a minimal forward pass to warm up CUDA kernels.
        This prevents the first real request from paying cold-start latency.
        """
        try:
            dummy_latent = torch.zeros(1, 1, 1024, device=self.device)
            dummy_embedding = torch.zeros(1, 512, 1, device=self.device)
            with torch.no_grad():
                self.model.inference(
                    text="Hello.",
                    language="en",
                    gpt_cond_latent=dummy_latent,
                    speaker_embedding=dummy_embedding,
                    temperature=0.7,
                    length_penalty=1.0,
                    repetition_penalty=10.0,
                    top_k=50,
                    top_p=0.85,
                )
        except Exception:
            # If warmup fails, we don't want to crash the whole engine — just log and continue.
            print("[TTSEngine] Warning: Warmup failed. First inference may be slow.")
            pass

    def load(self, model_dir: str = None):
        """
        Load XTTS-v2 weights into memory.

        Args:
            model_dir: Path to a local checkpoint directory.
                       If None, uses the default Coqui cache location.
        """
        t0 = time.perf_counter()

        if model_dir is None:
            # Coqui caches downloaded models here on Windows
            cache_base = Path.home() / "AppData" / "Local" / "tts"
            model_dir = cache_base / "tts_models--multilingual--multi-dataset--xtts_v2"

        model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\n"
                f"Run tests/test_model.py first to download the weights."
            )

        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_dir}")

        print(f"[TTSEngine] Loading model from: {model_dir}")

        self.config = XttsConfig()
        self.config.load_json(str(config_path))

        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config,
            checkpoint_dir=str(model_dir),
            eval=True,       # sets model to eval mode, disables dropout
            use_deepspeed=False
        )

        if self.device == "cuda":
            self.model.cuda()


        print(f"[TTSEngine] Warming up GPU...")
        self._warmup()
        self._load_time = time.perf_counter() - t0
        print(f"[TTSEngine] Model loaded in {self._load_time:.2f}s")

    def _check_loaded(self):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call engine.load() first.")

    def get_builtin_speaker_embedding(self, speaker_name: str = "Ana Florence"):
    
        self._check_loaded()

        import torch

        model_dir = Path.home() / "AppData" / "Local" / "tts" / \
                    "tts_models--multilingual--multi-dataset--xtts_v2"

        # Find the speakers file
        speakers_file = None
        for candidate in ["speakers_xtts.pth", "speakers.pth"]:
            path = model_dir / candidate
            if path.exists():
                speakers_file = path
                break

        if speakers_file is None:
            raise FileNotFoundError(
                f"No speakers file found in {model_dir}. "
                f"Expected speakers_xtts.pth or speakers.pth"
            )

        # speakers file is a dict: {speaker_name: {"gpt_cond_latent": ..., "speaker_embedding": ...}}
        speakers = torch.load(str(speakers_file), map_location="cpu")

        if speaker_name not in speakers:
            available = list(speakers.keys())
            raise ValueError(
                f"Speaker '{speaker_name}' not found.\n"
                f"Available speakers: {available}"
            )

        speaker_data = speakers[speaker_name]
        gpt_cond_latent = speaker_data["gpt_cond_latent"].to(self.device)
        speaker_embedding = speaker_data["speaker_embedding"].to(self.device)

        return gpt_cond_latent, speaker_embedding

    def get_speaker_embedding_from_audio(self, audio_path: str) -> tuple:
        """
        Extract conditioning latents from a preprocessed reference audio file.
        Phase 2 voice cloning entry point.

        Args:
            audio_path: Path to a preprocessed WAV file (output of AudioProcessor)

        Returns:
            (gpt_cond_latent, speaker_embedding) tensors on self.device
        """
        self._check_loaded()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        print(f"[TTSEngine] Extracting speaker embedding from: {audio_path.name}")

        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[str(audio_path)],
            gpt_cond_len=self.config.gpt_cond_len,
            max_ref_length=self.config.max_ref_len,
            sound_norm_refs=self.config.sound_norm_refs,
        )

        print(f"[TTSEngine] Speaker embedding extracted. "
              f"Latent shape: {gpt_cond_latent.shape} | "
              f"Embedding shape: {speaker_embedding.shape}")

        return gpt_cond_latent, speaker_embedding
    def synthesize_chunk(
        self,
        text: str,
        gpt_cond_latent,
        speaker_embedding,
        language: str = "en",
    ) -> tuple[np.ndarray, dict]:
        """
        Synthesize a single text chunk into a waveform.

        Args:
            text:               A single normalized chunk (output of chunker)
            gpt_cond_latent:    Speaker conditioning tensor
            speaker_embedding:  Speaker embedding tensor
            language:           Language code (default 'en')

        Returns:
            (waveform, timings)
            waveform: numpy array of float32 audio samples at SAMPLE_RATE
            timings:  dict with inference_sec key for benchmarking
        """
        self._check_loaded()

        if not text or not text.strip():
            raise ValueError("Empty chunk passed to synthesize_chunk.")

        t0 = time.perf_counter()

        with torch.no_grad():
            output = self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.7,        # controls variability — 0.7 is a good default
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=50,
                top_p=0.85,
            )

        inference_sec = time.perf_counter() - t0

        # output["wav"] is a list of floats — convert to numpy float32
        waveform = np.array(output["wav"], dtype=np.float32)

        timings = {
            "inference_sec": inference_sec,
            "text_length": len(text),
            "audio_duration_sec": len(waveform) / self.SAMPLE_RATE,
            "realtime_factor": (len(waveform) / self.SAMPLE_RATE) / inference_sec,
        }

        return waveform, timings

    def synthesize_chunks(
        self,
        chunks: list[str],
        gpt_cond_latent,
        speaker_embedding,
        language: str = "en",
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Synthesize a list of chunks and concatenate with silence between them.

        Returns:
            (full_waveform, all_timings)
            full_waveform: numpy array of float32 audio samples at SAMPLE_RATE
            all_timings:   list of dicts with inference_sec and other timing info for each chunk
        """
        self._check_loaded()

        if not chunks:
            raise ValueError("Empty chunk list passed to synthesize_chunks.")

        silence = np.zeros(
            int(self.CHUNK_SILENCE_SEC * self.SAMPLE_RATE),
            dtype=np.float32
        )

        waveforms = []
        all_timings = []

        for i, chunk in enumerate(chunks):
            print(f"[TTSEngine] Synthesizing chunk {i+1}/{len(chunks)}: '{chunk[:60]}...' " 
                  if len(chunk) > 60 else 
                  f"[TTSEngine] Synthesizing chunk {i+1}/{len(chunks)}: '{chunk}'")

            waveform, timings = self.synthesize_chunk(
                text=chunk,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                language=language,
            )

            waveforms.append(waveform)
            all_timings.append(timings)

            print(f"[TTSEngine]   → {timings['audio_duration_sec']:.2f}s audio "
                  f"in {timings['inference_sec']:.2f}s "
                  f"(RTF: {timings['realtime_factor']:.2f}x)")

            # Add silence between chunks (not after the last one)
            if i < len(chunks) - 1:
                waveforms.append(silence)

        full_waveform = np.concatenate(waveforms)
        return full_waveform, all_timings