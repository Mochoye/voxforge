# VoxForge 🎙️
### Real-Time Multi-Speaker Neural TTS System

VoxForge is a production-grade neural text-to-speech system built as an ML systems engineering project. It generates high-quality speech from text, supports voice cloning from short reference audio, and is designed for low-latency streaming deployment.

> **Focus:** Pipeline design, inference optimization, and production engineering — not model training.

---

## Architecture

```
Raw Text
   │
   ▼
┌──────────────────┐
│  Text Normalizer  │  numbers, currency, abbreviations, years
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Sentence Chunker │  boundary detection, merge short, split long
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌────────────────────────────────────┐
│   TTS Engine      │◄────│         Speaker Embedding           │
│   (XTTS-v2)      │     │                                    │
└────────┬─────────┘     │  Built-in speaker                  │
         │               │  OR                                │
         ▼               │  Reference Audio → VAD → Denoise   │
    WAV Audio Output     │  → XTTS Encoder → Cache            │
                         └────────────────────────────────────┘
```

**Model:** [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) — zero-shot multi-speaker TTS built on VITS. Supports 16 languages and voice cloning from a single reference clip.

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Baseline pipeline — text to speech, single speaker | ✅ Complete |
| 2 | Voice cloning via speaker embeddings | ✅ Complete |
| 3 | Inference optimization — ONNX, FP16, streaming | 🔜 Next |
| 4 | REST API + Docker deployment | ⏳ Planned |
| 5 | Evaluation — latency, quality, benchmarks | ⏳ Planned |

---

## Phase 1 — Baseline Pipeline

### Modules

| Module | Description |
|--------|-------------|
| `voxforge/normalizer.py` | Converts raw text to TTS-ready form. Expands currency (`$2M` → *two million dollars*), numbers, years (`2023` → *twenty twenty-three*), abbreviations (`Dr.` → *doctor*), strips unsupported characters |
| `voxforge/chunker.py` | Splits text into synthesis-safe chunks using `pysbd` sentence boundary detection. Merges short sentences, splits overlong ones at punctuation |
| `voxforge/engine.py` | Low-level XTTS-v2 inference engine. Loads model once, runs GPU warm-up, synthesizes per-chunk with per-stage timing |
| `voxforge/pipeline.py` | Wires all stages end-to-end. Accepts raw text, returns WAV file and full timing breakdown |
| `synthesize.py` | CLI entry point |

### Phase 1 Benchmark (RTX 3050, CUDA 11.8)

| Input | Chunks | Audio | Inference | RTF |
|-------|--------|-------|-----------|-----|
| Short sentence | 1 | 2.87s | 2.04s | **1.41x** |
| Medium (numbers + abbreviations) | 2 | 10.86s | 9.59s | **1.13x** |
| Long (2 sentences) | 2 | 14.94s | 9.35s | **1.61x** |

> RTF = Real-Time Factor. RTF above 1.0 means faster than real-time.
> All tests run post GPU warm-up. Model load time (~27s) is a one-time startup cost,
> eliminated in Phase 4 when the model stays loaded in a server process.

### Stage Timing Breakdown

| Stage | Time |
|-------|------|
| Text normalization | less than 2ms |
| Sentence chunking | less than 14ms |
| TTS inference | 95%+ of total |
| Audio save | less than 170ms |

---

## Phase 2 — Voice Cloning

### How It Works

Voice cloning works by extracting a speaker embedding from a short reference audio clip and conditioning the TTS model on it. No model fine-tuning required — XTTS-v2 supports zero-shot cloning natively.

```
reference_audio/my_voice.wav
        │
        ▼
┌─────────────────────┐
│   AudioProcessor     │  VAD → duration check → optional denoising
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   SpeakerCache       │  SHA256 hash lookup → shelve cache
└────────┬────────────┘  (skips re-extraction on repeat requests)
         │
         ▼
┌─────────────────────┐
│   XTTS-v2 Encoder    │  extracts gpt_cond_latent + speaker_embedding
└────────┬────────────┘
         │
         ▼
    Synthesis in cloned voice
```

### New Modules

| Module | Description |
|--------|-------------|
| `voxforge/audio_processor.py` | Validates and preprocesses reference audio. Runs WebRTC VAD (rejects clips below 30% voiced), checks duration (3–30s), optionally applies DeepFilterNet3 neural denoising |
| `voxforge/speaker_cache.py` | Persistent embedding cache using Python shelve. SHA256 file hash as key. Survives restarts. Cache hit skips the entire preprocessing and extraction pipeline |

### Phase 2 Benchmark (RTX 3050, reference audio 27.79s, voice ratio 85%)

| Stage | Time |
|-------|------|
| VAD + validation | ~50ms |
| DeepFilterNet3 denoising | ~1.2s |
| Speaker embedding extraction | ~800ms |
| Synthesis RTF | 1.46x |
| Cache hit (repeat request) | 0ms preprocessing |

### Voice Cloning — Known Limitations

XTTS-v2 is a zero-shot voice cloner — it has never seen your voice during training. It reliably clones pitch, timbre, and speaking pace. Accent and regional dialect require fine-tuning, which is outside the scope of this project.

Best results with reference audio that is 8–15 seconds, expressive (not flat monotone), and recorded in a quiet environment. For already-clean recordings use `--no-denoise` to skip denoising.

---

## Setup

### Requirements

- Python 3.10 or 3.11 (not 3.12)
- NVIDIA GPU with CUDA support (CPU works but is slow)
- ~2.5 GB disk space for XTTS-v2 model weights
- ~500 MB for DeepFilterNet3 weights (downloaded on first use)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/voxforge.git
cd voxforge

# 2. Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

# 3. Install PyTorch with CUDA (adjust cu118 to match your CUDA version)
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Download XTTS-v2 weights — first run only, around 1.8 GB
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', agree=True)"
```

### Running

```bash
# Built-in speaker (Phase 1 mode)
python synthesize.py "Hello, this is VoxForge speaking."

# Custom output path
python synthesize.py "Your text here." --output outputs/my_audio.wav

# Different built-in speaker
python synthesize.py "Your text here." --speaker "Craig Gutsy"

# Voice cloning from reference audio (Phase 2 mode)
python synthesize.py "Your text here." --reference reference_audio/my_voice.wav

# Voice cloning without denoising (recommended for clean recordings)
python synthesize.py "Your text here." --reference reference_audio/my_voice.wav --no-denoise

# Force re-extract embedding even if cached
python synthesize.py "Your text here." --reference reference_audio/my_voice.wav --force-reprocess
```

Available built-in speakers: Ana Florence, Claribel Dervla, Daisy Studious, Gracie Wise, Tammie Ema, Alison Dietlinde, Craig Gutsy, Damien Black

### Reference Audio Guidelines

For best voice cloning results:

- Duration: 8–15 seconds (sweet spot), minimum 3s, maximum 30s
- Content: natural expressive speech, not flat monotone
- Environment: quiet room, consistent volume, no background music
- Format: WAV or MP3, any sample rate (resampled automatically)
- Use `--no-denoise` if recording in a clean environment

### Running Tests

```bash
python tests/test_normalizer.py
python tests/test_chunker.py
python tests/test_speaker_cache.py
```

---

## Repository Structure

```
voxforge/
├── voxforge/                  # Core package
│   ├── __init__.py
│   ├── normalizer.py          # Text normalization
│   ├── chunker.py             # Sentence chunking
│   ├── engine.py              # XTTS-v2 inference engine
│   ├── pipeline.py            # End-to-end pipeline
│   ├── audio_processor.py     # Reference audio preprocessing (Phase 2)
│   └── speaker_cache.py       # Persistent embedding cache (Phase 2)
├── tests/                     # Unit tests
│   ├── test_normalizer.py
│   ├── test_chunker.py
│   └── test_speaker_cache.py
├── reference_audio/           # Reference clips for voice cloning (git-ignored)
├── outputs/                   # Synthesized audio (git-ignored)
├── models/                    # Speaker cache + model overrides (git-ignored)
├── synthesize.py              # CLI entry point
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---

## Stack

| Component | Library |
|-----------|---------|
| TTS Model | [coqui-tts](https://github.com/idiap/coqui-ai-TTS) — XTTS-v2 |
| Deep Learning | PyTorch 2.1 + CUDA 11.8 |
| Text Normalization | num2words, inflect |
| Sentence Splitting | pysbd |
| Audio I/O | soundfile, torchaudio |
| Voice Activity Detection | webrtcvad-wheels |
| Neural Denoising | deepfilternet (DeepFilterNet3) |
| Embedding Cache | Python shelve (Phase 4 → Redis) |

---

## Roadmap

- **Phase 3:** ONNX export, FP16 inference, chunked streaming, latency profiling
- **Phase 4:** FastAPI REST endpoints, Docker container, streaming response, Redis cache
- **Phase 5:** Full benchmark report — latency, throughput, UTMOS quality scores, speaker similarity analysis