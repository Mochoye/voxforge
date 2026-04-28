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
┌──────────────────┐     ┌──────────────────────┐
│   TTS Engine      │◄────│  Speaker Embedding    │  built-in or cloned voice
│   (XTTS-v2)      │     └──────────────────────┘
└────────┬─────────┘
         │
         ▼
    WAV Audio Output
```

**Model:** [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) — zero-shot multi-speaker TTS built on VITS. Supports 16 languages and voice cloning from a single reference clip.

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Baseline pipeline — text to speech, single speaker | ✅ Complete |
| 2 | Voice cloning via speaker embeddings | 🔜 Next |
| 3 | Inference optimization — ONNX, FP16, streaming | ⏳ Planned |
| 4 | REST API + Docker deployment | ⏳ Planned |
| 5 | Evaluation — latency, quality, benchmarks | ⏳ Planned |

---

## Phase 1 — What's Built

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

## Setup

### Requirements

- Python 3.10 or 3.11 (not 3.12)
- NVIDIA GPU with CUDA support (CPU works but is slow)
- ~2.5 GB disk space for model weights

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

# 5. Download model weights — first run only, around 1.8 GB
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', agree=True)"
```

### Running

```bash
# Basic synthesis
python synthesize.py "Hello, this is VoxForge speaking."

# Custom output path
python synthesize.py "Your text here." --output outputs/my_audio.wav

# Different built-in speaker
python synthesize.py "Your text here." --speaker "Craig Gutsy"
```

Available built-in speakers: Ana Florence, Claribel Dervla, Daisy Studious, Gracie Wise, Tammie Ema, Alison Dietlinde, Craig Gutsy, Damien Black

### Running Tests

```bash
python tests/test_normalizer.py
python tests/test_chunker.py
```

---

## Repository Structure

```
voxforge/
├── voxforge/               # Core package
│   ├── __init__.py
│   ├── normalizer.py       # Text normalization
│   ├── chunker.py          # Sentence chunking
│   ├── engine.py           # XTTS-v2 inference engine
│   └── pipeline.py         # End-to-end pipeline
├── tests/                  # Unit tests
│   ├── test_normalizer.py
│   └── test_chunker.py
├── outputs/                # Synthesized audio (git-ignored)
├── models/                 # Local model overrides (git-ignored)
├── synthesize.py           # CLI entry point
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

---

## Roadmap

- **Phase 2:** Accept reference audio, extract speaker embedding, clone voice
- **Phase 3:** ONNX export, FP16 inference, chunked streaming, latency profiling
- **Phase 4:** FastAPI REST endpoints, Docker container, streaming response
- **Phase 5:** Full benchmark report — latency, throughput, UTMOS quality scores