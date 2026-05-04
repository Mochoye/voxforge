# Changelog

All notable changes to VoxForge are documented here.
Each phase is a self-contained milestone.

---

## [Phase 1] — Baseline Pipeline
**Goal:** Raw text → synthesized speech, single speaker, no streaming.

### Added
- `voxforge/normalizer.py` — text normalization module
  - Currency expansion (`$1.5M` → *one point five million dollars*)
  - Number expansion (`42` → *forty-two*)
  - Year expansion (`2023` → *twenty twenty-three*)
  - Abbreviation map (`Dr.` → *doctor*, `vs.` → *versus*, etc.)
  - Special character stripping with prosodic punctuation preserved
- `voxforge/chunker.py` — sentence chunking module
  - `pysbd`-based sentence boundary detection
  - Short sentence merging (< 80 chars merged with next)
  - Long sentence splitting at punctuation (> 250 chars)
- `voxforge/engine.py` — low-level XTTS-v2 inference engine
  - Direct use of `XttsConfig` / `Xtts` APIs (not the high-level wrapper)
  - GPU warm-up on model load to eliminate cold-start latency
  - Per-chunk timing with RTF tracking
  - Built-in speaker embedding loader
  - Reference audio embedding stub (used in Phase 2)
- `voxforge/pipeline.py` — end-to-end pipeline
  - Wires normalizer → chunker → engine → audio save
  - Full timing breakdown per stage
  - Returns structured result dict
- `synthesize.py` — CLI entry point with `--output`, `--speaker`, `--language` flags
- `tests/test_normalizer.py` — 7 unit tests, all passing
- `tests/test_chunker.py` — 5 unit tests, all passing
- `outputs/phase1_benchmark.json` — baseline benchmark data for Phase 3 comparison

### Benchmark (RTX 3050, no optimization)
| Input | RTF | Total Time |
|-------|-----|------------|
| Short | 1.41× | 2.14s |
| Medium | 1.13× | 9.77s |
| Long | 1.61× | 9.48s |

### Known Limitations (addressed in later phases)
- Model load time ~27s — eliminated in Phase 4 (persistent server process)
- No voice cloning yet — Phase 2
- No streaming — Phase 3
- No API — Phase 4



## [Phase 2] — Voice Cloning via Speaker Embeddings

**Goal:** Accept reference audio, extract speaker embedding, synthesize in cloned voice.

### Added
- `voxforge/audio_processor.py` — reference audio preprocessing module
  - WebRTC VAD for voice activity detection (rejects clips below 30% voiced)
  - Duration validation (3s minimum, 30s maximum)
  - DeepFilterNet3 neural denoising (GPU-accelerated, default on)
  - SHA256 file hashing for cache keying
  - Saves processed audio alongside original for inspection
- `voxforge/speaker_cache.py` — persistent embedding cache using shelve
  - Stores gpt_cond_latent + speaker_embedding tensors on CPU
  - Survives process restarts (models/speaker_cache.db)
  - Full CRUD: get, set, has, delete, clear, list, size
  - Phase 4 will replace with Redis for multi-process access
- `voxforge/engine.py` — completed get_speaker_embedding_from_audio()
- `voxforge/pipeline.py` — set_speaker_from_audio() with cache-first logic
- `synthesize.py` — added --reference, --no-denoise, --force-reprocess flags
- `tests/test_speaker_cache.py` — 7 unit tests, all passing

### Key Findings
- Temperature 0.3 produces closer voice matching than 0.7 for XTTS-v2
- Denoising on already-clean recordings hurts clone quality (use --no-denoise)
- XTTS-v2 clones pitch, timbre, and pace reliably — accent requires fine-tuning
- Sweet spot for reference audio: 8–15s of expressive, consistent speech
- Cache hit skips entire preprocessing pipeline on repeat requests

### Benchmark (RTX 3050, reference audio 27.79s, voice ratio 85%)
| Stage | Time |
|-------|------|
| VAD + validation | ~50ms |
| DeepFilterNet3 denoising | ~1.2s |
| Speaker embedding extraction | ~800ms |
| Cache hit (second request) | 0ms |
| Synthesis RTF | 1.46x |

## [Phase 3] — Inference Optimization

**Goal:** Reduce latency through FP16, ONNX export, and chunked streaming.

### Added
- `voxforge/optimizer.py` — profiling, ONNX export attempt, torch.compile,
  benchmark matrix runner with per-config JSON output
- `voxforge/streamer.py` — producer-consumer chunked streaming engine using
  Python threading and queue. Yields audio chunks as they are synthesized.
- `benchmarks/run_benchmarks.py` — 3-configuration benchmark matrix runner
- `benchmarks/results/` — JSON results for fp32_baseline, fp16, fp16_compile
- `tests/test_streamer.py` — streaming pipeline integration test

### Benchmark Results (RTX 3050, CUDA 11.8)

| Config | bench1 RTF | bench2 RTF | bench3 RTF |
|--------|-----------|-----------|-----------|
| FP32 baseline | 1.46x | 1.56x | 1.58x |
| FP16 | 0.80x | 1.36x | 1.47x |
| FP16 + compile | 1.43x | 1.45x | 1.47x |

### Key Findings
- FP16 hurts performance on RTX 3050 — 2 tensor cores insufficient for
  autoregressive TTS on small batches. Auto-disabled on GPUs under 8GB VRAM.
- torch.compile unsupported on Windows (PyTorch 2.1 requires Triton)
- ONNX export blocked by dynamic control flow in XTTS-v2 GPT backbone
- Streaming: chunk 2 (4.53s inference) ready before chunk 1 (6.36s audio)
  finishes playing — zero gap in real playback
- First chunk latency: 5.47s for 90-char input on RTX 3050
- Overall streaming RTF: 1.25x

### Optimization Conclusion
FP32 with GPU warm-up is optimal for RTX 3050 class hardware.
FP16 flag retained in codebase — will benefit A100/V100/RTX 3080+ deployments.
torch.compile will be re-evaluated in Phase 4 Linux Docker container.