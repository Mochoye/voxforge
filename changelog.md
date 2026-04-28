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