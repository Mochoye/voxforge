"""
VoxForge Phase 3 — Optimization Module

Handles:
  - Per-stage profiling with torch.profiler
  - ONNX export of exportable subgraphs
  - Benchmark matrix runner
"""

import time
import json
import torch
import numpy as np
from pathlib import Path


# ── Profiling ─────────────────────────────────────────────────────────────────

def profile_stage(fn, label: str, warmup_runs: int = 1, timed_runs: int = 5) -> dict:
    """
    Profile a callable across multiple runs.

    Args:
        fn          : zero-argument callable wrapping the stage to profile
        label       : human-readable name for the stage
        warmup_runs : runs before timing starts (warms up CUDA kernels)
        timed_runs  : runs to average over

    Returns:
        dict with label, mean_ms, min_ms, max_ms, std_ms
    """
    # Warmup
    for _ in range(warmup_runs):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    result = {
        "label": label,
        "mean_ms": round(float(np.mean(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
        "max_ms": round(float(np.max(times)), 2),
        "std_ms": round(float(np.std(times)), 2),
        "runs": timed_runs,
    }

    print(f"[Profiler] {label}: "
          f"mean={result['mean_ms']}ms "
          f"min={result['min_ms']}ms "
          f"max={result['max_ms']}ms "
          f"std={result['std_ms']}ms")

    return result


def profile_pipeline(pipeline, test_cases: list[dict]) -> list[dict]:
    """
    Profile the full pipeline across a list of test cases.

    Args:
        pipeline   : loaded VoxForgePipeline instance
        test_cases : list of dicts with 'text', 'label', 'output_path'

    Returns:
        list of result dicts with timing breakdown per test case
    """
    results = []

    for case in test_cases:
        print(f"\n[Profiler] === {case['label']} ===")
        result = pipeline.synthesize(
            text=case["text"],
            output_path=case["output_path"],
            verbose=False,
        )

        audio_dur = result["total_audio_duration_sec"]
        inf_sec = result["timings"]["inference_sec"]
        rtf = round(audio_dur / inf_sec, 3) if inf_sec > 0 else 0

        entry = {
            "label": case["label"],
            "text_length": len(case["text"]),
            "chunks": len(result["chunks"]),
            "audio_duration_sec": audio_dur,
            "inference_sec": inf_sec,
            "rtf": rtf,
            "total_sec": result["timings"]["total_sec"],
            "normalization_ms": result["timings"]["normalization_ms"],
            "chunking_ms": result["timings"]["chunking_ms"],
            "save_ms": result["timings"]["save_ms"],
        }

        print(f"[Profiler] Audio: {audio_dur:.2f}s | "
              f"Inference: {inf_sec:.2f}s | "
              f"RTF: {rtf:.2f}x")

        results.append(entry)

    return results


# ── ONNX Export ───────────────────────────────────────────────────────────────

def export_speaker_encoder_onnx(engine, output_path: str = "models/speaker_encoder.onnx") -> str:
    """
    Export the XTTS-v2 speaker encoder (conditioning latent extractor)
    to ONNX format.

    Note: We export what we can. The full GPT backbone of XTTS-v2 has
    dynamic control flow that prevents clean ONNX export. The speaker
    encoder is a standard transformer encoder and exports cleanly.

    Args:
        engine      : loaded TTSEngine instance
        output_path : where to save the .onnx file

    Returns:
        path to saved ONNX file
    """
    engine._check_loaded()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ONNX] Attempting speaker encoder export...")

    try:
        # The speaker encoder processes mel spectrograms → embeddings
        # Dummy input: batch=1, mel_channels=80, time_steps=100
        dummy_mel = torch.randn(1, 80, 100, device=engine.device)

        # Try to access the speaker encoder submodule
        if hasattr(engine.model, 'hifigan_decoder'):
            print(f"[ONNX] Found HiFi-GAN decoder submodule.")

        if hasattr(engine.model, 'gpt'):
            print(f"[ONNX] GPT backbone detected — skipping (dynamic graph).")

        # Export the mel-to-embedding projection if accessible
        encoder = None
        for name, module in engine.model.named_modules():
            if 'speaker_encoder' in name.lower() and hasattr(module, 'forward'):
                encoder = module
                print(f"[ONNX] Found speaker encoder at: {name}")
                break

        if encoder is None:
            print(f"[ONNX] Speaker encoder not directly accessible as submodule.")
            print(f"[ONNX] XTTS-v2 integrates the encoder into the conditioning pipeline.")
            print(f"[ONNX] Skipping speaker encoder ONNX export.")
            return None

        torch.onnx.export(
            encoder,
            dummy_mel,
            str(output_path),
            opset_version=17,
            input_names=["mel_spectrogram"],
            output_names=["speaker_embedding"],
            dynamic_axes={
                "mel_spectrogram": {2: "time_steps"},
            },
            do_constant_folding=True,
        )

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[ONNX] Exported to: {output_path} ({size_mb:.1f} MB)")
        return str(output_path)

    except Exception as e:
        print(f"[ONNX] Export failed: {e}")
        print(f"[ONNX] This is expected for tightly integrated model architectures.")
        print(f"[ONNX] Optimization will rely on FP16 + torch.compile instead.")
        return None


def try_torch_compile(engine) -> bool:
    """
    Apply torch.compile to the model as an alternative to ONNX.
    torch.compile (Dynamo) handles dynamic graphs that ONNX cannot export.

    Returns True if compilation succeeded.
    """
    engine._check_loaded()

    print(f"[Compile] Attempting torch.compile on XTTS-v2...")

    try:
        # Use 'reduce-overhead' mode — best for repeated same-shape inputs
        # 'max-autotune' would be better but takes much longer to compile
        engine.model = torch.compile(
            engine.model,
            mode="reduce-overhead",
            fullgraph=False,   # allow graph breaks for dynamic control flow
        )
        print(f"[Compile] torch.compile applied successfully.")
        print(f"[Compile] Note: First inference after compile will be slow (tracing).")
        print(f"[Compile] Subsequent inferences will use the compiled graph.")
        return True

    except Exception as e:
        print(f"[Compile] torch.compile failed: {e}")
        return False


# ── Benchmark Matrix ───────────────────────────────────────────────────────────

def run_benchmark_matrix(
    test_cases: list[dict],
    output_dir: str = "benchmarks/results",
    reference_audio: str = None,
) -> dict:
    """
    Run the full 3-configuration benchmark matrix and save results.

    Configurations:
      1. Baseline  — FP32, no compile   (Phase 1 numbers for comparison)
      2. FP16      — autocast float16
      3. FP16 + compile — autocast + torch.compile

    Args:
        test_cases      : list of {text, label, output_path}
        output_dir      : where to save result JSON files
        reference_audio : if set, use voice cloning mode

    Returns:
        dict mapping config_name → list of result dicts
    """
    from voxforge.pipeline import VoxForgePipeline

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    configs = [
        {"name": "fp32_baseline", "use_fp16": False, "use_compile": False},
        {"name": "fp16",          "use_fp16": True,  "use_compile": False},
        {"name": "fp16_compile",  "use_fp16": True,  "use_compile": True},
    ]

    for config in configs:
        print(f"\n{'='*60}")
        print(f"[Benchmark] Configuration: {config['name']}")
        print(f"{'='*60}")

        # Fresh pipeline for each config
        pipeline = VoxForgePipeline(use_fp16=config["use_fp16"])
        pipeline.engine.load()

        if reference_audio:
            pipeline.set_speaker_from_audio(reference_audio, apply_denoising=True)
        else:
            pipeline.set_speaker("Ana Florence")

        if config["use_compile"]:
            success = try_torch_compile(pipeline.engine)
            if success:
                # Warm up the compiled model
                print(f"[Benchmark] Warming up compiled model...")
                pipeline.synthesize(
                    text="Warming up the compiled model.",
                    output_path=f"outputs/warmup_{config['name']}.wav",
                    verbose=False,
                )

        # Run test cases
        results = profile_pipeline(pipeline, test_cases)
        all_results[config["name"]] = results

        # Save per-config results
        result_path = output_dir / f"{config['name']}.json"
        with open(result_path, "w") as f:
            json.dump({
                "config": config["name"],
                "use_fp16": config["use_fp16"],
                "use_compile": config["use_compile"],
                "results": results,
            }, f, indent=2)

        print(f"[Benchmark] Results saved to: {result_path}")

        # Explicitly free GPU memory between configs
        del pipeline
        torch.cuda.empty_cache()

    # Save combined results
    combined_path = output_dir / "combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Benchmark] All results saved to: {combined_path}")
    return all_results


def print_comparison_table(all_results: dict, baseline_key: str = "fp32_baseline"):
    """
    Print a markdown-formatted comparison table from benchmark results.
    """
    baseline = {r["label"]: r for r in all_results.get(baseline_key, [])}

    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS — RTF COMPARISON")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'Test':<25} {'RTF':>6} {'Inf(s)':>8} {'vs Baseline':>12}")
    print(f"{'-'*20} {'-'*25} {'-'*6} {'-'*8} {'-'*12}")

    for config_name, results in all_results.items():
        for r in results:
            label = r["label"]
            rtf = r["rtf"]
            inf = r["inference_sec"]

            # Speedup vs baseline
            baseline_inf = baseline.get(label, {}).get("inference_sec", inf)
            speedup = round(baseline_inf / inf, 2) if inf > 0 else 1.0
            speedup_str = f"{speedup:.2f}x" if config_name != baseline_key else "baseline"

            print(f"{config_name:<20} {label:<25} {rtf:>6.2f} {inf:>8.2f} {speedup_str:>12}")

    print(f"{'='*80}")