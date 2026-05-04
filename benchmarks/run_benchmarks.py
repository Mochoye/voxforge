"""
VoxForge Phase 3 — Benchmark Matrix Runner

Runs all three optimization configurations against the same three
test cases used in Phase 1, so results are directly comparable.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --reference reference_audio/my_voice.wav
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voxforge.optimizer import run_benchmark_matrix, print_comparison_table

# Same three test cases as Phase 1 baseline
TEST_CASES = [
    {
        "label": "bench1_short",
        "text": "The quick brown fox jumps over the lazy dog.",
        "output_path": "outputs/bench_p3_short.wav",
    },
    {
        "label": "bench2_medium",
        "text": (
            "Dr. Smith raised $2M for his AI research lab in 2023. "
            "The team has 42 engineers working on the project. "
            "They expect results by next year."
        ),
        "output_path": "outputs/bench_p3_medium.wav",
    },
    {
        "label": "bench3_long",
        "text": (
            "VoxForge is a production-grade neural text to speech system. "
            "It supports voice cloning, low latency streaming, and scalable "
            "API deployment. The system uses pretrained models and is "
            "optimized for real-world use."
        ),
        "output_path": "outputs/bench_p3_long.wav",
    },
]


def main():
    parser = argparse.ArgumentParser(description="VoxForge Phase 3 Benchmark")
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Reference audio for voice cloning mode"
    )
    parser.add_argument(
        "--configs", type=str, default="all",
        help="Comma-separated configs to run: fp32_baseline,fp16,fp16_compile"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VoxForge Phase 3 — Benchmark Matrix")
    print("=" * 60)
    print(f"Test cases  : {len(TEST_CASES)}")
    print(f"Voice mode  : {'cloning' if args.reference else 'built-in (Ana Florence)'}")
    print(f"Output dir  : benchmarks/results/")
    print("=" * 60)

    all_results = run_benchmark_matrix(
        test_cases=TEST_CASES,
        output_dir="benchmarks/results",
        reference_audio=args.reference,
    )

    print_comparison_table(all_results)

    # Load Phase 1 baseline for comparison
    p1_path = Path("outputs/phase1_benchmarks.json")
    if p1_path.exists():
        with open(p1_path) as f:
            p1 = json.load(f)

        print(f"\n{'='*60}")
        print("PHASE 1 BASELINE (no optimization)")
        print(f"{'='*60}")
        for r in p1["results"]:
            print(f"  {r['test']:<25} RTF: {r['rtf']:.2f}x  "
                  f"Inference: {r['inference_sec']:.2f}s")


if __name__ == "__main__":
    main()