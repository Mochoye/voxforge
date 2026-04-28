"""
VoxForge Phase 1 — CLI entry point

Usage:
    python synthesize.py "Your text here"
    python synthesize.py "Your text here" --output outputs/custom.wav
    python synthesize.py "Your text here" --speaker "Craig Gutsy"
"""

import argparse
import json
from voxforge.pipeline import VoxForgePipeline


def main():
    parser = argparse.ArgumentParser(description="VoxForge TTS — Phase 1")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument(
        "--output", type=str,
        default="outputs/output.wav",
        help="Output WAV file path (default: outputs/output.wav)"
    )
    parser.add_argument(
        "--speaker", type=str,
        default="Ana Florence",
        help="Built-in speaker name (default: Ana Florence)"
    )
    parser.add_argument(
        "--language", type=str,
        default="en",
        help="Language code (default: en)"
    )
    args = parser.parse_args()

    pipeline = VoxForgePipeline()
    pipeline.load(speaker=args.speaker)

    result = pipeline.synthesize(
        text=args.text,
        output_path=args.output,
        language=args.language,
        verbose=True,
    )

    print("\n--- Result Summary ---")
    print(json.dumps(result["timings"], indent=2))
    print(f"Chunks processed : {len(result['chunks'])}")
    print(f"Audio duration   : {result['total_audio_duration_sec']}s")
    print(f"Output file      : {result['output_path']}")


if __name__ == "__main__":
    main()