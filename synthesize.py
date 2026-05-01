"""
VoxForge Phase 2 — CLI entry point

Usage:
    # Built-in speaker (Phase 1 mode)
    python synthesize.py "Your text here."

    # Voice cloning (Phase 2 mode)
    python synthesize.py "Your text here." --reference reference_audio/my_voice.wav

    # Force re-extract even if cached
    python synthesize.py "Text." --reference reference_audio/my_voice.wav --force-reprocess
"""

import argparse
import json
from voxforge.pipeline import VoxForgePipeline


def main():
    parser = argparse.ArgumentParser(description="VoxForge TTS — Phase 2")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument(
        "--output", type=str,
        default="outputs/output.wav",
        help="Output WAV file path (default: outputs/output.wav)"
    )
    parser.add_argument(
        "--speaker", type=str,
        default="Ana Florence",
        help="Built-in speaker name (ignored if --reference is set)"
    )
    parser.add_argument(
        "--reference", type=str,
        default=None,
        help="Path to reference audio for voice cloning (WAV or MP3)"
    )
    parser.add_argument(
        "--language", type=str,
        default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--no-denoise", action="store_true",
        help="Skip DeepFilterNet denoising on reference audio"
    )
    parser.add_argument(
        "--force-reprocess", action="store_true",
        help="Ignore cache and re-extract speaker embedding"
    )

    args = parser.parse_args()

    pipeline = VoxForgePipeline()

    if args.reference:
        # Phase 2 mode: voice cloning
        print(f"[VoxForge] Mode: Voice Cloning")
        print(f"[VoxForge] Reference: {args.reference}")
        pipeline.engine.load()
        report = pipeline.set_speaker_from_audio(
            audio_path=args.reference,
            apply_denoising=not args.no_denoise,
            force_reprocess=args.force_reprocess,
        )
        print(f"\n[VoxForge] Speaker report:")
        print(json.dumps({k: v for k, v in report.items() 
                         if k != "processed_path"}, indent=2))
    else:
        # Phase 1 mode: built-in speaker
        print(f"[VoxForge] Mode: Built-in Speaker ({args.speaker})")
        pipeline.load(speaker=args.speaker)

    result = pipeline.synthesize(
        text=args.text,
        output_path=args.output,
        language=args.language,
        verbose=True,
    )

    print("\n--- Result Summary ---")
    print(json.dumps(result["timings"], indent=2))
    print(f"Speaker          : {result['speaker']}")
    print(f"Chunks processed : {len(result['chunks'])}")
    print(f"Audio duration   : {result['total_audio_duration_sec']}s")
    print(f"Output file      : {result['output_path']}")


if __name__ == "__main__":
    main()