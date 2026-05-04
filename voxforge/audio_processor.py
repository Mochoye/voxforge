import os
import hashlib
import struct
import wave
import numpy as np
import torchaudio
import torch
import webrtcvad
from pathlib import Path


_df_model = None
_df_state = None


def _load_denoiser():
    """Load DeepFilterNet once, reuse forever."""
    global _df_model, _df_state
    if _df_model is None:
        from df.enhance import init_df, enhance
        _df_model, _df_state, _ = init_df()
    return _df_model, _df_state



VAD_SAMPLE_RATE = 16000       # webrtcvad only works at 8k, 16k, 32k, 48k
VAD_FRAME_MS = 30             # frame duration in ms (10, 20, or 30 only)
VAD_AGGRESSIVENESS = 2        # 0=least aggressive, 3=most aggressive
MIN_VOICE_RATIO = 0.3         # at least 30% of audio must be voiced
MIN_DURATION_SEC = 3.0        # reject clips shorter than this
MAX_DURATION_SEC = 30.0       # reject clips longer than this


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Load any audio file to a mono float32 numpy array.
    Returns (waveform, sample_rate).
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sr = torchaudio.load(str(audio_path))

    # Mix down to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze().numpy().astype(np.float32), sr


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample waveform from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return waveform

    tensor = torch.from_numpy(waveform).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
    return resampled.squeeze().numpy()


def run_vad(waveform: np.ndarray, sample_rate: int) -> tuple[float, list[bool]]:
    """
    Run WebRTC VAD on waveform.

    Returns:
        voice_ratio : fraction of frames that contain speech (0.0 – 1.0)
        frame_flags : list of booleans, one per frame (True = voiced)
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    # VAD requires 16kHz, 16-bit PCM
    if sample_rate != VAD_SAMPLE_RATE:
        waveform = resample(waveform, sample_rate, VAD_SAMPLE_RATE)

    # Convert float32 → int16 PCM bytes
    pcm = (waveform * 32768).astype(np.int16)
    raw_bytes = pcm.tobytes()

    frame_len = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)  # samples per frame
    frame_bytes = frame_len * 2                               # 2 bytes per int16

    flags = []
    offset = 0
    while offset + frame_bytes <= len(raw_bytes):
        frame = raw_bytes[offset: offset + frame_bytes]
        try:
            is_speech = vad.is_speech(frame, VAD_SAMPLE_RATE)
        except Exception:
            is_speech = False
        flags.append(is_speech)
        offset += frame_bytes

    if not flags:
        return 0.0, []

    voice_ratio = sum(flags) / len(flags)
    return voice_ratio, flags


def denoise(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Apply DeepFilterNet noise suppression.
    Input and output are float32 numpy arrays.
    """
    from df.enhance import enhance

    model, state = _load_denoiser()

    # DeepFilterNet expects float32 tensor with shape (1, samples) at its native SR
    tensor = torch.from_numpy(waveform).unsqueeze(0)

    # Resample to DeepFilterNet's expected sample rate if needed
    df_sr = state.sr()
    if sample_rate != df_sr:
        tensor = torchaudio.functional.resample(tensor, sample_rate, df_sr)

    enhanced = enhance(model, state, tensor)
    result = enhanced.squeeze().numpy()

    # Resample back to original rate
    if sample_rate != df_sr:
        result_tensor = torch.from_numpy(result).unsqueeze(0)
        result_tensor = torchaudio.functional.resample(result_tensor, df_sr, sample_rate)
        result = result_tensor.squeeze().numpy()

    return result.astype(np.float32)


def validate_audio(waveform: np.ndarray, sample_rate: int) -> dict:
    """
    Validate reference audio quality.

    Returns a dict with:
        valid       : bool — whether audio passes all gates
        duration    : float — duration in seconds
        voice_ratio : float — fraction of frames with speech
        issues      : list[str] — human-readable list of problems found
    """
    issues = []
    duration = len(waveform) / sample_rate

    # Gate 1: Duration
    if duration < MIN_DURATION_SEC:
        issues.append(
            f"Too short: {duration:.1f}s (minimum {MIN_DURATION_SEC}s)"
        )
    if duration > MAX_DURATION_SEC:
        issues.append(
            f"Too long: {duration:.1f}s (maximum {MAX_DURATION_SEC}s). "
            f"Trim to 10–20s for best results."
        )

    # Gate 2: Voice activity
    voice_ratio, _ = run_vad(waveform, sample_rate)
    if voice_ratio < MIN_VOICE_RATIO:
        issues.append(
            f"Too little speech detected: {voice_ratio:.0%} voiced "
            f"(minimum {MIN_VOICE_RATIO:.0%}). "
            f"Check for long silences or background noise."
        )

    return {
        "valid": len(issues) == 0,
        "duration": round(duration, 2),
        "voice_ratio": round(voice_ratio, 3),
        "issues": issues,
    }


def file_hash(audio_path: str) -> str:
    """
    Compute SHA256 hash of an audio file.
    Used as the cache key — same file always maps to same hash.
    """
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def process_reference_audio(
    audio_path: str,
    apply_denoising: bool = True,
) -> tuple[str, dict]:
    """
    Full preprocessing pipeline for a reference audio file.

    Steps:
        1. Load audio
        2. Validate (duration + VAD)
        3. Optionally denoise
        4. Save processed version to a temp path
        5. Return (processed_path, validation_report)

    The processed path is what gets passed to the speaker encoder.
    """
    audio_path = Path(audio_path)
    print(f"[AudioProcessor] Processing: {audio_path.name}")

    # Step 1: Load
    waveform, sr = load_audio(str(audio_path))
    print(f"[AudioProcessor]   Duration: {len(waveform)/sr:.2f}s | Sample rate: {sr}Hz")

    # Step 2: Validate
    report = validate_audio(waveform, sr)
    print(f"[AudioProcessor]   Voice ratio: {report['voice_ratio']:.0%}")

    if not report["valid"]:
        issues_str = " | ".join(report["issues"])
        raise ValueError(
            f"Reference audio failed validation: {issues_str}"
        )

    # Step 3: Denoise
    if apply_denoising:
        print(f"[AudioProcessor]   Denoising...")
        waveform = denoise(waveform, sr)
        print(f"[AudioProcessor]   Denoising complete.")

    # Step 4: Save processed file next to original
    processed_path = audio_path.parent / f"{audio_path.stem}_processed.wav"
    save_tensor = torch.from_numpy(waveform).unsqueeze(0)
    torchaudio.save(str(processed_path), save_tensor, sr)
    print(f"[AudioProcessor]   Saved processed audio: {processed_path.name}")

    report["processed_path"] = str(processed_path)
    return str(processed_path), report