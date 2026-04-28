import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# This will download XTTS-v2 weights (~1.8 GB) on first run
# Subsequent runs load from cache (~/.local/share/tts/)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts.tts_to_file(
    text="Hello. VoxForge is online. The pipeline is working correctly.",
    speaker="Ana Florence",       # built-in voice, no reference audio needed
    language="en",
    file_path="outputs/test_output.wav"
)

print("Done. Check outputs/test_output.wav")