import torch
import torchaudio
import soundfile
import pysbd
from TTS.api import TTS

print('All imports OK')
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
