import numpy as np
from pydub import AudioSegment
import tempfile
import os
import random
import torch
import torchaudio

def apply_torch_codecs(waveform, sample_rate, num_augmentations=3):
    waveform = waveform.unsqueeze(0)
    
    augmentations = [
        lambda w, sr: torchaudio.functional.apply_codec(w, sr, format="mp3"),
        lambda w, sr: torchaudio.functional.apply_codec(w, sr, format="ogg"),
        lambda w, sr: torchaudio.functional.apply_codec(w, sr, format="vorbis", compression=-1),
        lambda w, sr: torchaudio.functional.apply_codec(w, sr, format="wav", encoding="ULAW", bits_per_sample=8),
    ]
    
    
    for i in range(num_augmentations):
        augmentation = augmentations[torch.randint(len(augmentations), (1,)).item()]
        waveform = augmentation(waveform, sample_rate)
        
    return waveform.flatten().numpy(), sample_rate

