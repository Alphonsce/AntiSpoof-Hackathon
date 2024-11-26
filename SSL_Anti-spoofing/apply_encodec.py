from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor

from IPython.display import Audio
import os

import torchaudio

from torchaudio.transforms import Resample
from torchaudio.functional import resample
import torch

from tqdm import tqdm
import numpy as np

read_path = "/data/a.varlamov/asvspoof/ASVspoof2021_DF_eval/flac"

output_path = "/data/a.varlamov/asvspoof/DF_flacs_encodec"

model = EncodecModel.from_pretrained("facebook/encodec_24khz").to("cuda")

def augment_with_encodec(waveform, sample_rate):
    '''
    
    '''
    waveform = resample(waveform, sample_rate, 24000).flatten()
    
    audio_values = model(waveform.to("cuda").unsqueeze(0).unsqueeze(1)).audio_values.flatten().cpu().detach()

    aug_waveform = resample(audio_values, 24000, sample_rate)
    
    return aug_waveform.unsqueeze(0)


for i, path in tqdm(enumerate(os.listdir(read_path))):
    if i > 300_000:
        continue
    if np.random.uniform(0, 1) > 0.3:
        wave, sr = torchaudio.load(f"{read_path}/{path}")
        wave_aug = augment_with_encodec(wave, sr)
        # print(path)
        torchaudio.save(f"{read_path}/{path}", wave_aug, sample_rate=sr)