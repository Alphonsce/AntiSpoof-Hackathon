import torch
import torchaudio

path = "/dataset/2019_LA/ASVspoof2019_LA_train/flac/LA_T_1178793.flac"

torchaudio.load(path)