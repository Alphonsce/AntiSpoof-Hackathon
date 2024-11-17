import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from rawboost.data_utils_rawboost import process_Rawboost_feature

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False, train_type="2019"):
    """
    Train 2019_metadata format:
    ----
    speaker path no_need no_need label
    LA_0039 LA_E_2834763 - A11 spoof

    Eval 2021_eval format:
    speaker path idk idk idk label idk idk
    LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
    ---

    """

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    # I modified for train
    if is_train:
        for line in tqdm(l_meta, desc="Reading train data"):
            if train_type == "2019":
                _, key, _, _, label = line.strip().split(" ")
            elif train_type == "2021":
                _, key, _, _, _, label, _, _ = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _, label, _, _ = line.strip().split(" ")
            file_list.append(key)
        return file_list
    else:
        for line in tqdm(l_meta, desc="Reading eval data"):
            _, key, _, _, _, label, _, _ = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt : stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class TrainDataset(Dataset):
    def __init__(
        self,
        list_IDs,
        labels,
        base_dir,
        args=None,
        algo_rawboost=None,
        use_rawboost=False,
    ):
        """self.list_IDs	: list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)"""

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

        self.rawboost_algo = algo_rawboost
        self.rawboost_args = args
        self.use_rawboost = use_rawboost

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = sf.read(str(self.base_dir / f"{key}.flac"))
        if self.use_rawboost:
            X = process_Rawboost_feature(X, fs, self.rawboost_args, self.rawboost_algo)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class TestDataset(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),"""
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
