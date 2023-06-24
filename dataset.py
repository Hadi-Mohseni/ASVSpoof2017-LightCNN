from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Spectrogram, AmplitudeToDB
import torch
import pandas as pd
from typing import Literal
import os


def convert_label(x: str):
    return [1, 0] if x == "genuine" else [0, 1]


def clip(sig: torch.Tensor):
    desired_length = 864
    while sig.size()[2] < desired_length:
        sig = torch.cat([sig, sig], dim=2)
    return sig[:, :, :desired_length]


DEVICE = "cuda"
TRAIN_ANNOT_PATH = "/content/protocol_V2/protocol_V2/ASVspoof2017_V2_train.trn.txt"
DEV_ANNOT_PATH = "/content/protocol_V2/protocol_V2/ASVspoof2017_V2_dev.trl.txt"
EVAL_ANNOT_PATH = "/content/protocol_V2/protocol_V2/ASVspoof2017_V2_eval.trl.txt"
TRAIN_FOLDER_PATH = "/content/ASVspoof2017_V2_train/ASVspoof2017_V2_train/"
DEV_FOLDER_PATH = "/content/ASVspoof2017_V2_dev/ASVspoof2017_V2_dev/"
EVAL_FOLDER_PATH = "/content/ASVspoof2017_V2_eval/ASVspoof2017_V2_eval/"
CONVERTER = {1: convert_label}
DTYPES = {
    "names": ("file_name", "label", "spkr_id", "x", "y", "z", "w"),
    "formats": ("S", "f", "S", "S", "S", "S", "S"),
}


class Dataset(Dataset):
    spec = Spectrogram(n_fft=798, win_length=25, hop_length=10, normalized=True)
    ptod_converter = AmplitudeToDB()

    def __init__(self, type: Literal["train", "dev", "eval"]):
        if type == "train":
            self.file_path = TRAIN_FOLDER_PATH
            annot = TRAIN_ANNOT_PATH
        elif type == "dev":
            self.file_path = DEV_FOLDER_PATH
            annot = DEV_ANNOT_PATH
        elif type == "eval":
            self.file_path = EVAL_FOLDER_PATH
            annot = EVAL_ANNOT_PATH

        self.dataset = pd.read_csv(
            annot, converters=CONVERTER, names=DTYPES["names"], sep=" "
        )
        self.dataset = self.dataset.to_numpy()

    def __getitem__(self, index):
        file_path = os.path.join(self.file_path, self.dataset[index][0])
        feature, sr = torchaudio.load(file_path)
        feature = torchaudio.functional.vad(feature, sample_rate=sr)
        feature = self.ptod_converter(self.spec(feature))
        feature = clip(feature)
        label = torch.tensor(self.dataset[index][1], dtype=torch.float32, device=DEVICE)
        feature = feature.to(device=DEVICE)
        sample = {"feature": feature, "label": label}
        return sample

    def __len__(self):
        return len(self.dataset)
