from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Spectrogram
import torch
import pandas as pd
from typing import Literal
import os


def convert_label(x: str):
    return 1 if x == "bonafide" else 0


def clip(sig: torch.Tensor):
    desired_length = 4 * 16000
    while sig.size()[1] < desired_length:
        sig = torch.cat([sig, sig], dim=1)
    return sig[:, :desired_length]


DEVICE = "CUDA"
TRAIN_ANNOT_PATH = "/content/protocol_V2/protocol_V2/ASVspoof2017_V2_train.trn.txt"
DEV_ANNOT_PATH = "/content/protocol_V2/protocol_V2/ASVspoof2017_V2_dev.trl.txt"
EVAL_ANNOT_PATH = "/content/protocol_V2/protocol_V2/ASVspoof2017_V2_eval.trl.txt"
TRAIN_FOLDER_PATH = "/content/ASVspoof2017_V2_train/ASVspoof2017_V2_train"
DEV_FOLDER_PATH = "/content/ASVspoof2017_V2_dev/ASVspoof2017_V2_dev"
EVAL_FOLDER_PATH = "/content/ASVspoof2017_V2_eval/ASVspoof2017_V2_eval"
CONVERTER = {1: convert_label}
DTYPES = {
    "names": ("file_name", "label", "spkr_id"),
    "formats": ("S", "f", "S"),
}


class Dataset(Dataset):
    spec = Spectrogram(win_length=25, hop_length=10)

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
        self.dataset["file_name"] = os.path.join(
            self.file_path, self.dataset["file_name"]
        )
        self.dataset = self.dataset.to_numpy()

    def __getitem__(self, index):
        feature = torchaudio.load(self.dataset[index]["file_name"])
        feature = clip(feature)
        feature = self.spec(feature)
        label = torch.tensor(self.dataset[index][4], dtype=torch.float32, device=DEVICE)
        sample = {"feature": feature, "label": label}
        return sample

    def __len__(self):
        return len(self.dataset)
