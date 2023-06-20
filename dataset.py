from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms
from speechbrain.lobes.features import Fbank
import torch
import pandas as pd
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from typing import Literal


with open("./config.yaml") as hp_file:
    hparamas = load_hyperpyyaml(hp_file)

DEVICE = hparamas["DEVICE"]
TRAIN_ANNOT_PATH = hparamas["TRAIN_ANNOT_PATH"]
EVAL_ANNOT_PATH = hparamas["EVAL_ANNOT_PATH"]
DEV_ANNOT_PATH = hparamas["DEV_ANNOT_PATH"]
TRAIN_FILE_PATH = hparamas["TRAIN_FILE_PATH"]
DEV_FILE_PATH = hparamas["DEV_FILE_PATH"]
EVAL_FILE_PATH = hparamas["EVAL_FILE_PATH"]
DTYPES = {
    "names": ("spkr_id", "file_name", "spkr_feature", "attckr_feature", "label"),
    "formats": ("S", "S", "S", "S", "f"),
}


def convert_label(x: str):
    return 1 if x == "bonafide" else 0


def clip(sig: torch.Tensor):
    desired_length = 4 * 16000
    while sig.size()[1] < desired_length:
        sig = torch.cat([sig, sig], dim=1)
    return sig[:, :desired_length]


CONVERTER = {4: convert_label}


class Dataset(Dataset):
    fbank = Fbank(n_mels=80, context=False, deltas=False)
    time_mask = torchaudio.transforms.TimeMasking(80)
    frequency_mask = torchaudio.transforms.FrequencyMasking(80)

    def __init__(
        self,
        type: Literal["train", "dev", "eval"],
        balance: bool = False,
        aug=False,
        feature_type: Literal["waveform", "fbank"] = "fbank",
        clip=False,
    ):
        assert type in ["train", "dev", "eval"], "incoorect specified type"
        self.type = type
        self.aug = aug
        self.feature_type = feature_type
        self.clip = clip
        if type == "train":
            self.file_path = TRAIN_FILE_PATH
            annot = TRAIN_ANNOT_PATH
        elif type == "dev":
            self.file_path = DEV_FILE_PATH
            annot = DEV_ANNOT_PATH
        elif type == "eval":
            self.file_path = EVAL_FILE_PATH
            annot = EVAL_ANNOT_PATH
        self.dataset = pd.read_csv(
            annot, converters=CONVERTER, names=DTYPES["names"], sep=" "
        )

        if balance:
            self.dataset = self.balance(self.dataset)
        else:
            self.dataset = self.dataset.to_numpy()

    def __getitem__(self, index):
        file_name = self.dataset[index][1]
        file_path = self.file_path + file_name + ".flac"
        feature = torchaudio.load(file_path, format="flac")[0]

        if self.clip:
            feature = clip(feature)

        if self.feature_type == "fbank":
            feature = self.fbank(feature)

        if self.aug:
            feature = self.time_mask(feature)
            feature = self.frequency_mask(feature)

        # feature = torch.tensor(feature, dtype=torch.float32, device=DEVICE).squeeze(0)
        label = torch.tensor(self.dataset[index][4], dtype=torch.float32, device=DEVICE)
        sample = {"feature": feature, "label": label, "file_name": file_name}
        return sample

    def __len__(self):
        return len(self.dataset)

    def balance(self, dataset: pd.DataFrame):
        spoof_samples = dataset[dataset["label"] == 0]
        bonafide_samples = dataset[dataset["label"] == 1]
        bonafide_size = len(bonafide_samples)
        choices = np.arange(len(spoof_samples))
        choices = list(np.random.choice(choices, bonafide_size))
        spoof_samples = spoof_samples.iloc[choices]
        dataset = np.concatenate([spoof_samples, bonafide_samples], axis=0)
        return dataset
