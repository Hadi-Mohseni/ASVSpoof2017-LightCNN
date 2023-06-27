from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Spectrogram, AmplitudeToDB
import torch
import pandas as pd
import os
from typing import Literal, Dict
import numpy as np


def convert_label(label: Literal["spoof", "genuine"]) -> torch.Tensor:
    """
    convert_label

    convert string labels to numbers to make them compatible
    with loss function usage.

    Parameters
    ----------
    label : Literal[&quot;spoof&quot;, &quot;genuine&quot;]
        sample label that can be "spoof" or "genuine"

    Returns
    -------
    torch.Tensor
        [1, 0] if "genuine" else [0, 1]
    """
    label = torch.tensor([1]) if label == "genuine" else torch.tensor([0])
    label = label.to(dtype=torch.float32)
    return label


def resize(
    spec: torch.Tensor,
    time_len: int = 864,
    freq_len: int = 400,
    high_freq: bool = True,
) -> torch.Tensor:
    """
    resize

    clip or repeat samples to be in same size

    Parameters
    ----------
    sig : torch.Tensor
        spectrogram to be resized along time-axis

    out_length : int
        desired length of spectrogram along time-axis, default 864

    Returns
    -------
    torch.Tensor
        resized spectrogram
    """

    # repeat spectrogram if its size is samll
    while spec.size()[2] < time_len:
        spec = torch.cat([spec, spec], dim=2)

    # clip
    if high_freq:
        min_freq = spec.size()[1] - freq_len
        out_spec = spec[:, min_freq:, :time_len]
    else:
        out_spec = spec[:, :freq_len, :time_len]

    return out_spec


class Dataset(Dataset):
    """
    Dataset to be trained or evaluated. It extracts log power magnitude
    with window_length = 600 (37 msec) and hop_length = 200 (12 msec)
    """

    spec = Spectrogram(n_fft=798, win_length=600, hop_length=200, normalized=True)
    db_converter = AmplitudeToDB()

    def __init__(
        self,
        annot_path: str,
        ds_path: str,
        device: str = "cuda",
    ) -> None:
        """
        Parameters
        ----------
        annot_path : str
            path to the annotaion file
        ds_path : str
            path to the directory holding samples related to given annotation file
        device : str
            default cuda
        """
        self.ds_path = ds_path
        self.annot_path = annot_path
        self.device = device

        # read given annotation file
        converter = {1: convert_label}
        names = ("file_name", "label", "spkr_id", "x", "y", "z", "w")
        self.dataset = pd.read_csv(
            self.annot_path, converters=converter, names=names, sep=" "
        )
        self.dataset = self.dataset.to_numpy()

    def __getitem__(self, index: int) -> Dict:
        """
        return sample and label for the given index of dataset.
        this function should be implemented if you want to wrap
        pytorch Dataset.
        more information can be found on pytorch documentation.

        Returns
        -------
        Dict
            "feature" -> torch.Tensor, "Label" -> torch.Tensor
        """
        file_path = os.path.join(self.ds_path, self.dataset[index][0])
        wave = torchaudio.load(file_path)[0]
        log_spec = self.db_converter(self.spec(wave))
        feature = resize(log_spec)
        label = self.dataset[index][1]
        feature = feature.to(device=self.device)
        label = label.to(device=self.device)
        sample = {"feature": feature, "label": label}
        return sample

    def __len__(self):
        return len(self.dataset)
