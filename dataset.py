from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Spectrogram, AmplitudeToDB
import torch
import pandas as pd
import os
from typing import Literal, Dict


DEVICE = "cuda"
TRAIN_ANNOT_PATH = "/content/protocol_V2/protocol_V2/ASVspoof2017_V2_train.trn.txt"
DEV_ANNOT_PATH = "/content/protocol_V2/protocol_V2/ASVspoof2017_V2_dev.trl.txt"
EVAL_ANNOT_PATH = "/content/protocol_V2/protocol_V2/ASVspoof2017_V2_eval.trl.txt"
TRAIN_FOLDER_PATH = "/content/ASVspoof2017_V2_train/ASVspoof2017_V2_train/"
DEV_FOLDER_PATH = "/content/ASVspoof2017_V2_dev/ASVspoof2017_V2_dev/"
EVAL_FOLDER_PATH = "/content/ASVspoof2017_V2_eval/ASVspoof2017_V2_eval/"


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
    label = torch.tensor([1, 0]) if label == "genuine" else torch.tensor([0, 1])
    label = label.to(dtype=torch.float32)
    return label


def resize(spec: torch.Tensor, out_length: int = 864) -> torch.Tensor:
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
    while spec.size()[2] < out_length:
        spec = torch.cat([spec, spec], dim=2)

    # clip
    out_spec = spec[:, :, :out_length]

    return out_spec


class Dataset(Dataset):
    """
    Dataset to be trained or evaluated. It extracts log power magnitude
    with window_length = 600 (37 msec) and hop_length = 200 (12 msec)
    """

    spec = Spectrogram(n_fft=798, win_length=600, hop_length=200, normalized=True)
    db_converter = AmplitudeToDB()

    def __init__(self, annot_path: str, ds_path: str, device: str = "cuda") -> None:
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
        self.divece = device

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
        feature = torchaudio.load(file_path)[0]
        feature = self.db_converter(self.spec(feature))
        feature = resize(feature)
        label = self.dataset[index][1]
        feature = feature.to(device=self.device)
        label = label.to(device=self.device)
        sample = {"feature": feature, "label": label}
        return sample

    def __len__(self):
        return len(self.dataset)
