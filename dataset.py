from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Spectrogram, AmplitudeToDB
import torch
import pandas as pd
import os
from typing import Literal, Dict


class Dataset(Dataset):
    def __init__(self, annot_path: str, ds_path: str) -> None:
        """
        Parameters
        ----------
        annot_path : str
            path to the annotaion file
        ds_path : str
            path to the directory holding samples related to given annotation file
        """
        self.ds_path = ds_path
        self.annot_path = annot_path

        # read given annotation file
        converter = {1: self.convert_label}
        names = ("file_name", "label", "spkr_id", "x", "y", "z", "w")
        self.dataset = pd.read_csv(
            self.annot_path, converters=converter, names=names, sep=" "
        )
        self.dataset = self.dataset.to_numpy()
        self.spec_converter = Spectrogram(
            n_fft=798,
            win_length=700,
            hop_length=200,
            normalized=True,
        )
        self.db_converter = AmplitudeToDB()

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
        feature = self._extract_feature(wave)
        label = self.dataset[index][1]
        sample = {"feature": feature, "label": label}
        return sample

    def __len__(self):
        return len(self.dataset)

    def convert_label(self, label: Literal["spoof", "genuine"]) -> torch.Tensor:
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
            [1] if "genuine" else [0]
        """
        label = torch.tensor([1]) if label == "genuine" else torch.tensor([0])
        label = label.to(dtype=torch.int64)
        return label

    def _extract_feature(
        self, wave: torch.Tensor, time_len: int = 64000
    ) -> torch.Tensor:
        """
        cut or repeat wave to the desired time length

        Parameters
        ----------
        wave : torch.Tensor
            wave to be resized
        time_len : int, optional
            maximum length to be resized to, by default 64000

        Returns
        -------
        torch.Tensor
            resized wave
        """
        while wave.size()[1] < time_len:
            wave = torch.cat([wave, wave], dim=1)

        wave = wave[:, :time_len]
        return self.db_converter(self.spec_converter(wave))
