import numpy as np
import torch
import torchaudio
from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class ASVSpoofDataset(BaseDataset):
    LABEL_TO_ID = {
        "spoof": 0,
        "bonafide": 1,
    }

    def __init__(
        self, audio_dir, protocol_path, *args, **kwargs
    ):
        self.audio_dir = Path(audio_dir)
        self.protocol_path = Path(protocol_path)
        index = self._parse_protocol()
        super().__init__(index, *args, **kwargs)

    def _parse_protocol(self):
        index = []
        with self.protocol_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                fields = line.strip().split()
                smth1, path, _, smth2, label = fields
                audio_path = self.audio_dir / f"{path}.flac"
                index.append(
                    {
                        "path": audio_path,
                        "key": path,
                        "label": self.LABEL_TO_ID[label],
                    }
                )
        return index
    
    def load_object(self, path):
        waveform, sample_rate = torchaudio.load(path)
        return waveform.mean(dim=0)
