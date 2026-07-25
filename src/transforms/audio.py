import torch
import torch.nn.functional as F
from torch import nn


class FFTLCNNTransform(nn.Module):
    def __init__(
        self,
        n_fft: int = 1724,
        win_length: int = 1724,
        hop_length: int = 130,
        num_frames: int = 600,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.eps = eps
        self.target_length = (
            n_fft + (num_frames - 1) * hop_length
        )

        self.register_buffer(
            "window",
            torch.blackman_window(win_length),
        )

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        waveform = waveform[: self.target_length]
        if waveform.shape[0] < self.target_length:
            waveform = F.pad(
                waveform,
                (
                    0,
                    self.target_length - waveform.shape[0],
                ),
            )

        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        log_power_spectrum = torch.log(spectrum.abs().square().clamp_min(self.eps))
        return log_power_spectrum.unsqueeze(0)