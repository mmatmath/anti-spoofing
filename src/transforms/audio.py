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

        self.register_buffer(
            "window",
            torch.blackman_window(win_length),
        )

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        waveform = waveform.reshape(-1)
        if waveform.shape[-1] < self.win_length:
            waveform = F.pad(
                waveform,
                (
                    0,
                    self.win_length - waveform.shape[-1],
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

        log_power_spectrum = log_power_spectrum[..., : self.num_frames]
        if log_power_spectrum.shape[-1] < self.num_frames:
            log_power_spectrum = F.pad(
                log_power_spectrum,
                (
                    0,
                    self.num_frames - log_power_spectrum.shape[-1],
                ),
            )

        return log_power_spectrum.unsqueeze(0)
