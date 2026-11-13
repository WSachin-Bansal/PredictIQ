"""Audio utilities: waveform -> fixed-length log-Mel spectrogram."""
from __future__ import annotations

import numpy as np
import torch
import torchaudio


class MelExtractor(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        clip_seconds: float = 2.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
    ):
        super().__init__()
        self.n_samples = int(sample_rate * clip_seconds)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) or (T,) -> (B, 1, n_mels, frames)."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        # pad / crop to fixed length
        if wav.shape[-1] < self.n_samples:
            wav = torch.nn.functional.pad(wav, (0, self.n_samples - wav.shape[-1]))
        else:
            wav = wav[..., : self.n_samples]
        spec = self.to_db(self.mel(wav))
        # per-clip standardisation
        spec = (spec - spec.mean(dim=(-2, -1), keepdim=True)) / (
            spec.std(dim=(-2, -1), keepdim=True) + 1e-6
        )
        return spec.unsqueeze(1)


def load_wav(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)  # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav
