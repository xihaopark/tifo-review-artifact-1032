"""Dataset-level TIFO stationarity statistics and input transformation.

The historical operator is preserved for result reproducibility.  The
Hermitian variants use the non-redundant real FFT so independently learned
real/imaginary coefficient scales reconstruct a real sequence without dropping
an imaginary residual.  TIFO does not transform the backbone prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _tifo_fft_size(args) -> int:
    """Return the FFT size after optional right-side zero padding."""

    ratio = float(getattr(args, "tifo_zero_pad_ratio", 0.0))
    if ratio < 0:
        raise ValueError("tifo_zero_pad_ratio must be non-negative")
    return int(args.seq_len) + int(int(args.seq_len) * ratio)


class GlobalMaskCalculator:
    """Compute S(k, c) = mean_i A_i(k, c) / std_i A_i(k, c)."""

    def __init__(self, args, device):
        self.args = args
        self.device = torch.device(device)

    def compute_global_statistics(self, loader):
        channels = self.args.enc_in
        variant = getattr(self.args, "tifo_variant", "historical")
        fft_size = _tifo_fft_size(self.args)
        frequencies = fft_size if variant == "historical" else fft_size // 2 + 1
        amp_sum = torch.zeros(frequencies, channels, device=self.device)
        amp2_sum = torch.zeros_like(amp_sum)
        sample_count = 0

        with torch.no_grad():
            for data in loader:
                x = data[0].float().to(self.device)  # [B, L, C]
                if variant == "historical":
                    amplitude = torch.abs(torch.fft.fft(x, n=fft_size, dim=1))
                else:
                    if variant in {"identity_prior", "hermitian_aligned"}:
                        # Match the per-window normalization applied by the
                        # iTransformer/PatchTST input path before TIFO.
                        x = (x - x.mean(1, keepdim=True)) / torch.sqrt(
                            x.var(1, keepdim=True, unbiased=False) + 1e-5
                        )
                    amplitude = torch.abs(torch.fft.rfft(x, n=fft_size, dim=1))
                amp_sum += amplitude.sum(0)
                amp2_sum += amplitude.square().sum(0)
                sample_count += x.size(0)

        if sample_count == 0:
            raise ValueError("cannot compute TIFO statistics from an empty loader")
        mean = amp_sum / sample_count
        variance = (amp2_sum / sample_count - mean.square()).clamp_min(0.0)
        std = torch.sqrt(variance + 1e-5)
        score = mean / (std + 1e-5)
        mode = getattr(self.args, "tifo_score_mode", "data")
        if mode == "data":
            return score
        if mode == "ones":
            return torch.ones_like(score)
        if mode == "permuted":
            # Preserve each channel's score distribution while breaking its
            # association with frequency. A local generator avoids advancing
            # the model/training RNG stream.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(getattr(self.args, "tifo_score_seed", 1729)))
            shuffled = torch.empty_like(score)
            for channel in range(channels):
                order = torch.randperm(frequencies, generator=generator).to(score.device)
                shuffled[:, channel] = score[order, channel]
            return shuffled
        raise ValueError(f"unsupported tifo_score_mode: {mode}")


def run_filter(args, loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return GlobalMaskCalculator(args, device).compute_global_statistics(loader)


class FrequencyDomainFilter(nn.Module):
    """Paper-aligned TIFO input transform with a stable identity initialization."""

    def __init__(self, args, global_mask_amp):
        super().__init__()
        if global_mask_amp is None:
            raise ValueError("TIFO requires dataset-level stationarity statistics")

        self.variant = getattr(args, "tifo_variant", "historical")
        self.fft_size = _tifo_fft_size(args)
        frequencies, channels = global_mask_amp.shape
        expected_frequencies = (
            self.fft_size
            if self.variant == "historical"
            else self.fft_size // 2 + 1
        )
        if frequencies != expected_frequencies or channels != args.enc_in:
            raise ValueError(
                "invalid TIFO statistics shape: "
                f"got {tuple(global_mask_amp.shape)}, "
                f"expected {(expected_frequencies, args.enc_in)}"
            )

        self.seq_len = args.seq_len
        self.residual_alpha = float(getattr(args, "tifo_residual_alpha", 1.0))
        if not 0.0 <= self.residual_alpha <= 1.0:
            raise ValueError("tifo_residual_alpha must be in [0, 1]")
        self.register_buffer(
            "stationarity_score", global_mask_amp.detach().clone().float()
        )
        hidden_dim = int(getattr(args, "filter_dim", 512))

        if self.variant in {
            "historical",
            "hermitian_raw",
            "hermitian_aligned",
        }:
            dropout = float(getattr(args, "tifo_dropout", 0.5))

            def historical_mlp():
                return nn.Sequential(
                    nn.Linear(frequencies, hidden_dim),
                    nn.RReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, frequencies),
                )

            self.linear_r = historical_mlp()
            self.linear_i = historical_mlp()
            return

        prior_strength = float(getattr(args, "tifo_prior_strength", 0.0))
        if prior_strength < 0:
            raise ValueError("tifo_prior_strength must be non-negative")
        normalized_score = self.stationarity_score / self.stationarity_score.mean(
            dim=0, keepdim=True
        ).clamp_min(1e-5)
        score_prior = normalized_score.clamp_min(1e-4).pow(prior_strength)
        self.register_buffer("score_prior", score_prior.clamp(0.25, 4.0))

        def weight_mlp():
            network = nn.Sequential(
                nn.Linear(frequencies, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, frequencies),
            )
            # lambda = 2 * sigmoid(0) = 1, so TIFO starts as the identity.
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
            return network

        self.real_weight_mlp = weight_mlp()
        self.imag_weight_mlp = weight_mlp()

    def frequency_weights(self):
        if self.variant in {
            "historical",
            "hermitian_raw",
            "hermitian_aligned",
        }:
            score_by_channel = self.stationarity_score.transpose(0, 1)
            return (
                self.linear_r(score_by_channel).transpose(0, 1),
                self.linear_i(score_by_channel).transpose(0, 1),
            )
        score_by_channel = self.stationarity_score.transpose(0, 1)
        prior = self.score_prior.transpose(0, 1)
        real_weight = prior * 2.0 * torch.sigmoid(
            self.real_weight_mlp(score_by_channel)
        )
        imag_weight = prior * 2.0 * torch.sigmoid(
            self.imag_weight_mlp(score_by_channel)
        )
        return real_weight.transpose(0, 1), imag_weight.transpose(0, 1)

    def forward(self, x):
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"TIFO expected sequence length {self.seq_len}, got {x.size(1)}"
            )
        if self.residual_alpha == 0.0:
            return x
        spectrum = (
            torch.fft.fft(x, n=self.fft_size, dim=1)
            if self.variant == "historical"
            else torch.fft.rfft(x, n=self.fft_size, dim=1)
        )
        real_weight, imag_weight = self.frequency_weights()
        weighted_spectrum = torch.complex(
            spectrum.real * real_weight,
            spectrum.imag * imag_weight,
        )
        if self.variant == "historical":
            filtered = torch.fft.ifft(
                weighted_spectrum, n=self.fft_size, dim=1
            ).real
        else:
            filtered = torch.fft.irfft(weighted_spectrum, n=self.fft_size, dim=1)
        filtered = filtered[:, : self.seq_len, :]
        if self.residual_alpha == 1.0:
            return filtered
        return x + self.residual_alpha * (filtered - x)


def build_frequency_domain_filter(args, global_mask_amp):
    """Build TIFO without advancing the backbone/training RNG stream."""

    with torch.random.fork_rng(devices=[]):
        return FrequencyDomainFilter(args, global_mask_amp)
