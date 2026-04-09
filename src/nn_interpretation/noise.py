"""Noise functions for clean signals."""

from __future__ import annotations

import torch


def _noise_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def add_gaussian_noise(
    clean: torch.Tensor,
    sigma: float | None = None,
    snr_db: float | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Return a noisy copy of the input signals."""
    if sigma is None and snr_db is None:
        raise ValueError("Either sigma or snr_db must be provided.")
    if sigma is not None and sigma < 0:
        raise ValueError("sigma must be non-negative.")

    generator = _noise_generator(seed)

    if snr_db is not None:
        signal_power = clean.pow(2).mean(dim=1, keepdim=True)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        sigma_tensor = torch.sqrt(noise_power)
        noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype) * sigma_tensor
    else:
        noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype) * float(sigma)

    return clean + noise

