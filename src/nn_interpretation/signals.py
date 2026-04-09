"""Signal generation utilities."""

from __future__ import annotations

from typing import Tuple

import torch

from .config import DataGenLimits


def _get_generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def _sample_uniform(
    low: float,
    high: float,
    shape: Tuple[int, ...],
    generator: torch.Generator,
    device: str = "cpu",
) -> torch.Tensor:
    return low + (high - low) * torch.rand(shape, generator=generator, device=device)


def generate_clean_signals(
    num_signals: int,
    signal_length: int = 512,
    limits: DataGenLimits | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate sinusoidal signals:
    x = A*sin(W1*pi*T) + B*sin(W2*pi*T)
    """
    if limits is None:
        limits = DataGenLimits(
            a_min=0.5,
            a_max=1.5,
            b_min=0.5,
            b_max=1.5,
            w1_min=1.0,
            w1_max=5.0,
            w2_min=1.0,
            w2_max=5.0,
        )
    if num_signals <= 0:
        raise ValueError("num_signals must be > 0")
    if signal_length <= 0:
        raise ValueError("signal_length must be > 0")

    generator = _get_generator(seed)
    device = "cpu"

    a = _sample_uniform(limits.a_min, limits.a_max, (num_signals, 1), generator, device)
    b = _sample_uniform(limits.b_min, limits.b_max, (num_signals, 1), generator, device)
    w1 = _sample_uniform(limits.w1_min, limits.w1_max, (num_signals, 1), generator, device)
    w2 = _sample_uniform(limits.w2_min, limits.w2_max, (num_signals, 1), generator, device)
    t = torch.linspace(0.0, 1.0, signal_length, device=device).unsqueeze(0)

    return a * torch.sin(w1 * torch.pi * t) + b * torch.sin(w2 * torch.pi * t)

