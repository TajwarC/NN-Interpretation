"""Dataset and persistence helpers for clean/noisy signal pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class PairedSignalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset exposing (noisy, clean) pairs."""

    def __init__(self, noisy: torch.Tensor, clean: torch.Tensor) -> None:
        if noisy.shape != clean.shape:
            raise ValueError("noisy and clean tensors must have identical shapes.")
        self.noisy = noisy
        self.clean = clean

    def __len__(self) -> int:
        return int(self.noisy.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[idx], self.clean[idx]


def build_dataloader(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    dataset = PairedSignalDataset(noisy=noisy, clean=clean)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_dataset_pt(
    path: str | Path,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist paired tensors and optional metadata to .pt."""
    payload: dict[str, Any] = {
        "clean": clean.detach().cpu(),
        "noisy": noisy.detach().cpu(),
        "metadata": metadata or {},
    }
    torch.save(payload, str(path))


def load_dataset_pt(path: str | Path) -> dict[str, Any]:
    """Load paired tensors and metadata from .pt."""
    payload = torch.load(str(path), map_location="cpu")
    required = {"clean", "noisy", "metadata"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"Dataset file missing keys: {sorted(missing)}")
    return payload

