"""Utilities for extracting and plotting model weight matrices."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def extract_weight_matrices(model: nn.Module) -> dict[str, np.ndarray]:
    """Extract linear-layer weight matrices from a model."""
    matrices: dict[str, np.ndarray] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            matrices[name] = module.weight.detach().cpu().numpy()
    return matrices


def _scale_matrix(matrix: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "layer":
        min_v = float(np.min(matrix))
        max_v = float(np.max(matrix))
        denom = max(max_v - min_v, 1e-12)
        return (matrix - min_v) / denom
    if normalize == "none":
        return matrix
    raise ValueError("normalize must be 'layer' or 'none'.")


def plot_weight_matrices(
    weights: dict[str, np.ndarray],
    out_dir: str | Path,
    cmap: str = "viridis",
    normalize: str = "layer",
) -> list[Path]:
    """Render each matrix as a color image and save it."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    for layer_name, matrix in weights.items():
        scaled = _scale_matrix(matrix, normalize=normalize)
        fig, ax = plt.subplots(figsize=(6, 4))
        image = ax.imshow(scaled, aspect="auto", cmap=cmap)
        ax.set_title(f"Weights: {layer_name}")
        ax.set_xlabel("Input Index")
        ax.set_ylabel("Output Index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        file_name = f"{layer_name.replace('.', '_')}_weights.png"
        file_path = out_path / file_name
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        saved_files.append(file_path)
    return saved_files

