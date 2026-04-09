"""Model and training utilities for signal denoising."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig


def build_mlp_denoiser(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int | None = None,
    activation: str = "relu",
    bias: bool = False,
) -> nn.Module:
    """Build a fully connected denoiser with configurable depth."""
    output_dim = input_dim if output_dim is None else output_dim
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive.")

    activation_map: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    if activation not in activation_map:
        raise ValueError(f"Unsupported activation '{activation}'.")

    dims = [input_dim, *list(hidden_dims), output_dim]
    if len(dims) < 2:
        raise ValueError("At least one layer must be defined.")

    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1], bias=bias))
        if idx < len(dims) - 2:
            layers.append(activation_map[activation]())
    return nn.Sequential(*layers)


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    raise ValueError(f"Unsupported optimizer '{config.optimizer}'.")


def _build_loss(config: TrainingConfig) -> nn.Module:
    if config.loss.lower() == "mse":
        return nn.MSELoss()
    if config.loss.lower() == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss '{config.loss}'.")


def _run_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            batch_size = noisy.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += int(batch_size)
    return total_loss / max(total_samples, 1)


def train_denoiser(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, list[float]]:
    """Train a denoising model and return metric history."""
    if config is None:
        config = TrainingConfig()
    if config.seed is not None:
        torch.manual_seed(config.seed)

    device = torch.device(config.device)
    model = model.to(device)
    criterion = _build_loss(config)
    optimizer = _build_optimizer(model, config)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for _ in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            batch_size = noisy.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += int(batch_size)

        history["train_loss"].append(total_loss / max(total_samples, 1))
        if val_loader is not None:
            history["val_loss"].append(_run_eval(model, val_loader, criterion, device))
    return history

