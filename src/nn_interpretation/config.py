"""Configuration dataclasses for data generation and training."""

from dataclasses import dataclass


@dataclass(slots=True)
class DataGenLimits:
    """Ranges used to sample sinusoid parameters."""

    a_min: float
    a_max: float
    b_min: float
    b_max: float
    w1_min: float
    w1_max: float
    w2_min: float
    w2_max: float


@dataclass(slots=True)
class NoiseConfig:
    """Configuration for Gaussian noise."""

    sigma: float | None = None
    snr_db: float | None = None
    seed: int | None = None


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters used by the denoiser training loop."""

    epochs: int = 50
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss: str = "mse"
    device: str = "cpu"
    seed: int | None = None

