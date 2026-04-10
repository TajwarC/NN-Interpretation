"""Public API for the nn_interpretation package."""

from .config import DataGenLimits, NoiseConfig, TrainingConfig
from .datasets import PairedSignalDataset, build_dataloader, load_dataset_pt, save_dataset_pt
from .modeling import build_mlp_denoiser, train_denoiser
from .noise import add_gaussian_noise
from .signals import generate_clean_signals
from .visualization import extract_weight_matrices, plot_weight_matrices
from .signal_generator import generate_signals

__all__ = [
    "DataGenLimits",
    "NoiseConfig",
    "TrainingConfig",
    "PairedSignalDataset",
    "build_dataloader",
    "load_dataset_pt",
    "save_dataset_pt",
    "build_mlp_denoiser",
    "train_denoiser",
    "add_gaussian_noise",
    "generate_clean_signals",
    "extract_weight_matrices",
    "plot_weight_matrices",
    "generate_signals"
]

