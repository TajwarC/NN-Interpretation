from pathlib import Path

import torch

from nn_interpretation import (
    build_mlp_denoiser,
    extract_weight_matrices,
    load_dataset_pt,
    plot_weight_matrices,
    save_dataset_pt,
)


def test_save_and_load_dataset(tmp_path: Path):
    clean = torch.randn(10, 32)
    noisy = clean + 0.1 * torch.randn(10, 32)
    path = tmp_path / "pairs.pt"

    save_dataset_pt(path, clean=clean, noisy=noisy, metadata={"note": "unit-test"})
    loaded = load_dataset_pt(path)

    assert "clean" in loaded and "noisy" in loaded and "metadata" in loaded
    assert loaded["clean"].shape == clean.shape
    assert loaded["metadata"]["note"] == "unit-test"


def test_extract_and_plot_weights(tmp_path: Path):
    model = build_mlp_denoiser(input_dim=16, hidden_dims=[8], bias=False)
    weights = extract_weight_matrices(model)
    out_files = plot_weight_matrices(weights, out_dir=tmp_path)

    assert len(weights) > 0
    assert len(out_files) == len(weights)
    assert all(path.exists() for path in out_files)

