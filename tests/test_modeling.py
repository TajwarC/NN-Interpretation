import torch

from nn_interpretation import (
    TrainingConfig,
    add_gaussian_noise,
    build_dataloader,
    build_mlp_denoiser,
    generate_clean_signals,
    train_denoiser,
)


def test_build_model_bias_disabled():
    model = build_mlp_denoiser(input_dim=64, hidden_dims=[32, 16], bias=False)
    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3
    assert all(layer.bias is None for layer in linear_layers)


def test_train_smoke_reduces_loss():
    clean = generate_clean_signals(num_signals=64, signal_length=64, seed=0)
    noisy = add_gaussian_noise(clean, sigma=0.3, seed=1)
    train_loader = build_dataloader(noisy, clean, batch_size=16, shuffle=True)

    model = build_mlp_denoiser(input_dim=64, hidden_dims=[64], bias=False)
    config = TrainingConfig(epochs=4, learning_rate=5e-3, optimizer="adam", loss="mse", device="cpu", seed=0)
    history = train_denoiser(model, train_loader, config=config)

    assert len(history["train_loss"]) == 4
    assert history["train_loss"][-1] < history["train_loss"][0]

