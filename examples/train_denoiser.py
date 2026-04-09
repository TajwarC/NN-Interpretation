"""End-to-end example: generate data, train denoiser, and visualize weights."""

from pathlib import Path

from nn_interpretation import (
    DataGenLimits,
    TrainingConfig,
    add_gaussian_noise,
    build_dataloader,
    build_mlp_denoiser,
    extract_weight_matrices,
    generate_clean_signals,
    load_dataset_pt,
    plot_weight_matrices,
    save_dataset_pt,
    train_denoiser,
)


def main() -> None:
    num_signals = 1000
    signal_length = 512

    limits = DataGenLimits(
        a_min=0.2,
        a_max=1.5,
        b_min=0.2,
        b_max=1.5,
        w1_min=1.0,
        w1_max=8.0,
        w2_min=1.0,
        w2_max=8.0,
    )

    clean = generate_clean_signals(num_signals=num_signals, signal_length=signal_length, limits=limits, seed=42)
    noisy = add_gaussian_noise(clean, sigma=0.2, seed=42)

    dataset_path = Path("outputs/signal_dataset.pt")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_pt(dataset_path, clean=clean, noisy=noisy, metadata={"sigma": 0.2, "signal_length": signal_length})

    loaded = load_dataset_pt(dataset_path)
    train_loader = build_dataloader(loaded["noisy"], loaded["clean"], batch_size=64, shuffle=True)

    model = build_mlp_denoiser(input_dim=signal_length, hidden_dims=[256, 128, 256], activation="relu", bias=False)
    config = TrainingConfig(epochs=10, learning_rate=1e-3, optimizer="adam", loss="mse", device="cpu", seed=123)
    history = train_denoiser(model, train_loader, config=config)

    weights = extract_weight_matrices(model)
    output_files = plot_weight_matrices(weights, out_dir="outputs/weights", cmap="viridis", normalize="layer")

    print("Final training loss:", history["train_loss"][-1])
    print("Saved weight plots:")
    for path in output_files:
        print(" -", path)


if __name__ == "__main__":
    main()

