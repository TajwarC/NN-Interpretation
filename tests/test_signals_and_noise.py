import torch

from nn_interpretation import DataGenLimits, add_gaussian_noise, generate_clean_signals


def test_generate_clean_signals_shape_and_reproducible():
    limits = DataGenLimits(0.5, 1.5, 0.4, 1.2, 1.0, 4.0, 2.0, 6.0)
    s1 = generate_clean_signals(num_signals=16, signal_length=512, limits=limits, seed=7)
    s2 = generate_clean_signals(num_signals=16, signal_length=512, limits=limits, seed=7)
    assert s1.shape == (16, 512)
    assert torch.allclose(s1, s2)


def test_add_gaussian_noise_sigma_deterministic():
    clean = torch.zeros((8, 128))
    n1 = add_gaussian_noise(clean, sigma=0.2, seed=10)
    n2 = add_gaussian_noise(clean, sigma=0.2, seed=10)
    assert n1.shape == clean.shape
    assert torch.allclose(n1, n2)
    assert not torch.allclose(n1, clean)


def test_add_gaussian_noise_snr_mode():
    clean = torch.ones((4, 32))
    noisy = add_gaussian_noise(clean, snr_db=20.0, seed=4)
    assert noisy.shape == clean.shape

