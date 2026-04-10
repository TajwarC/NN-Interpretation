import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random

def generate_signals(num_instances: int, start: int, end: int, step: int, noise_sigma: float = 0.1):
    x = np.arange(start, end, step)
    signal_length = len(x)
    clean = np.zeros((num_instances, signal_length))
    noisy = np.zeros((num_instances, signal_length))
    for i in range(num_instances):
        noise = np.random.normal(0,noise_sigma, len(x))
        y = random.uniform(0,1)*np.sin(random.uniform(0,2.5)*x)
        y_noisy = y+noise
        clean[i,:] = y
        noisy[i,:] = y_noisy
    x_data = torch.tensor(noisy, dtype=torch.float32)
    y_data = torch.tensor(clean, dtype=torch.float32)
        
    return x_data, y_data



