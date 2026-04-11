import numpy as np
import matplotlib.pyplot as plt
import torch
import random

'''
Utility function for generating synthetic sinusoidal signals with additive Gaussian noise.

Args:
    num_instances (int): Number of signal instances to generate.
    start (float): Starting value of the x-axis for the signals.
    end (float): Ending value of the x-axis for the signals.
    step (float): Step size for the x-axis values.
    noise_sigma (float, optional): Standard deviation of the Gaussian noise to be added. Default is 0.1.

Returns:
    x_data (torch.Tensor): Tensor containing the noisy signals of shape (num_instances, signal_length).
    y_data (torch.Tensor): Tensor containing the clean signals of shape (num_instances, signal_length)).

'''

def signal_generator(num_instances: int, start: float=0, end: float=32.0, step: int=0.2, noise_sigma: float = 0.1):
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
        
    return x_data, y_data, signal_length



