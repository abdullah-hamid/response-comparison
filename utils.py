import numpy as np
import os
from data_loader import loadrpc



def read_txt_data(filepath):
    """
    Reads data from a simple text file, automatically skipping header lines.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    start_index = -1
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line:
            try:
                float(stripped_line.split()[0])
                start_index = i
                break
            except (ValueError, IndexError):
                continue
    if start_index == -1:
        raise ValueError(f"Could not find any numerical data in {os.path.basename(filepath)}.")
    return np.loadtxt(lines[start_index:])


# --- Universal Data Loader to use loadrpc (based on MTS MATLAB code)---
def load_ground_motion_data(filepath, channel_index=0):
    """
    Loads ground motion data from either a .txt or .tim file.
    """
    _, extension = os.path.splitext(filepath)

    if extension.lower() == '.txt':
        # .txt files are assumed to be single-column (or first column is data)
        data = read_txt_data(filepath)
        if data.ndim > 1:
            return data[:, 0]
        return data

    elif extension.lower() == '.tim':
        rpc_data = loadrpc(filepath)
        return rpc_data['x'][:, channel_index]

    else:
        raise ValueError(f"Unsupported file type: '{extension}'.")



def octave_spacing(f_min, f_max, n_per_octave):
    num_octaves = np.log2(f_max / f_min)
    num_points = int(num_octaves * n_per_octave) + 1
    return f_min * 2 ** (np.arange(num_points) / n_per_octave)


def ieee693_high_rrs_with_tolerances(damping_percent=5.0, num_points=500, performance_level='high', vertical = False, sa_upper_factor=1.5, sa_lower_factor=0.9):
    frequencies = np.logspace(np.log10(0.10), np.log10(100), num=num_points)
    d = damping_percent
    beta = (3.21 - 0.68 * np.log(d)) / 2.1156
    sa = np.zeros_like(frequencies)
    # Calculate multiplier depending if vertical or horizontal
    if vertical:
        nu = 0.8
    else:
        nu = 1.0

    # Performance Level coefficient
    if performance_level.lower() == 'high':
        alpha = 1.0;
    else:
        alpha = 0.5;

    # Calculate final multiplier that depends on direction (horizontal or vertical) and performance level
    rho  = alpha * nu

    for i, f in enumerate(frequencies):
        if f <= 1.1:
            sa[i] = rho * (2.288 * beta * f)
        elif 1.1 < f <= 8.0:
            sa[i] = rho * 2.50 * beta
        elif 8.0 < f <= 33.0:
            sa[i] = rho * (((26.4 * beta - 10.56) / f) - 0.8 * beta + 1.32)
        else:
            sa[i] = rho * (1.0)
    sa_upper = sa_upper_factor * sa
    sa_lower = sa_lower_factor * sa
    return frequencies, sa, sa_upper, sa_lower


def improved_tapered_cosine_wavelet(t, omega_i, t_i, gamma_i, beta=0.05):
    omega_i_prime = omega_i * np.sqrt(1 - beta ** 2)
    delta_t_i = np.arctan(np.sqrt(1 - beta ** 2) / beta) / omega_i_prime
    shifted_time = t - t_i + delta_t_i
    cosine_term = np.cos(omega_i_prime * shifted_time)
    gaussian_term = np.exp(- (shifted_time / gamma_i) ** 2)
    return cosine_term * gaussian_term


def gamma_from_frequency(f):
    return 1.178 * f ** (-0.93)

