"""
This experiment is not in the paper.
However, I wrote it after submission to verify the theoretical results regarding Ridge.
In particular, we wish to verify Figs. 3a and 3b.
"""

import os
from tqdm import tqdm
import numpy as np
from scipy.io import savemat
from joblib import Parallel, delayed

def sample(n, p, D, sigma, z1, sigma_noise, m):
    """ Returns a sample of (y1 | m, z1) """
    # sample tau, beta, epsilon
    tau = np.random.randn(n, D)
    beta = np.random.randn(D) / np.sqrt(D)
    if m == 1:
        if p > n+1:  # interpolation regime, y1 is just the measurement for z1
            epsilon = np.random.randn() * sigma
            y1 = (z1 @ beta + epsilon)[0]
            true_y = y1
            return y1, true_y
        else:
            tau[0] = z1
    epsilon = np.random.randn(len(tau)) * sigma
    meas = np.dot(tau, beta) + epsilon
    beta_hat = np.linalg.lstsq(tau[:, :p], meas, rcond=None)[0]
    y1 = z1[0, :p] @ beta_hat
    if m == 0:
        y1 = y1 + np.random.randn() * sigma_noise
    true_y = (z1 @ beta)[0]
    return y1, true_y

n = 100  # number of data points
D = 3000
sigmas_noise = [0, 0.5, 1, 1.25, 1.5, 2]
num_samples = 50000
ps = np.array([120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400,
               600, 1000, 1500, 2000, 3000])
sigma = 1  # noise std
num_z1s = 10
num_procs = 48

save_dir = os.path.join('data/experiment_noise', f'n{n}_D{D}_sigma{sigma}')
os.makedirs(save_dir, exist_ok=True)

print(save_dir)

for zi in range(num_z1s):
    print(f'{zi+1} out of {num_z1s}')
    z1 = np.random.randn(1, D)
    for pi, p in enumerate(tqdm(ps)):
        yhat_m0 = np.zeros((len(sigmas_noise), num_samples))
        yhat_m1 = np.zeros((len(sigmas_noise), num_samples))
        truey_m0 = np.zeros((len(sigmas_noise), num_samples))
        truey_m1 = np.zeros((len(sigmas_noise), num_samples))
        for ni, sigma_noise in enumerate(sigmas_noise):
            samples_m0 = Parallel(n_jobs=num_procs)(delayed(sample)(n, p, D, sigma, z1, sigma_noise, 0) for i in range(num_samples))
            samples_m1 = Parallel(n_jobs=num_procs)(delayed(sample)(n, p, D, sigma, z1, sigma_noise, 1) for i in range(num_samples))
            for i in range(num_samples):
                yhat_m0[ni, i] = samples_m0[i][0]
                yhat_m1[ni, i] = samples_m1[i][0]
                truey_m0[ni, i] = samples_m0[i][1]
                truey_m1[ni, i] = samples_m1[i][1]
        save_dict = {
            'yhat_m0': yhat_m0, 'yhat_m1': yhat_m1,
            'truey_m0': truey_m0, 'truey_m1': truey_m1,
            'z1': z1, 'n': n, 'D': D, 'gamma': p/n, 'sigma': sigma, 'p': p,
            'sigmas_noise': sigmas_noise
        }
        savemat(os.path.join(save_dir, f'p{p}_z1{zi}.mat'), save_dict)
