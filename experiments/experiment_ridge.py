"""
This experiment was not in the submission.
However, I wrote it after submission to verify the theoretical results regarding Ridge.
In particular, we wish to verify Figs. 3a and 3b.
"""

""" Linear regression on Gaussian data -- distributions are simulated """

import os
from tqdm import tqdm
import numpy as np
from scipy.io import savemat
from joblib import Parallel, delayed

def sample(n, p, D, sigma, z1, lmbd, m):
    """ Returns a sample of (y1 | m, z1) """
    # sample tau, beta, epsilon
    tau = np.random.randn(n, D)
    beta = np.random.randn(D) / np.sqrt(D)
    if m == 1:
        tau[0] = z1
    epsilon = np.random.randn(len(tau)) * sigma
    meas = np.dot(tau, beta) + epsilon
    Xp = tau[:, :p]
    # beta_hat = np.linalg.lstsq(Xp.T @ Xp + n * lmbd * np.eye(p), Xp.T @ meas, rcond=None)[0]
    # https://stackoverflow.com/questions/27476933/numpy-linear-regression-with-regularization
    # y1 = z1[0, :p] @ beta_hat
    # manually calculating pseudoinverse seems to be much quicker than using lstsq
    if nlmbd > 0:
        Xinv = np.linalg.inv(Xp.T @ Xp + n * lmbd * np.eye(p)) @ Xp.T
    else:
        Xinv = Xp.T @ np.linalg.inv(Xp @ Xp.T)
    y1 = z1[0, :p] @ Xinv @ meas
    true_y = (z1 @ beta)[0]
    return y1, true_y

n = 100  # number of data points
D = 3000
nlmbds = [0, 30, 100, 200, 500]
num_samples = 50000
ps = np.array([105, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
               210, 220, 230, 240, 250, 275, 300, 325, 350, 400, 500,
               600, 800, 1000, 1250, 1500, 2000, 3000])
sigma = 1  # noise std
num_z1s = 10
num_procs = 64

save_dir = os.path.join('data/experiment_ridge', f'n{n}_D{D}_sigma{sigma}')
os.makedirs(save_dir, exist_ok=True)

print(save_dir)

for zi in range(num_z1s):
    print(f'{zi+1} out of {num_z1s}')
    z1 = np.random.randn(1, D)
    for pi, p in enumerate(tqdm(ps)):
        yhat_m0 = np.zeros((len(nlmbds), num_samples))
        yhat_m1 = np.zeros((len(nlmbds), num_samples))
        truey_m0 = np.zeros((len(nlmbds), num_samples))
        truey_m1 = np.zeros((len(nlmbds), num_samples))
        for li, nlmbd in enumerate(nlmbds):
            samples_m0 = Parallel(n_jobs=num_procs)(delayed(sample)(n, p, D, sigma, z1, nlmbd / n, 0) for i in range(num_samples))
            samples_m1 = Parallel(n_jobs=num_procs)(delayed(sample)(n, p, D, sigma, z1, nlmbd / n, 1) for i in range(num_samples))
            for i in range(num_samples):
                yhat_m0[li, i] = samples_m0[i][0]
                yhat_m1[li, i] = samples_m1[i][0]
                truey_m0[li, i] = samples_m0[i][1]
                truey_m1[li, i] = samples_m1[i][1]
        save_dict = {
            'yhat_m0': yhat_m0, 'yhat_m1': yhat_m1,
            'truey_m0': truey_m0, 'truey_m1': truey_m1,
            'z1': z1, 'n': n, 'D': D, 'gamma': p/n, 'sigma': sigma, 'p': p,
            'nlmbds': nlmbds
        }
        savemat(os.path.join(save_dir, f'p{p}_z1{zi}.mat'), save_dict)
