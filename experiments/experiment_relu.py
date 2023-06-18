from tqdm import tqdm
import numpy as np
from scipy.io import savemat
from joblib import Parallel, delayed
import os


def sample(n, p, D, sigma, z1, m):
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
    # observed measurements are random relu features
    meas_vectors = np.random.randn(D, p)
    meas_vectors = meas_vectors / np.linalg.norm(meas_vectors, axis=0)
    observed_features = np.maximum(tau @ meas_vectors, 0)
    beta_hat = np.linalg.lstsq(observed_features, meas, rcond=None)[0]
    y1 = z1 @ meas_vectors @ beta_hat
    true_y = (z1 @ beta)[0]
    return y1, true_y


n = 100  # number of data points
D = 5000
num_samples = 100000
ps = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
               110, 120, 140, 160, 180, 200, 225, 250, 275, 300, 325, 350, 375, 400,
               450, 500, 550, 600, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
sigma = 1  # noise std
bin_width = 0.001
num_z1s = 1
save_samples = True
specified = False
overwrite = False

if specified:
    Dstring = '_specified_'
else:
    Dstring = f'_D{D}_'
save_dir = os.path.join('data/experiment_relu', f'n{n}{Dstring}sigma{sigma}')
if save_samples:
    os.makedirs(save_dir, exist_ok=True)


mi_accs = np.zeros((num_z1s, len(ps)))
for zi in range(num_z1s):
    print(f'{zi+1} out of {num_z1s}')
    full_z1 = np.random.randn(1, D)
    for p in tqdm(ps):
        if os.path.exists(os.path.join(save_dir, f'p{p}_z1{zi}.mat')) and not overwrite:
            continue
        if specified:
            z1 = full_z1[:p]
            D = p
        else:
            z1 = full_z1
        samples_m0 = Parallel(n_jobs=32)(delayed(sample)(n, p, D, sigma, z1, 0) for i in range(num_samples))
        samples_m1 = Parallel(n_jobs=32)(delayed(sample)(n, p, D, sigma, z1, 1) for i in range(num_samples))
        yhat_m0 = np.zeros(num_samples)
        yhat_m1 = np.zeros(num_samples)
        truey_m0 = np.zeros(num_samples)
        truey_m1 = np.zeros(num_samples)
        for i in range(num_samples):
            yhat_m0[i] = samples_m0[i][0]
            yhat_m1[i] = samples_m1[i][0]
            truey_m0[i] = samples_m0[i][1]
            truey_m1[i] = samples_m1[i][1]
        save_dict = {
            'yhat_m0': yhat_m0, 'yhat_m1': yhat_m1,
            'truey_m0': truey_m0, 'truey_m1': truey_m1,
            'z1': z1, 'n': n, 'D': D, 'gamma': p/n, 'sigma': sigma, 'p': p,
            'specified': specified, 'num_z1s': num_z1s
        }
        if save_samples:
            savemat(os.path.join(save_dir, f'p{p}_z1{zi}.mat'), save_dict)
