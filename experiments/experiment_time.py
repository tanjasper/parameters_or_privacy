""" You observe the function at some time points """

from tqdm import tqdm
import numpy as np
from scipy.io import savemat
from joblib import Parallel, delayed
import os


def sample(n, p, D, sigma, z1_idx, num_signal_samples, random_features, specified, m):
    """ Returns a sample of (y1 | m, z1) """
    # sample tau, beta, epsilon
    assert m in [0, 1], f'm should be either 0 or 1. Instead, got m = {m}'
    num_tries = 0  # sometimes least squares fails for some hopefully rare instances, avoid those
    while num_tries < 5:
        try:
            k = np.arange(D)  # all frequencies; k will be the frequencies included in the ground truth signal
            if specified:
                if random_features:
                    k = np.sort(np.random.choice(k, p, replace=False))
                else:
                    k = np.arange(p)
                D = k
                # fit_freqs_idx is which of the k true frequencies we will fit the signal to
                fit_freqs_idx = np.range(len(k))  # since specified, we are using the same frequencies as true frequencies
            else:
                if random_features:
                    fit_freqs_idx = np.random.choice(np.arange(len(k)), p, replace=False)  # the frequencies we will fit observations to
                else:
                    fit_freqs_idx = np.arange(len(k))
            N = 2*D-1  # number of frequency components
            beta = np.random.randn(D) / np.sqrt(D)
            z1 = 2 * np.cos(2 * np.pi * k[np.newaxis, ...] * z1_idx / N)
            z1 = z1 / np.linalg.norm(z1)
            if m == 1 and p > n+1:  # z1 is a training data point, y1 is just the measurement for z1
                epsilon = np.random.randn() * sigma
                y1 = (z1 @ beta + epsilon).item()
                true_y = y1
                return y1, true_y
            # choose random time points in the signal to sample
            ints = list(range(num_signal_samples))
            ints.remove(z1_idx)
            xidxs = np.random.choice(ints, n, replace=False)  # random time points
            # we consider real signals, so conjugate symmetric frequency representations (complex exponentials --> cosines)
            tau = 2*np.cos(2 * np.pi * k[np.newaxis, ...] * xidxs[..., np.newaxis] / N)
            if m == 1:
                tau[0] = z1
            tau = tau / np.linalg.norm(tau, axis=1)[..., np.newaxis]  # make measurement vectors unit norm
            epsilon = np.random.randn(len(tau)) * sigma
            meas = tau @ beta + epsilon
            beta_hat = np.linalg.lstsq(tau[:, fit_freqs_idx], meas, rcond=None)[0]
            y1 = z1[0, fit_freqs_idx] @ beta_hat
            true_y = (z1 @ beta).item()
            break
        except:
            num_tries += 1
    return y1, true_y


n = 128  # number of time points observed
D = 1024
specified = False  # D is set to p
z1_idxs = np.sort(np.random.choice(np.arange(1024), 100, replace=False))
num_samples = 100000
ps = [8, 16, 32, 48, 64, 72, 80, 88, 96, 104, 112, 118, 124,
      132, 148, 160, 176, 192, 224, 240, 248, 256, 264, 288,
      320, 384, 448, 464, 480, 488, 496, 504, 512, 520, 528,
      536, 544, 640, 768, 896, 1024]
sigma = 0  # noise std
bin_width = 0.001
num_signal_samples = 1024  # number of total time points
save_samples = True
random_features = False
num_procs = 16

if specified:
    specstring = 'specified'
else:
    specstring = 'unspecified'
if random_features:
    featstring = 'randomfeats'
else:
    featstring = 'fixedfeats'
save_dir = f'data/experiment_time/n{n}_D{D}_{specstring}_{featstring}_sigma{sigma}'
os.makedirs(save_dir, exist_ok=True)

for zi, z1_idx in enumerate(z1_idxs):
    print(f'{zi+1} out of {len(z1_idxs)}')
    for gi, p in enumerate(tqdm(ps)):
        samples_m0 = Parallel(n_jobs=num_procs)(delayed(sample)(n, D, p, sigma, z1_idx, num_signal_samples, random_features, specified, 0) for i in range(num_samples))
        samples_m1 = Parallel(n_jobs=num_procs)(delayed(sample)(n, D, p, sigma, z1_idx, num_signal_samples, random_features, specified, 1) for i in range(num_samples))
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
            'z1': z1_idx, 'n': n, 'D': D, 'gamma': p/n, 'sigma': sigma, 'p': p,
            'specified': specified, 'num_signal_samples': num_signal_samples,
            'random_features': random_features
        }
        if save_samples:
            savemat(os.path.join(save_dir, f'p{p}_z1idx{z1_idx}.mat'), save_dict)
