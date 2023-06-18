""" Latent space model as in Hastie section 5.4

This assumes fixed W.
Since W is overdetermined, there is not necessarily a corresponding z1 for every x1, W pair
"""

from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import savemat
from joblib import Parallel, delayed
# from utils import calculate_accuracy, double_hist
import os


def sample(n, p, d, W, sigma, z1, x1, m):
    """ Returns a sample of (y1 | m, z1) """
    # sample tau, beta, epsilon
    Zlatent = np.random.randn(n, d)
    theta = np.random.randn(d, 1) / np.sqrt(d)  # E[||theta||] = 1
    U = np.random.randn(n, p)  # noise for X
    epsilon = np.random.randn(n, 1) * sigma  # noise for Y
    X = Zlatent @ W.T + U
    if m == 1:  # z1 is a training data point
        X[0] = x1
    y = Zlatent @ theta + epsilon
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    y1 = x1 @ beta_hat
    true_y = (z1 @ theta)[0]
    return y1, true_y


d = 20  # fixed latent space dimension
n = 200  # number of data points
num_samples = 100000
ps = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 320, 350, 400, 450, 500, 550, 600,
               700, 850, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000, 6000, 8000])
sigma = 1  # noise std
bin_width = 0.1
num_z1s = 20
fixed_W = True
visualize_hists = False
save_samples = True
calculate_acc = False

save_dir = os.path.join('data/experiment_latent', f'n{n}_d{d}_sigma{sigma}')
os.makedirs(save_dir, exist_ok=True)

mi_accs = np.zeros((num_z1s, len(ps)))
for zi in range(num_z1s):
    print(f'{zi+1} out of {num_z1s}')
    # fixed W
    max_p = ps[-1]
    W = np.random.randn(max_p, d)
    W /= np.linalg.norm(W, axis=1)[..., np.newaxis]  # each row has norm 1
    z1 = np.random.randn(1, d)  # true latent vector
    x1 = z1 @ W.T  # observed vector
    for gi, p in enumerate(tqdm(ps)):
        samples_m0 = Parallel(n_jobs=8)(delayed(sample)(n, p, d, W[:p], sigma, z1, x1[..., :p], 0) for i in range(num_samples))
        samples_m1 = Parallel(n_jobs=8)(delayed(sample)(n, p, d, W[:p], sigma, z1, x1[..., :p], 1) for i in range(num_samples))
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
            'z1': z1, 'x1': x1, 'n': n, 'd': d, 'gamma': p/n, 'sigma': sigma, 'p': p,
            'W': W
        }
        if save_samples:
            savemat(os.path.join(save_dir, f'p{p}_z1{zi}.mat'), save_dict)
        # if visualize_hists:
        #     if not np.mod(gi, 4):
        #         plt.figure()
        #     plt.subplot(2, 2, np.mod(gi, 4)+1)
        #     double_hist(yhat_m0, yhat_m1, bins=200, label1='m=0', label2='m=1', new_fig=False, title=f'gamma={gamma}')
        #     if np.mod(gi, 4) == 3 or gi == len(ps) - 1:
        #         plt.show()
        # if calculate_acc:
        #     min_bin = min(np.min(yhat_m0), np.min(yhat_m1))
        #     max_bin = max(np.max(yhat_m0), np.max(yhat_m1))
        #     bins = np.arange(min_bin - bin_width, max_bin + bin_width, bin_width)
        #     hist_m0, _ = np.histogram(yhat_m0, bins=bins, density=True)
        #     hist_m1, _ = np.histogram(yhat_m1, bins=bins, density=True)
        #     _acc = calculate_accuracy(0.5 * hist_m0 * bin_width, 0.5 * hist_m1 * bin_width)
        #     print(f'Gamma: {p/n:.4f}. MI acc: {_acc}')


