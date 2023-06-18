""" Latent space"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import membership_advantage, parse_experiment_files
import os

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{newtxtext}')
plt.rcParams['font.family'] = 'Times New Roman'

load_dir = 'data/experiment_latent/n200_d20_sigma1'
ps, z1_idxs, n, num_samples = parse_experiment_files(load_dir, experiment=4)
ps = np.array(ps)[4:]
gammas = ps / n
min_perc = 0.01
max_perc = 99.99
num_z1s = 20
num_bins = 150
use_std_err = True

mi_advs_all = np.zeros((num_z1s, len(ps)))
mses_all = np.zeros((num_z1s, len(ps)))

for z1_idx in tqdm(range(num_z1s)):
    for pi, p in enumerate(ps):
        base_string = f'p{p}_z1{z1_idx}.mat'
        data = loadmat(os.path.join(load_dir, base_string))
        yhat_m0 = data['yhat_m0'][0]
        yhat_m1 = data['yhat_m1'][0]
        truey_m0 = data['truey_m0'][0]
        min_bin = np.nanpercentile(np.concatenate((yhat_m0, yhat_m1)), min_perc)
        max_bin = np.nanpercentile(np.concatenate((yhat_m0, yhat_m1)), max_perc)
        bins = np.linspace(min_bin, max_bin, num_bins)
        _bin_width = bins[1] - bins[0]
        hist_m0, _ = np.histogram(yhat_m0, bins=bins, density=True)
        hist_m1, _ = np.histogram(yhat_m1, bins=bins, density=True)
        mi_advs_all[z1_idx, pi] = membership_advantage(hist_m0 * _bin_width, hist_m1 * _bin_width)
        mses_all[z1_idx, pi] = np.mean((data['yhat_m0'] - data['truey_m0']) ** 2)

mi_advs = np.mean(mi_advs_all, 0)
mi_stds = np.std(mi_advs_all, 0) / np.sqrt(num_z1s)
mses = np.mean(mses_all, 0)
mses_stds = np.std(mses_all, 0)

plt.figure(figsize=(5,4.5))
plt.plot(gammas, mi_advs, 'o-', linewidth=3, markersize=5)
plt.fill_between(gammas, mi_advs - mi_stds, mi_advs + mi_stds, alpha=0.4)
plt.xscale('log')
plt.xlabel(r'$\gamma = p/n$', fontsize=22)
plt.ylabel('Membership Advantage', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout(pad=1)
plt.axvline(x=1, linestyle='--', linewidth=3, color='tab:orange')
plt.savefig('figs/experiment_latent.pdf')
plt.show()