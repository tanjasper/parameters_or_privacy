from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import membership_advantage, parse_experiment_files
import os

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{newtxtext}')
plt.rcParams['font.family'] = 'Times New Roman'

load_dir = 'data/experiment_time/n128_D1024_unspecified_fixedfeats_sigma0'
ps, z1_idxs, n, num_samples = parse_experiment_files(load_dir, experiment=3)
ps = np.array(ps)[4:]
gammas = np.array(ps) / n
min_perc = 0.01
max_perc = 99.99
num_z1s = 20
num_bins = 150
z1_idxs = z1_idxs[:20]
visualize = False
use_std_err = True

mi_advs = np.zeros((num_z1s, len(ps)))
mses = np.zeros((num_z1s, len(ps)))
for z1i, z1_idx in enumerate(tqdm(z1_idxs)):
    for pi, p in enumerate(ps):
        base_string = f'p{p}_z1idx{z1_idx}.mat'
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
        if z1i == 0 and visualize:
            plt.hist(yhat_m0, bins=bins, density=True)
            plt.hist(yhat_m1, bins=bins, density=True)
            plt.title(pi)
            plt.show()
        mi_advs[z1i, pi] = membership_advantage(hist_m0 * _bin_width, hist_m1 * _bin_width)
        mses[z1i, pi] = np.mean((data['yhat_m0'] - data['truey_m0']) ** 2)

plt.figure(figsize=(5, 4.5))
mi_means = np.mean(mi_advs, 0)
mi_stds = np.std(mi_advs, 0)
if use_std_err:
    mi_stds = mi_stds / np.sqrt(num_z1s)
plt.plot(gammas, mi_means, 'o-', color='tab:blue')
plt.fill_between(gammas, mi_means - mi_stds, mi_means + mi_stds, facecolor='tab:blue', alpha=0.5)
plt.xscale('log')
plt.xlabel(r'$\gamma = p/n$', fontsize=22)
plt.ylabel('Membership Advantage', fontsize=22)
plt.xticks([1, 10], fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout(pad=1)
plt.axvline(x=1, linestyle='--', linewidth=3, color='tab:orange')
plt.savefig('figs/experiment_time.pdf')
plt.show()