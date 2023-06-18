from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import membership_advantage, parse_experiment_files
import os
from scipy.stats import norm

load_dir = 'data/experiment1/n100_D3000_sigma1'
ps, z1_idxs, n, num_samples = parse_experiment_files(load_dir, experiment=1)
ps = np.array(ps)
gammas = ps / n
ps = ps[gammas > 1]
gammas = gammas[gammas > 1]
min_perc = 0.01
max_perc = 99.99
num_z1s = 20
num_bins = 150
D = 3000

print('Calculating membership advantages')
mi_advs = np.zeros((num_z1s, len(ps)))
mse = np.zeros((num_z1s, len(ps)))
x0s = np.zeros((num_z1s, D))
for z1_idx in tqdm(range(num_z1s)):
    for pi, p in enumerate(ps):
        base_string = f'p{p}_z1{z1_idx}.mat'
        data = loadmat(os.path.join(load_dir, base_string))
        yhat_m0 = data['yhat_m0'][0]
        yhat_m1 = data['yhat_m1'][0]
        truey_m0 = data['truey_m0'][0]
        x0s[z1_idx] = data['z1']
        min_bin = np.nanpercentile(np.concatenate((yhat_m0, yhat_m1)), min_perc)
        max_bin = np.nanpercentile(np.concatenate((yhat_m0, yhat_m1)), max_perc)
        bins = np.linspace(min_bin, max_bin, num_bins)
        _bin_width = bins[1] - bins[0]
        hist_m0, _ = np.histogram(yhat_m0, bins=bins, density=True)
        hist_m1, _ = np.histogram(yhat_m1, bins=bins, density=True)
        mi_advs[z1_idx, pi] = membership_advantage(hist_m0 * _bin_width, hist_m1 * _bin_width)
        mse[z1_idx, pi] = np.mean((data['truey_m0'] - data['yhat_m0']) ** 2)

sigma = data['sigma'].item()
D = data['D'].item()
theor_ps = ps[ps > n]
theor_gammas = theor_ps / n
theor_mi = np.zeros((num_z1s, len(theor_ps)))
for z1_idx in range(num_z1s):
    x0 = x0s[z1_idx][..., np.newaxis]
    x0_sq_norm = np.linalg.norm(x0) ** 2
    sigma1 = np.sqrt(sigma ** 2 + x0_sq_norm / D)
    for i, p in enumerate(theor_ps):
        x0p_sq_norm = np.linalg.norm(x0[:p]) ** 2
        if p/n <= 1:
            sigma0 = np.sqrt((1/D + (1+sigma**2-ps[i]/D)/(n-ps[i]-1)) * x0p_sq_norm)
        else:
            sigma0 = np.sqrt((n/p) * (1/D + (1+sigma**2 - p / D)/(p - n - 1)) * x0p_sq_norm)
        if sigma0 > sigma1:
            alpha = np.sqrt((sigma0**2 * sigma1**2 * np.log(sigma0**2 / sigma1**2)) / (sigma0**2 - sigma1**2))
            theor_mi[z1_idx, i] = 2 * (norm.cdf(alpha / sigma1) - norm.cdf(alpha / sigma0))
        else:
            alpha = np.sqrt((sigma0 ** 2 * sigma1 ** 2 * np.log(sigma1 ** 2 / sigma0 ** 2)) / (sigma1 ** 2 - sigma0 ** 2))
            theor_mi[z1_idx, i] = 2 * (norm.cdf(alpha / sigma0) - norm.cdf(alpha / sigma1))

plt.figure(figsize=(5.5,4.5))
mi_stds = np.std(mi_advs, 0)
mi_means = np.mean(mi_advs, 0)
theor_mi_means = np.mean(theor_mi, 0)
theor_mi_stds = np.std(theor_mi, 0)
plt.plot(gammas, mi_means, 'o-', label='Experimental', color='tab:orange', markersize=8)
plt.fill_between(gammas, mi_means - mi_stds, mi_means + mi_stds, facecolor='tab:orange', alpha=0.5)
plt.plot(theor_gammas, theor_mi_means, '--', label='Asymptotic', linewidth=4, color='tab:blue')
plt.xscale('log')
plt.xlabel(r'$\gamma = p/n$', fontsize=22)
plt.ylabel('Membership Advantage', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout(pad=1)
plt.savefig('figs/experiment1.pdf')
plt.show()