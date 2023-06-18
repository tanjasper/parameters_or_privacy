""" Figure 2a """

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt

settings = [
    {
         'n': int(1e3),
         'D': int(1e7),
         'sigma': 1,
         'gammas': 1.04 ** np.arange(1, 234, 1).astype(int)
    }
]
num_settings = len(settings)
num_x0_samples = 100

plt.figure(figsize=(5.5,4.5))
for seti in range(num_settings):
    s = settings[seti]
    n, D, sigma, gammas = s['n'], s['D'], s['sigma'], s['gammas']
    ps = np.round(gammas * n).astype(int)
    mi_advs = np.zeros((len(gammas), num_x0_samples))
    for kk in tqdm(range(num_x0_samples)):
        x0 = np.random.randn(D, 1)
        x0_sq_norm = np.linalg.norm(x0) ** 2
        x0p_sq_norms = np.zeros(len(gammas))
        for i, p in enumerate(ps):
            x0p_sq_norms[i] = np.linalg.norm(x0[:p]) ** 2

        # variances of the gaussians
        sigma0s = np.zeros(len(gammas))
        for i, g in enumerate(gammas):
            if g <= 1:
                sigma0s[i] = np.sqrt((1/D + (1+sigma**2-ps[i]/D)/(n-ps[i]-1)) * x0p_sq_norms[i])
            else:
                sigma0s[i] = np.sqrt((n/ps[i]) * (1/D + (1+sigma**2 - ps[i] / D)/(ps[i] - n - 1)) * x0p_sq_norms[i])
        sigma1 = np.sqrt(sigma**2 + x0_sq_norm / D)

        # alphas
        alphas = np.zeros(len(gammas))
        for i in range(len(gammas)):
            sigma0 = sigma0s[i]
            if sigma0 > sigma1:
                alphas[i] = np.sqrt((sigma0**2 * sigma1**2 * np.log(sigma0**2 / sigma1**2)) / (sigma0**2 - sigma1**2))
            else:
                alphas[i] = np.sqrt(
                    (sigma0 ** 2 * sigma1 ** 2 * np.log(sigma1 ** 2 / sigma0 ** 2)) / (sigma1 ** 2 - sigma0 ** 2))

        # membership advantages
        for i in range(len(gammas)):
            if sigma0s[i] > sigma1:
                mi_advs[i, kk] = 2 * (norm.cdf(alphas[i] / sigma1) - norm.cdf(alphas[i] / sigma0s[i]))
            else:
                mi_advs[i, kk] = 2 * (norm.cdf(alphas[i] / sigma0s[i]) - norm.cdf(alphas[i] / sigma1))

    plt.plot(gammas, mi_advs.mean(1), linewidth=6, label=f'n={n}, D={D}, $\sigma$={sigma}')

plt.xlabel('$\gamma = p/n$', fontsize=22)
plt.xlim(1)
plt.ylim([0, 1])
plt.xscale('log')
plt.ylabel('Membership Advantage', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout(pad=1.0)
# plt.legend(fontsize=18)
plt.savefig('figs/asymptotic_membership_advantage.pdf')
plt.show()

print('done')
