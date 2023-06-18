from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import re
from collections import defaultdict
from scipy.io import loadmat


def parse_experiment_files(load_dir, experiment=None):
    """ Obtain the p parameter values and number of z1s used in this experiment """

    all_file_paths = os.listdir(load_dir)
    # parse files
    ps = []
    num_samples = 0
    z1s_per_p = defaultdict(list)
    for f in all_file_paths:
        if experiment in [3, 5]:
            reg_string = "p\d+_z1idx\d+.mat"
        else:
            reg_string = "p\d+_z1\d+.mat"
        if re.match(reg_string, f):
            components = re.split('_', f)
            p = int(components[0][1:])
            if experiment in [3, 5]:
                z1_idx = int(components[1][5:-4])
            else:
                z1_idx = int(components[1][2:-4])
            z1s_per_p[p].append(z1_idx)
            ps.append(p)
            # load the f to obtain number of samples
            if not num_samples:
                data = loadmat(os.path.join(load_dir, f))
                num_samples = data['yhat_m0'].size
                n = data['n'].item()
    ps = list(set(ps))  # remove duplicates
    ps.sort()
    # find the z1 idxs all p's have in common
    z1_sets = [set(z1s_per_p[k]) for k in z1s_per_p]
    z1_idxs = list(z1_sets[0].intersection(*z1_sets))
    return ps, z1_idxs, n, num_samples


def enumerated_product(*args):
    """ enumerate with product to return both idxs and the product iters """
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))


def double_hist(data1, data2, label1='data1', label2='data2', bins=100, min_percentile=0, max_percentile=100,
                title=None, ylabel=None, xlabel=None, savename=None, new_fig=True, return_bins=False, min_bin=None, max_bin=None):
    """ Plot histograms of two sets of data """
    if min_bin is None:
        min_bin = np.nanpercentile(np.concatenate((data1, data2)), min_percentile)
    if max_bin is None:
        max_bin = np.nanpercentile(np.concatenate((data1, data2)), max_percentile)
    if new_fig:
        plt.figure()
    _, plot_bins, _ = plt.hist(data1, bins=bins, range=(min_bin, max_bin), density=True, label=label1, alpha=0.5)
    plt.hist(data2, bins=bins, range=(min_bin, max_bin), density=True, label=label2, alpha=0.5)
    plt.legend()
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if savename:
        plt.savefig(savename)
    if new_fig:
        plt.show()
    if return_bins:
        return plot_bins


def double_2Dhist(data1_x, data1_y, data2_x, data2_y, label1='data1', label2='data2',
                  bins=100, min_percentile=0, max_percentile=100):
    min_xbin = np.nanpercentile(np.concatenate((data1_x, data2_x)), min_percentile)
    max_xbin = np.nanpercentile(np.concatenate((data1_x, data2_x)), max_percentile)
    min_ybin = np.nanpercentile(np.concatenate((data1_y, data2_y)), min_percentile)
    max_ybin = np.nanpercentile(np.concatenate((data1_y, data2_y)), max_percentile)

    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    ax2 = fig.add_subplot(projection='3d')
    ax3 = fig.add_subplot(projection='3d')
    hist1, xedges, yedges = np.histogram2d(data1_x, data1_y, bins=bins, range=[[min_xbin, max_xbin], [min_ybin, max_ybin]], density=True)
    hist2, xedges, yedges = np.histogram2d(data2_x, data2_y, bins=bins,
                                           range=[[min_xbin, max_xbin], [min_ybin, max_ybin]], density=True)

    # widths of bars
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + dx/2, yedges[:-1] + dy/2, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the bars
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz1 = hist1.ravel()
    dz2 = hist2.ravel()

    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz1, zsort='average', label=label1)
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz2, zsort='average', label=label2)
    ax3.bar3d(xpos, ypos, zpos, dx, dy, dz1, zsort='average', alpha=0.5, label=label1)
    ax3.bar3d(xpos, ypos, zpos, dx, dy, dz2, zsort='average', alpha=0.5, label=label2)
    plt.show()


def multi_hist_plot(data, labels=None, min_percentile=0, max_percentile=100, bins=100):
    min_bin = np.percentile(np.concatenate(data), min_percentile)
    max_bin = np.percentile(np.concatenate(data), max_percentile)
    plt.figure()
    for i, d in enumerate(data):
        _hist, _bins = np.histogram(d, bins=bins, range=(min_bin, max_bin), density=True)
        if labels is not None:
            plt.plot(_bins, _hist, label=labels[i])
        else:
            plt.plot(_bins, _hist)
    if labels is not None:
        plt.legend()
    plt.show()


def calculate_accuracy(dist1, dist2):
    """ Given two vectors representing pdfs, calculate \int min(dist1, dist2)

    dist1 and dist2 should have the same length and are assumed to have matching indices
    sum(dist1.sum(), dist2.sum()) should equal 1
    """
    return 1 - np.minimum(dist1, dist2).sum()


def membership_advantage(dist0, dist1):
    # Advantage is Pr(A = 1 | m = 1) - Pr(A = 1 | m = 0)
    # Find where A = 1 (where p(m=1) > p(m=0)) and then calculate the difference
    A1_region = np.where(dist1 > dist0)
    return dist1[A1_region].sum() - dist0[A1_region].sum()


def shuffle(data):
    assert type(data) is tuple
    num_data_points = len(data[0])
    for d in data:
        assert len(d) == num_data_points
    shuffled_idx = torch.randperm(num_data_points)
    shuffled_data = []
    for d in data:
        shuffled_data.append(d[shuffled_idx])
    return tuple(shuffled_data)


def dual_plot(x, y1, y2, ylabel1='', ylabel2='', xlabel='', save_loc='', show=True):
    """ 2 plots each with their own y-axis

    source: https://cmdlinetips.com/2019/10/how-to-make-a-plot-with-two-different-y-axis-in-python-with-matplotlib/
    """
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(x, y1, color="tab:blue", marker="o", markersize=8, linewidth=4)
    # set x-axis label
    ax.set_xlabel(xlabel, fontsize=24)
    # set y-axis label
    ax.set_ylabel(ylabel1, color="tab:blue", fontsize=24)
    plt.yticks(fontsize=14, color='tab:blue')
    plt.xticks(fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(x, y2, color="tab:orange", marker="o", markersize=8, linewidth=4)
    ax2.set_ylabel(ylabel2, color="tab:orange", fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, color='tab:orange')
    plt.tight_layout(pad=1)
    if save_loc:
        plt.savefig(save_loc)
    if show:
        plt.show()