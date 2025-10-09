import math, random, statistics, itertools
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt

# Refactored from hardcoded function name
def replicate_abm(func, p, n_seeds=12): 
    seeds = list(range(1, n_seeds+1))
    prey_list, pred_list = [], []
    for s in seeds:
        P, Q = func(p, seed=s)
        prey_list.append(P); pred_list.append(Q)
    return np.vstack(prey_list), np.vstack(pred_list)

def mean_ci(arr2d):
    mean = arr2d.mean(axis=0)
    std = arr2d.std(axis=0, ddof=1)
    n = arr2d.shape[0]
    half = 1.96 * std / math.sqrt(n) if n>1 else np.zeros_like(mean)
    return mean, mean-half, mean+half

def plot_with_ci(mean, lo, hi, label):
    plt.figure()
    plt.fill_between(range(len(mean)), lo, hi, alpha=0.2)
    plt.plot(mean, label=label)
    plt.xlabel('Time step'); plt.ylabel('Population')
    plt.title(label); plt.legend(); plt.show()

def extinction_prob(mat, threshold=1):
    final = mat[:, -1]
    return (final < threshold).mean()
