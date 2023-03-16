import matplotlib.pyplot as plt
import numpy as np


def plot_scalars(values, label='', ax=None):
    if ax is None:
        ax = plt.gca()
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    ax.plot(mean, label=label)
    ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=.2)
