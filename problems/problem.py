import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from collections import defaultdict

from utils import norm
from plots import plot_scalars
from algorithms import algorithms


class Metric:
    OPT_DIST = 'optimal distance'
    GRAD_NORM = 'gradient norm'


class Problem:
    def fx(self, x): raise NotImplementedError
    def dfx(self, x): raise NotImplementedError
    def proj(self, x, grad): raise NotImplementedError
    def get_displayed_x1(self, x): return x[..., 0]
    def get_displayed_x2(self, x): return x[..., 1]

    def __init__(self, opt=None, dim=(2,), space=[-1, 1]):
        self.dim = dim
        self.opts = opt
        self.space = space

        if opt is not None and opt.ndim == len(dim):
            self.opts = opt[None, ...]  # opt is a list of best points

        def metric_opt_dist(x):
            return np.min([
                norm(x - opt, start_idx=-len(self.dim))for opt in self.opts
            ], axis=0)

        def metric_grad_norm(x):
            return norm(self.dfx(x), start_idx=-len(self.dim))**2

        self.metrics = {
            Metric.OPT_DIST: metric_opt_dist,
            Metric.GRAD_NORM: metric_grad_norm,
        }

    def plot_fct(self, ax=None):
        if ax is None:
            ax = plt.figure(figsize=(5, 5)).gca()
        space = np.linspace(*self.space, 20)
        X1, X2 = np.meshgrid(space, space)
        X = np.stack([X1, X2], axis=-1)
        x = np.stack([X1.flatten(), X2.flatten()], axis=-1)

        ax.contourf(X1, X2, self.fx(X), 40)
        ax.quiver(x[:, 0], x[:, 1], *self.dfx(x).T, color='k')
        if self.opts is not None:
            for opt in self.opts:
                ax.scatter(*opt, marker='x', color='red', s=200)

    def plot_trials_convergence(self, trials, metric, title='', ax=None, show_legend=True, show_metric_title=True):
        assert metric in self.metrics

        if ax is None:
            ax = plt.figure(figsize=(5, 5)).gca()

        fct = self.metrics[metric]
        for algo, algo_trials in trials.items():
            algo_trials = np.array(algo_trials)  # (n_trials, n, *dim)
            values = fct(algo_trials)
            plot_scalars(values, label=algo, ax=ax)

        if show_legend:
            ax.legend(loc='lower center', bbox_to_anchor=(.5, -.25), ncol=3)
        if show_metric_title:
            ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xlabel('iteration')
        ax.set_yscale('log')
        ax.set_xscale('log')

    def compare_trials_path(self, trials, title='', axs=None):
        if axs is None:
            _, axs = plt.subplots(1, len(trials), figsize=(
                len(trials) * 5, 5))

        for i, (algo, algo_trials) in enumerate(trials.items()):
            self.plot_fct(axs[i])
            trial = algo_trials[0]
            x1, x2 = self.get_displayed_x1(trial), self.get_displayed_x2(trial)
            axs[i].plot(x1, x2, '-', color='r', label=algo)
            axs[i].scatter(x1[0], x2[0], color='r')
            axs[i].set_title(algo)
        axs[0].set_ylabel(title)

    def run_trials(self, n_iter, n_trials=1, seed=42, **extra_args):
        if seed is not None:
            np.random.seed(seed)

        trials = defaultdict(list)
        for _ in trange(n_trials, desc=f'trials ({extra_args})'):
            x0 = np.random.uniform(*self.space, size=self.dim)
            x0 = self.proj(x0, 0)
            for name, algo in algorithms.items():
                # fmt: off
                xs = algo(x0, self.dfx, self.proj, n_iter, **extra_args) # (n_iter, *dim)
                xs = np.stack(list(xs), axis=0)
                trials[name].append(xs)

        return trials
