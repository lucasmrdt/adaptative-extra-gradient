import numpy as np

from algorithms import simplex_proj
from utils import norm
from .problem import Problem


class RessourceAllocationPb(Problem):
    def __init__(self, dim=2, space=[0, 1], seed=42):
        np.random.seed(seed)
        self.mu = np.array([1, .8])
        # self.opt = np.array([.5, .5])
        self.opt = np.ones(dim) / dim
        self.lbd = 2
        super().__init__(opt=self.opt, dim=(dim,), space=[0, 1])

    def dual_local_norm(self, v, x):
        return np.sum(x*np.abs(v), axis=-1)

    def fx(self, x):
        y = np.sum(1/(x + 1e-10), axis=-1)
        y = np.clip(y, 0, 50)
        return y

    def dfx(self, x):
        max_norm = 50
        grad = -1/(x**2 + 1e-10)
        n = norm(grad, start_idx=-1)
        grad[n > max_norm] *= max_norm/n[n > max_norm][:, None]
        grad[..., 1] = -grad[..., 1]
        # if norm(grad, start_idx=-1) > max_norm:
        #     grad /= norm(grad)
        return grad

    def proj(self, x, grad):
        x = x + grad
        # print('before', x)
        # x = np.clip(x, *self.space)
        # print(x)
        # x = np.clip(x, *self.space)
        # print(x.shape, grad.shape if grad is not 0 else 0)
        # print(1/(x**2) - grad)
        # print(1 - grad*x**2)
        # print(np.maximum(1 - grad*x**2, 0))
        # x = x / np.sqrt(np.maximum(1 - grad*x**2, 1e-10))
        # print('before ok', x)
        x = simplex_proj(x)
        x = np.clip(x, 1e-10, 1)
        x = simplex_proj(x)
        # print('after ok', x)
        # print('ok', x.shape)
        return x
