import numpy as np

from .problem import Problem


class BilinearMinMax(Problem):
    def __init__(self, dim=100, space=[-10, 10], seed=42):
        np.random.seed(seed)
        self.A = np.random.randn(dim, dim)
        opt_1 = np.random.uniform(*space, dim)
        opt_2 = np.random.uniform(*space, dim)
        self.opt = np.array([opt_1, opt_2])
        super().__init__(opt=self.opt, dim=(2, dim), space=space)

    def dfx(self, x):
        opt_1, opt_2 = self.opt
        grad_1 = np.einsum('...ij,...j->...i',
                           self.A, x[..., 1, :] - opt_2)
        grad_2 = np.einsum('...ij,...j->...i',
                           -self.A.T, (x[..., 0, :] - opt_1))
        grad = np.stack((grad_1, grad_2), axis=-2)
        return grad

    def proj(self, x, grad):
        x = x + grad
        return np.clip(x, *self.space)
