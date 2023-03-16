import numpy as np

from .problem import Problem


class BilinearMinMax(Problem):
    def __init__(self, dim=2, space=[-10, 10], seed=42):
        np.random.seed(seed)
        self.cov = np.random.randn(dim, dim)
        super().__init__(opt=self.opt, dim=(dim, 2), space=space)
        print(self.space)

    def fx(self, x):
        if x.shape[-1] != self.dim[0]:
            x = x[..., None]
        opt_1, opt_2 = self.opt
        return np.einsum('...i,...ij,...j->...', x[..., 0, :] - opt_1, self.A, x[..., 1, :] - opt_2)

    def dfx(self, x):
        opt_1, opt_2 = self.opt
        should_reshape = (x.shape[-1] != self.dim[0])
        if should_reshape:
            # add new axis (when dim=1): (..., 2) -> (..., 2, 1)
            x = x[..., None]
        grad_1 = (x[..., 1, :] - opt_2) @ self.A.T
        grad_2 = -(x[..., 0, :] - opt_1) @ self.A
        if should_reshape:
            # remove the added axis (when dim=1): (..., 1) -> (...)
            grad_1, grad_2 = grad_1[..., 0], grad_2[..., 0]
        grad = np.stack((grad_1, grad_2), axis=-1)
        return grad

    def proj(self, x, grad):
        x = x + grad
        return np.clip(x, *self.space)
