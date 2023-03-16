import numpy as np

from .problem import Problem


class BilinearMinMax(Problem):
    def __init__(self, dim=100, space=[-10, 10], seed=42):
        np.random.seed(seed)
        A = np.random.randn(dim, dim)
        self.A = A
        # self.A = A @ A.T
        # self.A = self.A
        opt_1 = np.random.uniform(*space, dim)
        opt_2 = np.random.uniform(*space, dim)
        self.opt = np.array([opt_1, opt_2])
        print(self.opt)
        super().__init__(opt=self.opt, dim=(dim, 2), space=space)
        print(self.space)

    def fx(self, x):
        if x.shape[-1] != self.dim[0]:
            x = x[..., None]
        opt_1, opt_2 = self.opt
        return np.einsum('...i,...ij,...j->...', x[..., 0, :] - opt_1, self.A, x[..., 1, :] - opt_2)

    def dfx(self, x):
        print(x.shape, self.A.shape)
        opt_1, opt_2 = self.opt
        # should_reshape = (x.shape[-1] != self.dim[0])
        # if should_reshape:
        #     # add new axis (when dim=1): (..., 2) -> (..., 2, 1)
        #     x = x[..., None]
        grad_1 = (x[..., 1, :] - opt_2) @ self.A.T
        grad_2 = -(x[..., 0, :] - opt_1) @ self.A
        # if should_reshape:
        #     # remove the added axis (when dim=1): (..., 1) -> (...)
        #     grad_1, grad_2 = grad_1[..., 0], grad_2[..., 0]
        grad = np.stack((grad_1, grad_2), axis=-1)
        print(x.shape, grad.shape)
        return grad

    def proj(self, x, grad):
        x = x + grad
        return np.clip(x, *self.space)
