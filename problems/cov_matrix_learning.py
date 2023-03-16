import numpy as np

from .problem import Problem


class CovMatrixLearning(Problem):
    def __init__(self, dim=2, space=[-10, 10], seed=42):
        np.random.seed(seed)
        self.d = dim
        self.cov = np.random.randn(dim, dim)
        super().__init__(opt=self.cov, dim=(2, dim, dim), space=space)
        print(self.space)

    def dfx(self, x):
        print('ok')
        theta, phi = x[..., 0, :, :], x[..., 1, :, :]
        x = np.random.multivariate_normal(
            np.zeros(self.d), self.cov, 128)
        z = np.random.multivariate_normal(
            np.zeros(self.d), self.cov, 128)
        # grad_1 = x.T@x
        print(phi.shape, theta.shape, z.shape)
        zz = np.einsum('...i,...j->...ij', z, z)
        # grad_1 = np.einsum('...ij,...j->...i', phi.T + phi, theta @
        grad_1 = (phi.T + phi)@theta@z@z.T
        print(grad_1.shape)
        grad_2 = x@x.T
        grad = np.stack((grad_1, grad_2), axis=-2)
        return grad

    def proj(self, x, grad):
        x = x + grad
        return np.clip(x, *self.space)
