import numpy as np

from .problem import Problem


class CovMatrixLearning(Problem):
    def __init__(self, dim=2, space=[-10, 10], seed=42):
        np.random.seed(seed)
        self.d = dim
        A = np.random.randn(dim, dim)
        self.cov = A.T@A
        super().__init__(opt=self.cov, dim=(2, dim, dim), space=space)
        print(self.space)

    def multi_dfx(self, x):
        df = []
        for i in range(x.shape[0]):
            _df = []
            for j in range(x.shape[1]):
                _df.append(self.dfx(x[i, j]))
            df.append(_df)
        return np.array(df)

    def dfx(self, x):
        if len(x.shape) != 3:
            return self.multi_dfx(x)
        theta, phi = x[..., 0, :, :], x[..., 1, :, :]

        z_1 = np.random.multivariate_normal(
            np.zeros(self.d), self.cov, 128)
        z_2 = np.random.multivariate_normal(
            np.zeros(self.d), self.cov, 128)
        zz_1 = np.einsum('...i,...j->...ij', z_1, z_1)  # z_1z_1^T
        zz_2 = np.einsum('...i,...j->...ij', z_2, z_2)  # z_2z_2^T

        grad_1 = np.mean(
            zz_1 - (phi.T + phi)@theta@zz_2, axis=0)
        grad_2 = np.mean(
            np.einsum('...i,...j->...ij', z_2@theta, z_2@theta), axis=0)

        grad = np.stack((grad_1, grad_2), axis=-3)
        return grad

    def proj(self, x, grad):
        x = x + grad
        return x
