import numpy as np

from algorithms import projections
from .problem import Problem


class RockPaperScissorsPb(Problem):
    def __init__(self):
        self.A = np.array([
            #rock,  paper, scissors
            [0,     -1,    1],   # rock
            [1,     0,     -1],  # paper
            [-1,    1,     0],   # scissors
        ])
        opt = np.array([1/3, 1/3, 1/3])
        super().__init__(opt=opt, dim=(2, 3), space=[0, 1])

    def fx(self, x):
        return np.einsum('...i,...ij,...j->...', x[..., 0, :], self.A, x[..., 1, :])

    def dfx(self, x):
        grad = np.stack((
            x[..., 1, :] @ self.A.T,
            -x[..., 0, :] @ self.A), axis=-2)
        return grad

    def proj(self, x, grad):
        x = x + grad
        x[0, :] = projections.simplex_proj(x[0, :])
        x[1, :] = projections.simplex_proj(x[1, :])
        return x
