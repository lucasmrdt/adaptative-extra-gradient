import numpy as np

from .problem import Problem


class BattleOfTheSexesPb(Problem):
    def __init__(self):
        self.P1_A = np.array([
            # foot   #theatre
            [2,     0],   # foot
            [0,     1],  # theatre
        ])
        self.P2_A = np.array([
            # foot   #theatre
            [1,     0],   # foot
            [0,     2],  # theatre
        ])

        opt = np.array([
            [0, 0],  # foot
            [1, 1],  # theatre
            [2/3, 1/2],  # mixed
        ])
        super().__init__(dim=(2,), space=[0, 1], opt=opt)

    def fx(self, x):  # show only for P1 for now
        x_p1 = np.stack([x[..., 0], 1 - x[..., 0]], axis=-1)
        x_p2 = np.stack([x[..., 1], 1 - x[..., 1]], axis=-1)
        return np.einsum('...i,...ij,...j->...', x_p1, self.P1_A, x_p2)

    def dfx(self, x):
        x_p1 = np.stack([x[..., 0], 1 - x[..., 0]], axis=-1)
        x_p2 = np.stack([x[..., 1], 1 - x[..., 1]], axis=-1)
        g_p1 = np.einsum('...i,...ij,...j->...',
                         np.array([1, -1]), self.P1_A, x_p2)
        g_p2 = np.einsum('...i,...ij,...j->...',
                         x_p1, self.P2_A, np.array([1, -1]))
        grad = np.stack((-g_p1, -g_p2), axis=-1)
        return grad

    def proj(self, x, grad):
        return np.clip(x + grad, *self.space)

    def get_displayed_x1(self, x):
        return x[..., 0]

    def get_displayed_x2(self, x):
        return x[..., 1]
