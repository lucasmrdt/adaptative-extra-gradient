import numpy as np

from .problem import Problem


class BilinearPb(Problem):
    def __init__(self, opt=np.array([0, 0]), space=[-1, 1]):
        self.opt = opt
        super().__init__(opt=opt, dim=(2,), space=space)

    def fx(self, x):
        return (x[..., 0] - self.opt[0]) * (x[..., 1] - self.opt[1])

    def dfx(self, x):
        grad = np.stack((
            x[..., 1] - self.opt[1],
            -x[..., 0] + self.opt[0]), axis=-1)
        return grad

    def proj(self, x, grad):
        return np.clip(x + grad, *self.space)
