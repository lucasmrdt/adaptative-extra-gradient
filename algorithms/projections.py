import numpy as np


def simplex_proj(y):
    # from https://arxiv.org/pdf/1309.1541.pdf
    u = -np.sort(-y)
    j = np.arange(1, len(y)+1)
    cumsum_u = np.cumsum(u)
    rho = y + 1/j * (1 - cumsum_u)
    rho = np.max(np.where(rho > 0)[0])
    lbd = 1/(rho+1) * (1 - cumsum_u[rho])
    return np.maximum(y + lbd, 0)
