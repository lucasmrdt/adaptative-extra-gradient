import numpy as np
from utils import norm


def universal_mirror_prox(x0, df, proj, n, ergodic=True, **_):
    D = np.sqrt(2)
    G0 = 2

    x_t_sum = 0
    Z_t_sum = 0

    y_t = x0
    yield x0
    for i in range(n):
        y_t_prev = y_t
        eta_t = D / np.sqrt(G0 + Z_t_sum)

        Mt = df(y_t)

        x_t = proj(y_t_prev, -eta_t*Mt)

        gt = df(x_t)

        y_t = proj(y_t_prev, -eta_t*gt)

        Z_t = (norm(x_t - y_t)**2 + norm(x_t - y_t_prev)**2) / (5*eta_t**2)
        Z_t_sum += Z_t
        x_t_sum += x_t

        if ergodic:
            yield 1/(i+1) * x_t_sum
        else:
            yield x_t
