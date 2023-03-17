import numpy as np

from utils import norm


def default_dual_local_norm(v, x):
    return norm(v)


def adaprox(x0, df, proj, n, dual_local_norm=default_dual_local_norm, ergodic=True, **_):
    delta_squared_sum = 0
    eta_sum = 0
    weighted_x_t05_sum = 0

    eta_t = 1
    x_t = x0

    yield x0
    for _ in range(n):
        # t+1/2
        v_t = df(x_t)
        x_t05 = proj(x_t, -eta_t*v_t)

        # t+1
        v_t05 = df(x_t05)
        x_t = proj(x_t, -eta_t*v_t05)

        # ergodic average
        eta_sum += eta_t
        weighted_x_t05_sum += eta_t * x_t05

        if ergodic:
            yield weighted_x_t05_sum / eta_sum
        else:
            yield x_t05

        # adaptive step size
        d_t = dual_local_norm(v_t05 - v_t, x_t05)
        delta_squared_sum += d_t**2
        eta_t = 1 / np.sqrt(1 + delta_squared_sum)
