def extra_gradient(x0, df, proj, n, eg_eta=1, ergodic=True, **_):
    eta = eg_eta
    eta_sum = 0
    weighted_x_t05_sum = 0

    x_t = x0

    yield x0
    for i in range(n):
        # t+1/2
        x_t05 = proj(x_t, -eta*df(x_t))

        # t+1
        x_t = proj(x_t, -eta*df(x_t05))

        # ergodic average
        eta_sum += eta
        weighted_x_t05_sum += eta * x_t05

        if ergodic:
            yield weighted_x_t05_sum / eta_sum
        else:
            yield x_t05
