def norm(x, start_idx=0):
    # print('norm', x.shape, start_idx)
    if start_idx < 0:
        start_idx = x.ndim + start_idx
    return (x**2).sum(axis=tuple(range(start_idx, x.ndim)))**0.5
