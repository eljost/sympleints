import itertools as it


def canonical_order(L):
    inds = list()
    for i in range(L + 1):
        l = L - i
        for n in range(i + 1):
            m = i - n
            inds.append((l, m, n))
    return inds


def shell_iter(Ls):
    """Iterator over cartesian product of L values in Ls."""
    return it.product(*[canonical_order(L) for L in Ls])


def shell_shape(Ls, cartesian=True):
    if cartesian:
        size = lambda L: (L + 2) * (L + 1) // 2
    else:  # Spherical
        size = lambda L: 2 * L + 1
    return tuple(map(size, Ls))


def shell_shape_iter(Ls, **kwargs):
    shape = shell_shape(Ls, **kwargs)
    ranges = [range(s) for s in shape]
    return it.product(*ranges)
