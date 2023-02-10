import itertools as it
import sys
import time

from colorama import Fore, Style
from sympy import IndexedBase, Matrix, Symbol


RST = Style.RESET_ALL  # colorama reset


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


def shell_shape_iter(Ls, ncomponents: int = 0, start_at: int = 0, **kwargs):
    shape = shell_shape(Ls, **kwargs)
    if ncomponents > 0:
        shape = (ncomponents, *shape)
    ranges = [range(start_at, s + start_at) for s in shape]
    iter_ = it.product(*ranges)
    return iter_


def func_name_from_Ls(name, Ls):
    return name + "_" + "".join(str(l) for l in Ls)


def get_center(i):
    symbs = [Symbol(str(i) + ind, real=True) for ind in ("x", "y", "z")]
    return Matrix([*symbs]).T  # Return column vector


def get_map(i, center_i):
    array = IndexedBase(i, shape=3)
    array_map = dict(zip(center_i, array))
    return array, array_map


class Timer:
    def __init__(self, msg, prefix="", width=0, thresh=1.0):
        self.msg = msg
        self.prefix = prefix
        self.width = width
        self.thresh = thresh

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        dur = time.time() - self.start
        color = Fore.RED if dur > self.thresh else ""
        print(f"{color}{self.prefix}{self.msg:>{self.width}s} took {dur:> 8.3f} s" + RST)
        sys.stdout.flush()


def get_timer_getter(prefix, width=40):
    def get_timer(*args):
        return Timer(*args, prefix=prefix, width=width)
    return get_timer
