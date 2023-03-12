import enum  # from enum import Enum
import functools
import itertools as it
import sys
import time

from colorama import Fore, Style
from sympy import IndexedBase, Matrix, Symbol

from sympleints.config import L_MAX, L_AUX_MAX


L_MAP = {
    0: "s",
    1: "p",
    2: "d",
    3: "f",
    4: "g",
    5: "h",
    6: "i",
    7: "j",
    8: "k",
}


@functools.total_ordering
class BFKind(enum.Enum):
    CART = enum.auto()
    SPH = enum.auto()

    def __lt__(self, other):
        return self.value < other.value


RST = Style.RESET_ALL  # colorama reset


def canonical_order(L):
    inds = list()
    for i in range(L + 1):
        l = L - i
        for n in range(i + 1):
            m = i - n
            inds.append((l, m, n))
    return inds


def sph_order(L):
    # m from -L, -L + 1, ..., 0, 1, ..., L
    return [(L, m) for m in range(-L, L + 1)]


def get_order_funcs_for_kinds(kinds):
    func_map = {
        BFKind.CART: canonical_order,
        BFKind.SPH: sph_order,
    }
    return [func_map[kind] for kind in kinds]


def cart_label(angmoms):
    assert len(angmoms) == 3
    L = L_MAP[sum(angmoms)]
    cart_inds = it.chain(*[[c] * l for l, c in zip(angmoms, ("x", "y", "z")) if l > 0])
    label = str(L) + "".join(map(str, cart_inds))
    return label


def shell_iter(Ls, kinds=None):
    """Iterator over angular momenta/quantum number products."""
    if kinds is None:
        kinds = [BFKind.CART] * len(Ls)
    assert len(kinds) == len(Ls)
    funcs = get_order_funcs_for_kinds(kinds)
    return it.product(*[func(L) for func, L in zip(funcs, Ls)])


def shell_shape(Ls, ncomponents: int = 0, cartesian=True):
    if cartesian:
        size = lambda L: (L + 2) * (L + 1) // 2
    else:  # Spherical
        size = lambda L: 2 * L + 1
    shape = tuple(map(size, Ls))
    if ncomponents > 0:
        shape = (ncomponents,) + shape
    return shape


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
    def __init__(self, msg, prefix="", width=0, thresh=1.0, logger=None):
        self.msg = msg
        self.prefix = prefix
        self.width = width
        self.thresh = thresh
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        dur = time.time() - self.start
        color = Fore.RED if dur > self.thresh else ""
        msg = f"{color}{self.prefix}{self.msg:>{self.width}s} took {dur:> 8.3f} s" + RST
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)
        sys.stdout.flush()


def get_timer_getter(**kwargs):
    def get_timer(*args):
        return Timer(*args, **kwargs)

    return get_timer
