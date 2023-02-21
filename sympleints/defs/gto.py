import functools

from sympy import exp, Function

from sympleints import shell_iter


class CartGTO1d(Function):

    """1D Cartesian Gaussian function; not normalized.
    Centered at A, evaluated at R."""

    @classmethod
    @functools.cache
    def eval(cls, i, ax, A, R):
        RA = R - A
        return RA**i * exp(-ax * RA**2)


def gen_gto_3d(La, ax, A, R):
    x, y, z = [CartGTO1d(La[i], ax, A[i], R[i]) for i in range(3)]
    return x * y * z


def gen_gto3d_shell(La_tot, ax, A, R):
    lmns = list(shell_iter((La_tot,)))
    exprs = [gen_gto_3d(La, ax, A, R) for La, in lmns]
    return exprs, lmns
