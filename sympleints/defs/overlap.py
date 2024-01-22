import functools

import sympy as sym

from sympleints import shell_iter
from sympleints.defs.multipole import gen_multipole_shell
from sympleints.symbols import center_rP, center_rP2


def gen_overlap_shell(La_tot, Lb_tot, a, b, A, B):
    """Overlap integral between two shells."""
    exprs, lmns = gen_multipole_shell(La_tot, Lb_tot, a, b, A, B)
    return exprs, lmns


class Prefactor:
    rP = sym.Matrix(center_rP).reshape(1, 3)
    rP2 = sym.Matrix(center_rP2).reshape(1, 3)

    @functools.cache
    def __call__(self, i, k, m, j, l, n):
        rPx, rPy, rPz = self.rP
        rPx2, rPy2, rPz2 = self.rP2
        expr = rPx ** (i + j) * rPy ** (k + l) * rPz ** (m + n)
        # Substitute squares in, as these are also available
        expr = expr.subs(rPx**2, rPx2).subs(rPy**2, rPy2).subs(rPz**2, rPz2)
        return expr


def gen_prefactor_shell(La_tot, Lb_tot):
    lmns = list(shell_iter((La_tot, Lb_tot)))
    pref = Prefactor()
    exprs = [pref(*La, *Lb) for La, Lb in lmns]
    return exprs, lmns
