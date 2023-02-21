import functools

from sympy import pi, sqrt

from sympleints import shell_iter
from sympleints.defs import TwoCenter1d


class Multipole1d(TwoCenter1d):
    """1d multipole-moment integral of order 'e', between primitive 1d Gaussians
    Ga = G_i(a, r, A) and Gb = G_j(b, r, B) with Cartesian quantum number i and j,
    exponents a and b, centered at A (B). The origin of the multipole expansion is
    at R.
    """

    @functools.cache
    def __call__(self, i, j, e):
        ang_moms = (i, j, e)
        if any([_ < 0 for _ in ang_moms]):
            return 0

        recur = self

        def vrr(i, j, e, X):
            return X * recur(i, j, e) + 1 / (2 * self.p) * (
                i * recur(i - 1, j, e) + j * recur(i, j - 1, e) + e * recur(i, j, e - 1)
            )

        # Base case
        if all([_ == 0 for _ in ang_moms]):
            return sqrt(pi / self.p) * self.K
        # Decrement i
        elif i > 0:
            return vrr(i - 1, j, e, self.PA)
        # Decrement j
        elif j > 0:
            return vrr(i, j - 1, e, self.PB)
        elif e > 0:
            return vrr(i, j, e - 1, self.PR)


def gen_multipole_3d(La, Lb, a, b, A, B, Le, R):
    x, y, z = [
        Multipole1d(a, A[i], b, B[i], R[i])(La[i], Lb[i], Le[i]) for i in range(3)
    ]
    return x * y * z


def gen_multipole_shell(La_tot, Lb_tot, a, b, A, B, Le_tot=0, R=(0.0, 0.0, 0.0)):
    lmns = list(shell_iter((Le_tot, La_tot, Lb_tot)))
    exprs = [gen_multipole_3d(La, Lb, a, b, A, B, Le, R) for Le, La, Lb in lmns]
    # Drop Le from angular momenta, only return (La, Lb) tuples.
    lmns = [(La, Lb) for Le, La, Lb in lmns]
    return exprs, lmns


def gen_diag_quadrupole_shell(La_tot, Lb_tot, a, b, A, B, R=(0.0, 0.0, 0.0)):
    exprs = list()
    lmns = list(shell_iter((La_tot, Lb_tot)))
    for Le in ((2, 0, 0), (0, 2, 0), (0, 0, 2)):
        for La, Lb in lmns:
            exprs.append(gen_multipole_3d(La, Lb, a, b, A, B, Le, R))
    lmns = lmns + lmns + lmns
    return exprs, lmns
