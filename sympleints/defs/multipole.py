import functools

from sympy import pi, sqrt

from sympleints import shell_iter
from sympleints.defs import TwoCenter1d, RecurStrategy, Strategy


class Multipole1d(TwoCenter1d):
    """1d multipole-moment integral of order 'e', between primitive 1d Gaussians
    Ga = G_i(a, r, A) and Gb = G_j(b, r, B) with Cartesian quantum number i and j,
    exponents a and b, centered at A (B). The origin of the multipole expansion is
    at R.
    """

    @functools.cache
    def __call__(self, i, j, e):
        def base_case():
            return sqrt(pi / self.p) * self.K

        def vrr(i, j, e, X):
            return X * self(i, j, e) + 1 / (2 * self.p) * (
                i * self(i - 1, j, e) + j * self(i, j - 1, e) + e * self(i, j, e - 1)
            )

        vrr_bra = functools.partial(vrr, X=self.PA)
        vrr_ket = functools.partial(vrr, X=self.PB)
        vrr_e = functools.partial(vrr, X=self.PR)

        strat = RecurStrategy(base_case, (vrr_bra, vrr_ket, vrr_e), Strategy.HIGHEST)
        return strat.recur(i, j, e)


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
