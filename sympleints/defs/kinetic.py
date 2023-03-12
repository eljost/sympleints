import functools

from sympleints import shell_iter
from sympleints.defs import Multipole1d, RecurStrategy, Strategy, TwoCenter1d


class Overlap1d(Multipole1d):
    """Required for kinetic energy integrals."""

    @functools.cache
    def __call__(self, i, j):
        return Multipole1d(self.ax, self.A, self.bx, self.B)(i, j, 0)


class Kinetic1d(TwoCenter1d):
    @functools.cache
    def __call__(self, i, j):
        def recur_rel(i, j, X):
            return X * self(i, j) + 1 / (2 * self.p) * (
                i * self(i - 1, j) + j * self(i, j - 1)
            )

        def recur_ovlp(i, j):
            return Overlap1d(self.ax, self.A, self.bx, self.B)(i, j)

        def base_case():
            return (
                self.ax - 2 * self.ax**2 * (self.PA**2 + 1 / (2 * self.p))
            ) * recur_ovlp(0, 0)

        def vrr_bra(i, j):
            # Eq. (9.3.41)
            return recur_rel(i, j, self.PA) + self.bx / self.p * (
                2 * self.ax * recur_ovlp(i + 1, j) - (i) * recur_ovlp(i - 1, j)
            )

        def vrr_ket(i, j):
            # Eq. (9.3.42)
            return recur_rel(i, j, self.PB) + self.ax / self.p * (
                2 * self.bx * recur_ovlp(i, j + 1) - (j) * recur_ovlp(i, j - 1)
            )

        strategy = RecurStrategy(base_case, (vrr_bra, vrr_ket), Strategy.HIGHEST)
        return strategy.recur(i, j)


def gen_kinetic_3d(La, Lb, a, b, A, B):
    Tx, Ty, Tz = [Kinetic1d(a, A[i], b, B[i])(La[i], Lb[i]) for i in range(3)]
    Sx, Sy, Sz = [Overlap1d(a, A[i], b, B[i])(La[i], Lb[i]) for i in range(3)]
    return (Tx * Sy * Sz) + (Sx * Ty * Sz) + (Sx * Sy * Tz)


def gen_kinetic_shell(La_tot, Lb_tot, a, b, A, B):
    lmns = list(shell_iter((La_tot, Lb_tot)))
    exprs = [gen_kinetic_3d(La, Lb, a, b, A, B) for La, Lb in lmns]
    return exprs, lmns
