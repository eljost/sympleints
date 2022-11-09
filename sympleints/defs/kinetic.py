import functools

from sympleints import shell_iter
from sympleints.defs import Multipole1d, TwoCenter1d


class Overlap1d(Multipole1d):
    """Required for kinetic energy integrals."""

    @functools.cache
    def __call__(self, i, j):
        return Multipole1d(self.a, self.A, self.b, self.B)(i, j, 0)


class Kinetic1d(TwoCenter1d):
    @functools.cache
    def __call__(self, i, j):
        if i < 0 or j < 0:
            return 0

        def recur_rel(i, j, X):
            return X * self(i, j) + 1 / (2 * self.p) * (
                i * self(i - 1, j) + j * self(i, j - 1)
            )

        def recur_ovlp(i, j):
            return Overlap1d(self.a, self.A, self.b, self.B)(i, j)

        # Base case
        if i == 0 and j == 0:
            return (
                self.a - 2 * self.a**2 * (self.PA**2 + 1 / (2 * self.p))
            ) * recur_ovlp(i, j)
        # Decrement i
        elif i > 0:
            # Eq. (9.3.41)
            return recur_rel(i - 1, j, self.PA) + self.b / self.p * (
                2 * self.a * recur_ovlp(i, j) - i * recur_ovlp(i - 2, j)
            )
        # Decrement j
        elif j > 0:
            # Eq. (9.3.41)
            return recur_rel(i, j - 1, self.PB) + self.a / self.p * (
                2 * self.b * recur_ovlp(i, j) - j * recur_ovlp(i, j - 2)
            )


def gen_kinetic_3d(La, Lb, a, b, A, B):
    Tx, Ty, Tz = [Kinetic1d(a, A[i], b, B[i])(La[i], Lb[i]) for i in range(3)]
    Sx, Sy, Sz = [Overlap1d(a, A[i], b, B[i])(La[i], Lb[i]) for i in range(3)]
    return Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz


def gen_kinetic_shell(La_tot, Lb_tot, a, b, A, B):
    exprs = [
        gen_kinetic_3d(La, Lb, a, b, A, B) for La, Lb in shell_iter((La_tot, Lb_tot))
    ]
    return exprs
