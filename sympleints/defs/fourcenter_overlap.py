import functools

from sympy import exp, pi, sqrt

from sympleints import shell_iter
from sympleints.defs import FourCenter1d


class FourCenterOverlap1d(FourCenter1d):
    @functools.cache
    def __call__(self, i, j, k, l):
        ang_moms = (i, j, k, l)
        if any([_ < 0 for _ in ang_moms]):
            return 0

        recur = self

        def vrr(i, j, k, l, X):
            return X * recur(i, j, k, l) + 1 / (2 * self.g) * (
                i * recur(i - 1, j, k, l)
                + j * recur(i, j - 1, k, l)
                + k * recur(i, j, k - 1, l)
                + l * recur(i, j, k, l - 1)
            )

        # Base case
        if all([_ == 0 for _ in ang_moms]):
            g = self.g
            PQ = self.P - self.Q
            red_exp = self.p * self.q / g
            return sqrt(pi / g) * exp(-red_exp * PQ**2)
        # Decrement i
        elif i > 0:
            return vrr(i - 1, j, k, l, self.GA)
        # Decrement j
        elif j > 0:
            return vrr(i, j - 1, k, l, self.GB)
        elif k > 0:
            return vrr(i, j, k - 1, l, self.GC)
        elif l > 0:
            return vrr(i, j, k, l - 1, self.GD)


def gen_fourcenter_overlap_3d(La, Lb, Lc, Ld, a, b, c, d, A, B, C, D):
    x, y, z = [
        FourCenterOverlap1d(a, A[i], b, B[i], c, C[i], d, D[i])(
            La[i], Lb[i], Lc[i], Ld[i]
        )
        for i in range(3)
    ]
    return x * y * z


def gen_fourcenter_overlap_shell(
    La_tot, Lb_tot, Lc_tot, Ld_tot, a, b, c, d, A, B, C, D
):
    exprs = [
        gen_fourcenter_overlap_3d(La, Lb, Lc, Ld, a, b, c, d, A, B, C, D)
        for La, Lb, Lc, Ld in shell_iter((La_tot, Lb_tot, Lc_tot, Ld_tot))
    ]
    return exprs
