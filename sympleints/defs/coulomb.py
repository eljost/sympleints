import functools

import numpy as np
from sympy import exp, Function, pi, sqrt

from sympleints import shell_iter
from sympleints.defs import TwoCenter1d

# Placeholder for the Boys-function. The actual Boys-function will be
# imported in the generated python module.
boys = Function("boys")


class Coulomb(TwoCenter1d):
    """Nucleus at C."""

    @functools.cache
    def eval(self, i, k, m, j, l, n, N):
        ang_moms = (i, k, m, j, l, n)
        if any([am < 0 for am in ang_moms]):
            return 0

        def recur(N, *inds):
            """Simple wrapper to pass all required arguments."""
            return self.eval(*inds, N)  # , a, b, A, B, C)

        def decr(to_decr, decr_ind):
            one = np.zeros(3, dtype=int)
            one[decr_ind] = 1

            def Ni(inds):
                return inds[decr_ind]

            bra = np.array((i, k, m), dtype=int)
            ket = np.array((j, l, n), dtype=int)

            if to_decr == "bra":
                X = self.PA
                bra[decr_ind] -= 1
            else:
                X = self.PB
                ket[decr_ind] -= 1

            bra_decr = bra - one
            ket_decr = ket - one

            return (
                X[decr_ind] * recur(N, *bra, *ket)
                - self.PC[decr_ind] * recur(N + 1, *bra, *ket)
                + 1
                / (2 * self.p)
                * Ni(bra)
                * (recur(N, *bra_decr, *ket) - recur(N + 1, *bra_decr, *ket))
                + 1
                / (2 * self.p)
                * Ni(ket)
                * (recur(N, *bra, *ket_decr) - recur(N + 1, *bra, *ket_decr))
            )

        # Base case
        if all([am == 0 for am in ang_moms]):
            K = exp(-self.mu * self.AB.dot(self.AB))
            return 2 * pi / self.p * K * boys(N, self.p * self.PC.dot(self.PC))
        elif i > 0:
            return decr("bra", 0)
        elif j > 0:
            return decr("ket", 0)
        elif k > 0:
            return decr("bra", 1)
        elif l > 0:
            return decr("ket", 1)
        elif m > 0:
            return decr("bra", 2)
        elif n > 0:
            return decr("ket", 2)


class CoulombShell(Function):
    @classmethod
    def eval(cls, La_tot, Lb_tot, a, b, A, B, C=(0.0, 0.0, 0.0)):
        exprs = [
            # Coulomb(*La, *Lb, 0, a, b, A, B, C)
            Coulomb(a, A, b, B, C).eval(*La, *Lb, 0)
            for La, Lb in shell_iter((La_tot, Lb_tot))
        ]
        # print(Coulomb.eval.cache_info())
        return exprs


class TwoCenterTwoElectron(Function):
    @classmethod
    @functools.cache
    def eval(cls, ia, ja, ka, ib, jb, kb, N, a, b, A, B):
        ang_moms = np.array((ia, ja, ka, ib, jb, kb), dtype=int)
        ang_moms2d = ang_moms.reshape(-1, 3)
        if any([am < 0 for am in ang_moms]):
            return 0

        p = a + b
        P = (a * A + b * B) / p
        mu = (a * b) / p

        def recur(N, *inds):
            return cls(*inds, N, a, b, A, B)

        def vrr(bra_or_ket, cart_ind):
            assert bra_or_ket in ("bra", "ket")

            if bra_or_ket == "bra":
                ind1 = 0
                ind2 = 1
            else:
                ind1 = 1
                ind2 = 0
            exps = (a, b)
            X = (P - A, P - B)[ind1][cart_ind]

            l1 = ang_moms2d[ind1, cart_ind] - 1
            exp1 = exps[ind1]
            decr1 = ang_moms2d.copy()
            decr1[ind1, cart_ind] -= 1
            decr11 = decr1.copy()
            decr11[ind1, cart_ind] -= 1

            l2 = ang_moms2d[ind2, cart_ind]
            exp2 = exps[ind2]
            decr12 = ang_moms2d.copy()
            decr12[ind1, cart_ind] -= 1
            decr12[ind2, cart_ind] -= 1

            decr1 = decr1.flatten()
            decr11 = decr11.flatten()
            decr12 = decr12.flatten()

            return (
                X * recur(N + 1, *decr1)
                + l1
                / (2 * exp1)
                * (recur(N, *decr11) - exp2 / p * recur(N + 1, *decr11))
                + l2 / (2 * p) * recur(N + 1, *decr12)
            )

        vrr_bra = functools.partial(vrr, "bra")
        vrr_ket = functools.partial(vrr, "ket")

        # Base case
        if (ang_moms == 0).all():
            AB = A - B
            return 2 * pi**2.5 / sqrt(p) / (a * b) * boys(N, mu * AB.dot(AB))
        elif ia > 0:
            return vrr_bra(0)
        elif ja > 0:
            return vrr_bra(1)
        elif ka > 0:
            return vrr_bra(2)
        elif ib > 0:
            return vrr_ket(0)
        elif jb > 0:
            return vrr_ket(1)
        elif kb > 0:
            return vrr_ket(2)


class TwoCenterTwoElectronShell(Function):
    @classmethod
    def eval(cls, La_tot, Lb_tot, a, b, A, B):
        exprs = [
            TwoCenterTwoElectron(*La, *Lb, 0, a, b, A, B)
            for La, Lb in shell_iter((La_tot, Lb_tot))
        ]
        # print(TwoCenterTwoElectron.eval.cache_info())
        return exprs


class ThreeCenterTwoElectronBase(Function):
    """
    https://pubs.rsc.org/en/content/articlelanding/2004/CP/b413539c

    There is an error in the base case (00|0). One must divide by
    sqrt(eta + gamma), not multiply.
    """

    @classmethod
    @functools.cache
    def eval(cls, ia, ja, ka, ib, jb, kb, ic, jc, kc, N, a, b, c, A, B, C):
        ang_moms = np.array((ia, ja, ka, ib, jb, kb, ic, jc, kc), dtype=int)
        ang_moms2d = ang_moms.reshape(-1, 3)
        if any([am < 0 for am in ang_moms]):
            return 0

        p = a + b
        P = (a * A + b * B) / p
        mu = (a * b) / p
        X_PC = P - C

        rho = p * c / (p + c)

        def recur(N, *inds):
            """Simple wrapper to pass all required arguments.

            Here we don't use ThreeCenterTwoElectronBase, but the derived classes,
            that provide the 'aux_vrr' attribute, to distinguish between the different
            vertical recursion relations."""
            return cls(*inds, N, a, b, c, A, B, C)

        def recur_hrr(cart_ind):
            """Horizontal recursion relation to transfer angular momentum in bra.

            (a, b+1_i|c) = (a + 1_i,b|c) + X_AB (ab|c)
            """
            assert N == 0
            incr_ang_moms = ang_moms2d.copy()
            incr_ang_moms[0, cart_ind] += 1  # Increase in left orbital
            incr_ang_moms[1, cart_ind] -= 1  # Decrease in right orbital

            decr_ang_moms = ang_moms2d.copy()
            decr_ang_moms[1, cart_ind] -= 1  # Decrease in right orbital

            incr_ang_moms = incr_ang_moms.flatten()
            decr_ang_moms = decr_ang_moms.flatten()

            AB_dir = (A - B)[cart_ind]

            return recur(N, *incr_ang_moms) + AB_dir * recur(N, *decr_ang_moms)

        def recur_vrr(cart_ind):
            assert (ib, jb, kb) == (0, 0, 0)
            assert (ic, jc, kc) == (0, 0, 0)

            decr_a = ang_moms2d.copy()
            decr_a[0, cart_ind] -= 1  # Decrease in bra left orbital
            decr_aa = decr_a.copy()
            decr_aa[0, cart_ind] -= 1  # Decrease in bra left orbital again

            PA_dir = (P - A)[cart_ind]
            PC_dir = (P - C)[cart_ind]
            ai = (ia, ja, ka)[cart_ind] - 1
            _2p = 2 * p

            decr_a = decr_a.flatten()
            decr_aa = decr_aa.flatten()

            return (
                PA_dir * recur(N, *decr_a)
                - rho / p * PC_dir * recur(N + 1, *decr_a)
                + ai / _2p * (recur(N, *decr_aa) - rho / p * recur(N + 1, *decr_aa))
            )

        def recur_vrr_aux(cart_ind):
            decr_c = ang_moms2d.copy()
            decr_c[2, cart_ind] -= 1

            decr_cc = decr_c.copy()
            decr_cc[2, cart_ind] -= 1

            decr_ac = decr_c.copy()
            decr_ac[0, cart_ind] -= 1

            decr_bc = decr_c.copy()
            decr_bc[1, cart_ind] -= 1

            PC_dir = (P - C)[cart_ind]
            la = (ia, ja, ka)[cart_ind]
            lb = (ib, jb, kb)[cart_ind]
            lc = (ic, jc, kc)[cart_ind] - 1

            decr_c = decr_c.flatten()
            decr_cc = decr_cc.flatten()
            decr_ac = decr_ac.flatten()
            decr_bc = decr_bc.flatten()

            return (
                p / (p + c) * PC_dir * recur(N + 1, *decr_c)
                + lc
                / (2 * c)
                * (recur(N, *decr_cc) - p / (p + c) * recur(N + 1, *decr_cc))
                + la / (2 * (p + c)) * recur(N + 1, *decr_ac)
                + lb / (2 * (p + c)) * recur(N + 1, *decr_bc)
            )

        def recur_vrr_aux_sph(cart_ind):
            assert (ib, jb, kb) == (0, 0, 0)
            decr_c = ang_moms2d.copy()
            decr_c[2, cart_ind] -= 1
            decr_ac = decr_c.copy()
            decr_ac[0, cart_ind] -= 1
            La = (ia, ja, ka)[cart_ind]
            PC_dir = (P - C)[cart_ind]
            return (
                rho
                / c
                * (
                    PC_dir * recur(N + 1, *decr_c.flatten())
                    + La / (2 * p) * recur(N + 1, *decr_ac.flatten())
                )
            )

        recur_vrr_aux_funcs = {
            "cart": recur_vrr_aux,
            "sph": recur_vrr_aux_sph,
        }
        recur_vrr_aux_func = recur_vrr_aux_funcs[cls.aux_vrr]

        # Base case
        if (ang_moms == 0).all():
            X_AB = A - B
            r2_PC = X_PC.dot(X_PC)
            r2_AB = X_AB.dot(X_AB)
            chi = rho * r2_PC
            K = exp(-mu * r2_AB)
            return 2 * pi**2.5 / sqrt(p + c) / (p * c) * K * boys(N, chi)
        elif ib > 0:
            return recur_hrr(0)
        elif jb > 0:
            return recur_hrr(1)
        elif kb > 0:
            return recur_hrr(2)
        elif ic > 0:
            return recur_vrr_aux_func(0)
        elif jc > 0:
            return recur_vrr_aux_func(1)
        elif kc > 0:
            return recur_vrr_aux_func(2)
        elif ia > 0:
            return recur_vrr(0)
        elif ja > 0:
            return recur_vrr(1)
        elif ka > 0:
            return recur_vrr(2)


class ThreeCenterTwoElectron(ThreeCenterTwoElectronBase):
    aux_vrr = "cart"


class ThreeCenterTwoElectronSph(ThreeCenterTwoElectronBase):
    aux_vrr = "sph"


class ThreeCenterTwoElectronShell(Function):
    @classmethod
    def eval(cls, La_tot, Lb_tot, Lc_tot, a, b, c, A, B, C):
        exprs = [
            ThreeCenterTwoElectron(*La, *Lb, *Lc, 0, a, b, c, A, B, C)
            for La, Lb, Lc in shell_iter((La_tot, Lb_tot, Lc_tot))
        ]
        # print(ThreeCenterTwoElectron.eval.cache_info())
        return exprs


class ThreeCenterTwoElectronSphShell(Function):
    @classmethod
    def eval(cls, La_tot, Lb_tot, Lc_tot, a, b, c, A, B, C):
        exprs = [
            ThreeCenterTwoElectronSph(*La, *Lb, *Lc, 0, a, b, c, A, B, C)
            for La, Lb, Lc in shell_iter((La_tot, Lb_tot, Lc_tot))
        ]
        # print(ThreeCenterTwoElectron.eval.cache_info())
        return exprs
