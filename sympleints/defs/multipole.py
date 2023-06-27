import functools

from sympy import pi, sqrt, S

from sympleints import shell_iter
from sympleints.sym_solid_harmonics import mOrder, Rlm_polys_up_to_l_iter
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


def gen_multipole_sph_shell(La_tot, Lb_tot, a, b, A, B, Le_tot=2):
    """Integrals up to spherical quadrupoles.

    The minus is NOT taken into account here, but must be taken into account
    in the code, that utilizes these integrals.

    Different approaches are possible:

    Take one origin argument and calculate all multipoles w.r.t this origin,
    or use a different origin for all primitive pairs, and shift later.
    """
    # Here, the multipole origin is the natural center of the Gaussian overlap
    # distribution and depends on the orbital exponents.
    R = (a * A + b * B) / (a + b)

    """Step 1:

    Generate the necessary regular solid harmonic operators. We do this
    by generating the appropriate regular solid harmonic polynomials.
    They can be converted to sympy.Poly-objects in the basis of x, y and z.

    From these Poly-objects the required coefficients to combine the original
    multipole integrals are easily extracted."""

    # m runs from -l to +l. This is not Stone's ordering, which is 0, +1, -1,
    # +2, -2, ...!
    polys, lms = zip(*Rlm_polys_up_to_l_iter(Le_tot, order=mOrder.NATURAL))
    poly_terms = [poly.terms() for poly in polys]

    # Gather unique multipole-operators from all terms (Le values)
    unique_Le_operators = set()
    L_tot = La_tot + Lb_tot
    for pterms in poly_terms:
        for Le, _ in pterms:
            unique_Le_operators.add(Le)

    Le_exprs = dict()

    lmns = list(shell_iter((La_tot, Lb_tot)))
    # Generate the required basic multipole integrals and store the expressions in
    # a dictionary, with the operator-tuple as key.
    for Le in unique_Le_operators:
        for La, Lb in lmns:
            Le_expr = gen_multipole_3d(La, Lb, a, b, A, B, Le, R)
            Le_exprs.setdefault(Le, list()).append(Le_expr)

    # Combine the basic multipole integrals into final integrals over spherical
    # multipole operators.
    exprs = list()
    all_lmns = list()

    # Loop over all operators that we wan't to generate.
    for pterms in poly_terms:
        # Every spherical multipole integral is the sum of multiple basic multipole integrals.
        # Depending on the angular momenta of the involved basis functions some integrals
        # may be 0, so we initialize all expressions with 0.
        tmp_exprs = [S.Zero] * len(lmns)

        # Don't generate unneccesary expressions, as the product of two primitive
        # Gaussians with angular momenta La and Lb gives rise to multipoles up to order
        # La + Lb at most.
        # So, 2 s-orbitals produce a charge, a s- and a p-orbital give rise to a charge
        # and a dipole moment etc.
        Le0, _ = pterms[0]
        Le0 = sum(Le0)
        if Le0 > L_tot:
            exprs.extend(tmp_exprs)
            all_lmns += lmns
            continue

        # Build up the final expressions. Every operator can comprise multiple basic
        # integrals. Here, we loop over all terms in the polynomial and add the respective
        # basic integrals, multiplied by the appropriate coefficient.
        for Le, coeff in pterms:
            for i, expr in enumerate(Le_exprs[Le]):
                tmp_exprs[i] += coeff * expr
        exprs.extend(tmp_exprs)
        all_lmns += lmns
    return exprs, all_lmns
