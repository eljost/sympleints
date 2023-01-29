#!/usr/bin/env python3

# [1] https://doi.org/10.1063/1.450106
#     Efficient recursive computation of molecular integrals over Cartesian
#     Gaussian functions
#     Obara, Saika, 1986
# [2] https://doi.org/10.1002/9781119019572
#     Molecular Electronic-Structure Theory
#     Helgaker, Jørgensen, Olsen
# [3] https://doi.org/10.1021/acs.jctc.7b00788
#     LIBRETA: Computerized Optimization and Code Synthesis for
#     Electron Repulsion Integral Evaluation
#     Jun Zhang
# [4] https://doi.org/10.1039/B413539C
#     Efficient evaluation of three-center two-electron integrals over Gaussian functions
#     Ahlrichs, 2004
# [5] EVALUATING MANY-ELECTRON MOLECULAR INTEGRALS FOR QUANTUM CHEMISTRY
#     James Christopher Womack, PhD Thesis
# [6] https://doi.org/10.1063/1.4983393
#     Efficient evaluation of three-center Coulomb integrals
#     Samu, Kállay, 2017
# [7] https://arxiv.org/pdf/2210.03192.pdf
#     Memory-Efficient Recursive Evaluation of 3-Center Gaussian Integrals
#     Asadchev, Valeev, 2022


import argparse

from datetime import datetime
import functools
import itertools as it
from math import prod
import os
from pathlib import Path
import sys
import time

from sympy import (
    Array,
    cse,
    factorial2,
    flatten,
    IndexedBase,
    Matrix,
    permutedims,
    pi,
    simplify,
    sqrt,
    Symbol,
    symbols,
    tensorcontraction as tc,
    tensorproduct as tp,
)

from sympleints.config import L_MAX, L_AUX_MAX, L_MAP
from sympleints.defs.coulomb import (
    CoulombShell,
    TwoCenterTwoElectronShell,
    ThreeCenterTwoElectronShell,
    ThreeCenterTwoElectronSphShell,
)
from sympleints import canonical_order, shell_iter
from sympleints.defs.fourcenter_overlap import gen_fourcenter_overlap_shell
from sympleints.defs.gto import CartGTOShell
from sympleints.defs.kinetic import gen_kinetic_shell
from sympleints.defs.multipole import gen_diag_quadrupole_shell, gen_multipole_shell
from sympleints.defs.overlap import gen_overlap_shell
from sympleints.render import write_render

try:
    from pysisyphus.wavefunction.cart2sph import cart2sph_coeffs
except ModuleNotFoundError:
    print("pysisyphus is not installed. Disabling Cartesian->spherical transformation.")

KEYS = (
    "cgto",
    "ovlp",
    "dpm",
    "dqpm",
    "qpm",
    "kin",
    "coul",
    "3c2e",
    "3c2e_sph",
)
ONE_THRESH = 1e-14


def get_center(i):
    symbs = [Symbol(str(i) + ind, real=True) for ind in ("x", "y", "z")]
    return Matrix([*symbs]).T  # Return column vector


def get_map(i, center_i):
    array = IndexedBase(i, shape=3)
    array_map = dict(zip(center_i, array))
    return array, array_map


def cart2spherical(L_tots, exprs):
    assert len(L_tots) > 0

    # Coefficient matrices for Cartesian-to-spherical conversion
    coeffs = [Array(CART2SPH[L]) for L in L_tots]
    cart_shape = [(l + 1) * (l + 2) // 2 for l in L_tots]
    cart = Array(exprs).reshape(*cart_shape)

    sph = tc(tp(coeffs[0], cart), (1, 2))
    if len(L_tots) == 1:
        pass
    elif len(L_tots) == 2:
        sph = tc(tp(sph, coeffs[1].transpose()), (1, 2))
    elif len(L_tots) == 3:
        _, Cb, Cc = coeffs
        sph = tc(tp(permutedims(sph, (0, 2, 1)), Cb.transpose()), (2, 3))
        sph = tc(tp(permutedims(sph, (0, 2, 1)), Cc.transpose()), (2, 3))
    else:
        raise Exception(
            "Cartesian -> spherical transformation for 4-center integrals "
            "is not implemented!"
        )

    # Cartesian-to-spherical transformation introduces quite a number of
    # multiplications by 1.0, which are uneccessary. Here, we try to drop
    # some of them by replacing numbers very close to +1.0 with 1.
    sph = sph.replace(lambda n: n.is_Number and (abs(n - 1) <= ONE_THRESH), lambda n: 1)
    # TODO: maybe something along the lines
    # sph = map(lambda expr: expr.evalf(), flatten(sph))
    # is faster?
    return flatten(sph)


@functools.cache
def norm_pgto(lmn, exponent):
    """Norm of a primitive Cartesian GTO with angular momentum L = l + m + n."""
    L = sum(lmn)
    fact2l, fact2m, fact2n = [factorial2(2 * _ - 1) for _ in lmn]
    return sqrt(
        2 ** (2 * L + 1.5)
        * exponent ** (L + 1.5)
        / fact2l
        / fact2m
        / fact2n
        / pi**1.5
    )


def get_pgto_normalization(L_tots, exponents):
    """Norms could also be precomputed up to L_MAX with a generic exponent.
    Then, the desired exponents could be substituted in."""

    L_norms = list()
    for L_tot, exponent in zip(L_tots, exponents):
        L_norms.append([norm_pgto(lmn, exponent) for lmn in canonical_order(L_tot)])
    # prod() is actually from the math module of the standard library.
    # sympy seems to lack a simple function that just multiplies its arguments.
    exprs = [prod(ns) for ns in it.product(*L_norms)]
    return exprs


def apply_to_components(exprs, components, func):
    """Apply function func to cexprs in components of exprs.

    For n components the order (cexprs_0, cexprs_1, ..., cexprsn) in exprs is expected."""
    nexprs = len(exprs)
    nexprs_per_component = nexprs // components
    mod_exprs = list()
    for i in range(components):
        comp_exprs = exprs[i * nexprs_per_component : (i + 1) * nexprs_per_component]
        mod_exprs.extend(func(comp_exprs))
    return mod_exprs


def integral_gen_for_L(
    int_func, Ls, exponents, contr_coeffs, name, maps, sph=False, norm_pgto=False
):
    time_str = time.strftime("%H:%M:%S")
    start = datetime.now()
    print(f"{time_str} - Generating {Ls} {name}")
    sys.stdout.flush()

    if maps is None:
        maps = list()

    # Generate actual list of integral expressions.
    expect_nexprs = len(list(shell_iter(Ls)))
    exprs = int_func(*Ls)
    contr_coeff_prod = functools.reduce(
        lambda di, dj: di * dj, contr_coeffs[: len(Ls)], 1
    )
    exprs = [contr_coeff_prod * expr for expr in exprs]
    print("\t... Multiplied contraction coefficients")
    nexprs = len(exprs)
    assert len(exprs) % expect_nexprs == 0
    components = nexprs // expect_nexprs
    print("\t... generated expressions")

    # Normalize primitive Cartesian GTOs.
    if norm_pgto:
        pgto_norms = get_pgto_normalization(Ls, exponents)
        exprs = apply_to_components(
            exprs,
            components,
            lambda cexprs: [norm * expr for norm, expr in zip(pgto_norms, cexprs)],
        )
        print("\t... multiplied pGTO normalization factors")
    sys.stdout.flush()

    # Carry out Cartesian-to-spherical transformation, if requested. Or do this later?
    if sph:
        exprs = apply_to_components(
            exprs, components, lambda cexprs: cart2spherical(Ls, cexprs)
        )
        print("\t... did Cartesian to spherical conversion")
        sys.stdout.flush()

    # Common subexpression elimination
    repls, reduced = cse(list(exprs), order="none")
    print("\t... did common subexpression elimination")
    sys.stdout.flush()

    # Replacement expressions, used to form the reduced expressions.
    for i, (lhs, rhs) in enumerate(repls):
        rhs = simplify(rhs)
        rhs = rhs.evalf()
        # Replace occurences of Ax, Ay, Az, ... with A[0], A[1], A[2], ...
        rhs = functools.reduce(lambda rhs, map_: rhs.xreplace(map_), maps, rhs)
        repls[i] = (lhs, rhs)

    # Reduced expression, i.e., the final integrals/expressions.
    for i, red in enumerate(reduced):
        red = simplify(red)
        red = red.evalf()
        reduced[i] = functools.reduce(lambda red, map_: red.xreplace(map_), maps, red)
    # Carry out Cartesian-to-spherical transformation, if requested.
    # if sph:
    # reduced = cart2spherical(Ls, reduced)
    # print("\t... did Cartesian -> Spherical conversion")
    # sys.stdout.flush()

    dur = datetime.now() - start
    print(f"\t... finished in {str(dur)} h")
    sys.stdout.flush()
    return (repls, reduced), Ls


def integral_gen_getter(contr_coeffs, sph=False, norm_pgto=False):
    def integral_gen(
        int_func,
        L_maxs,
        exponents,
        name,
        maps=None,
        sph=sph,
        norm_pgto=norm_pgto,
    ):
        if maps is None:
            maps = list()
        ranges = [range(L + 1) for L in L_maxs]

        for Ls in it.product(*ranges):
            yield integral_gen_for_L(
                int_func, Ls, exponents, contr_coeffs, name, maps, sph, norm_pgto
            )

    return integral_gen


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lmax",
        default=L_MAX,
        type=int,
        help="Generate 1e-integrals up to this maximum angular momentum.",
    )
    parser.add_argument(
        "--lauxmax",
        default=L_AUX_MAX,
        type=int,
        help="Maximum angular moment for integrals using auxiliary functions.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write out generated integrals to the current directory, potentially overwriting the present modules.",
    )
    parser.add_argument(
        "--out-dir",
        default="devel_ints",
        help="Directory, where the generated integrals are written.",
    )
    keys_str = f"({', '.join(KEYS)})"
    parser.add_argument(
        "--keys",
        nargs="+",
        help=f"Generate only certain expressions. Possible keys are: {keys_str}. "
        "If not given, all expressions are generated.",
    )
    parser.add_argument("--sph", action="store_true")
    parser.add_argument("--norm-pgto", action="store_true")

    return parser.parse_args()


def run():
    args = parse_args(sys.argv[1:])

    l_max = args.lmax
    l_aux_max = args.lauxmax
    sph = args.sph
    norm_pgto = args.norm_pgto
    out_dir = Path(args.out_dir if not args.write else ".")
    keys = args.keys
    if keys is None:
        keys = list()
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    try:
        global CART2SPH
        CART2SPH = cart2sph_coeffs(max(l_max, l_aux_max), zero_small=True)
    except NameError:
        pass

    INT_KIND = "Spherical" if sph else "Cartesian"

    # Cartesian basis function centers A, B, C and D.
    center_A = get_center("A")
    center_B = get_center("B")
    center_C = get_center("C")
    center_D = get_center("D")
    # center_R = get_center("R")  TODO: refactor code to use R as multipole origin
    Xa, Ya, Za = symbols("Xa Ya Za")

    # Orbital exponents ax, bx, cx, dx.
    ax, bx, cx, dx = symbols("ax bx cx dx", real=True)

    # Contraction coefficients da, db, dc, dd.
    contr_coeffs = symbols("da db dc dd", real=True)
    da, db, dc, dd = contr_coeffs

    # These maps will be used to convert {Ax, Ay, ...} to array quantities
    # in the generated code. This way an iterable/np.ndarray can be used as
    # function argument instead of (Ax, Ay, Az, Bx, By, Bz).
    A, A_map = get_map("A", center_A)
    B, B_map = get_map("B", center_B)
    C, C_map = get_map("C", center_C)
    D, D_map = get_map("D", center_D)

    boys_import = ("from pysisyphus.wavefunction.ints.boys import boys",)

    integral_gen = integral_gen_getter(
        contr_coeffs=contr_coeffs,
        sph=sph,
        norm_pgto=norm_pgto,
    )

    #################
    # Cartesian GTO #
    #################

    def gto():
        def gto_doc_func(L_tot):
            (La_tot,) = L_tot
            shell_a = L_MAP[La_tot]
            return (
                f"3D {INT_KIND} {shell_a}-Gaussian shell.\n\n"
                "Exponent a, centered at A, evaluated at (Xa, Ya, Za) + A."
            )

        # This code can evaluate multiple points at a time
        gto_Ls = integral_gen(
            lambda La_tot: CartGTOShell(La_tot, ax, Xa, Ya, Za),
            (l_max,),
            (ax,),
            "gto",
        )
        name = ("sph" if sph else "cart") + "_gto3d"
        write_render(
            gto_Ls,
            (ax, da, Xa, Ya, Za),
            name,
            gto_doc_func,
            out_dir,
            c=False,
        )
        print()

    #####################
    # Overlap integrals #
    #####################

    def overlap():
        def ovlp_doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return f"{INT_KIND} 3D ({shell_a}{shell_b}) overlap integral."

        ovlp_ints_Ls = integral_gen(
            lambda La_tot, Lb_tot: gen_overlap_shell(La_tot, Lb_tot, ax, bx, A, B),
            (l_max, l_max),
            (ax, bx),
            "overlap",
            (A_map, B_map),
        )
        write_render(
            ovlp_ints_Ls,
            (ax, da, A, bx, db, B),
            "ovlp3d",
            ovlp_doc_func,
            out_dir,
            c=True,
        )
        print()

    ###########################
    # Dipole moment integrals #
    ###########################

    def dipole():
        def dipole_doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return (
                f"{INT_KIND} 3D ({shell_a}{shell_b}) dipole moment integrals.\n"
                "The origin is at C."
            )

        dipole_comment = """
        Dipole integrals are given in the order:
        for cart_dir in (x, y, z):
            for bf_a in basis_functions_a:
                for bf_b in basis_functions_b:
                    dipole_integrals(cart_dir, bf_a, bf_b)

        So for <s_a|μ|s_b> it will be:

            <s_a|x|s_b>
            <s_a|y|s_b>
            <s_a|z|s_b>
        """

        dipole_ints_Ls = integral_gen(
            lambda La_tot, Lb_tot: gen_multipole_shell(
                # Le_tot = 1
                La_tot,
                Lb_tot,
                ax,
                bx,
                A,
                B,
                1,
                C,
            ),
            (l_max, l_max),
            (ax, bx),
            "dipole moment",
            (A_map, B_map, C_map),
        )
        write_render(
            dipole_ints_Ls,
            (ax, da, A, bx, db, B, C),
            "dipole3d",
            dipole_doc_func,
            out_dir,
            comment=dipole_comment,
            c=True,
        )
        print()

    ###########################################
    # Diagonal of quadrupole moment integrals #
    ###########################################

    def diag_quadrupole():
        def diag_quadrupole_doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return (
                f"{INT_KIND} 3D ({shell_a}{shell_b}) quadrupole moment integrals\n"
                "for operators x², y² and z². The origin is at C."
            )

        diag_quadrupole_comment = """
        Diagonal of the quadrupole moment matrix with operators x², y², z².

        for rr in (xx, yy, zz):
            for bf_a in basis_functions_a:
                for bf_b in basis_functions_b:
                        quadrupole_integrals(bf_a, bf_b, rr)
        """

        diag_quadrupole_ints_Ls = integral_gen(
            lambda La_tot, Lb_tot: gen_diag_quadrupole_shell(
                La_tot, Lb_tot, ax, bx, center_A, center_B, center_C
            ),
            (l_max, l_max),
            (ax, bx),
            "diag quadrupole moment",
            (A_map, B_map, C_map),
        )
        write_render(
            diag_quadrupole_ints_Ls,
            (ax, da, A, bx, db, B, C),
            "diag_quadrupole3d",
            diag_quadrupole_doc_func,
            out_dir,
            comment=diag_quadrupole_comment,
            c=True,
        )
        print()

    ###############################
    # Quadrupole moment integrals #
    ###############################

    def quadrupole():
        def quadrupole_doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return (
                f"{INT_KIND} 3D ({shell_a}{shell_b}) quadrupole moment integrals.\n"
                "The origin is at C."
            )

        quadrupole_comment = r"""
        Quadrupole integrals contain the upper triangular part of the symmetric
        3x3 quadrupole matrix.

        / xx xy xz \\
        |    yy yz |
        \       zz /
        """

        quadrupole_ints_Ls = integral_gen(
            lambda La_tot, Lb_tot: gen_multipole_shell(
                # Le_tot = 2
                La_tot,
                Lb_tot,
                ax,
                bx,
                center_A,
                center_B,
                2,
                center_C,
            ),
            (l_max, l_max),
            (ax, bx),
            "quadrupole moment",
            (A_map, B_map, C_map),
        )

        write_render(
            quadrupole_ints_Ls,
            (ax, da, A, bx, db, B, C),
            "quadrupole3d",
            quadrupole_doc_func,
            out_dir,
            comment=quadrupole_comment,
            c=True,
        )
        print()

    ############################
    # Kinetic energy integrals #
    ############################

    def kinetic():
        def kinetic_doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return f"{INT_KIND} 3D ({shell_a}{shell_b}) kinetic energy integral."

        kinetic_ints_Ls = integral_gen(
            lambda La_tot, Lb_tot: gen_kinetic_shell(
                La_tot, Lb_tot, ax, bx, center_A, center_B
            ),
            (l_max, l_max),
            (ax, bx),
            "kinetic",
            (A_map, B_map),
        )
        write_render(
            kinetic_ints_Ls,
            (ax, da, A, bx, db, B),
            "kinetic3d",
            kinetic_doc_func,
            out_dir,
            c=True,
        )
        print()

    #########################
    # 1el Coulomb Integrals #
    #########################

    def coulomb():
        def coulomb_doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return f"{INT_KIND} ({shell_a}{shell_b}) 1-electron Coulomb integral."

        coulomb_ints_Ls = integral_gen(
            lambda La_tot, Lb_tot: CoulombShell(
                La_tot, Lb_tot, ax, bx, center_A, center_B, center_C
            ),
            (l_max, l_max),
            (ax, bx),
            "coulomb3d",
            (A_map, B_map, C_map),
        )

        write_render(
            coulomb_ints_Ls,
            (ax, da, A, bx, db, B, C),
            "coulomb3d",
            coulomb_doc_func,
            out_dir,
            c=False,
            py_kwargs={"add_imports": boys_import},
        )
        print()

    ###############################################
    # Two-center two-electron repulsion integrals #
    ###############################################

    def _2center2electron():
        def _2center2el_doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return (
                f"{INT_KIND} ({shell_a}|{shell_b}) "
                "two-center two-electron repulsion integral."
            )

        _2center2el_ints_Ls = integral_gen(
            lambda La_tot, Lb_tot: TwoCenterTwoElectronShell(
                La_tot,
                Lb_tot,
                ax,
                bx,
                center_A,
                center_B,
            ),
            (l_aux_max, l_aux_max),
            (ax, bx),
            "_2center2el3d",
            (A_map, B_map),
        )

        write_render(
            _2center2el_ints_Ls,
            (ax, da, A, bx, db, B),
            "_2center2el3d",
            _2center2el_doc_func,
            out_dir,
            c=False,
            py_kwargs={"add_imports": boys_import},
        )
        print()

    #################################################
    # Three-center two-electron repulsion integrals #
    #################################################

    def _3center2electron():
        def _3center2el_doc_func(L_tots):
            La_tot, Lb_tot, Lc_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            shell_c = L_MAP[Lc_tot]
            return (
                f"{INT_KIND} ({shell_a}{shell_b}|{shell_c}) "
                "three-center two-electron repulsion integral."
            )

        _3center2el_ints_Ls = integral_gen(
            lambda La_tot, Lb_tot, Lc_tot: ThreeCenterTwoElectronShell(
                La_tot, Lb_tot, Lc_tot, ax, bx, cx, center_A, center_B, center_C
            ),
            (l_max, l_max, l_aux_max),
            (ax, bx, cx),
            "_3center2el3d",
            (A_map, B_map, C_map),
        )
        write_render(
            _3center2el_ints_Ls,
            (ax, da, A, bx, db, B, cx, dc, C),
            "_3center2el3d",
            _3center2el_doc_func,
            out_dir,
            c=False,
            py_kwargs={"add_imports": boys_import},
        )
        print()

    def _3center2electron_sph():
        def _3center2el_doc_func(L_tots):
            La_tot, Lb_tot, Lc_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            shell_c = L_MAP[Lc_tot]
            return (
                f"{INT_KIND} ({shell_a}{shell_b}|{shell_c}) "
                "three-center two-electron repulsion integral for conversion to "
                "spherical harmonics.\nUses Ahlrichs' vertical recursion relation, "
                "that leaves out some terms, that cancel\nwhen convertig to "
                "spherical harmonics."
            )

        _3center2el_ints_Ls = integral_gen(
            lambda La_tot, Lb_tot, Lc_tot: ThreeCenterTwoElectronSphShell(
                La_tot, Lb_tot, Lc_tot, ax, bx, cx, center_A, center_B, center_C
            ),
            (l_max, l_max, l_aux_max),
            (ax, bx, cx),
            "_3center2el3d_sph",
            (A_map, B_map, C_map),
            sph=False,
        )
        write_render(
            _3center2el_ints_Ls,
            (ax, da, A, bx, db, B, cx, dc, C),
            "_3center2el3d_sph",
            _3center2el_doc_func,
            out_dir,
            c=False,
            py_kwargs={"add_imports": boys_import},
        )
        print()

    #################################
    # Four-center overlap integrals #
    #################################

    def fourcenter_overlap():
        def doc_func(L_tots):
            La_tot, Lb_tot, Lc_tot, Ld_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            shell_c = L_MAP[Lc_tot]
            shell_d = L_MAP[Ld_tot]
            return (
                f"{INT_KIND} ({shell_a}{shell_b}{shell_c}{shell_d}) overlap integral."
            )

        # La_tot, Lb_tot, Lc_tot, Ld_tot, a, b, c, d, A, B, C, D
        ints_Ls = integral_gen(
            lambda La_tot, Lb_tot, Lc_tot, Ld_tot: gen_fourcenter_overlap_shell(
                La_tot,
                Lb_tot,
                Lc_tot,
                Ld_tot,
                ax,
                bx,
                cx,
                dx,
                center_A,
                center_B,
                center_C,
                center_D,
            ),
            (l_max, l_max, l_max, l_max),
            (ax, bx, cx, dx),
            "four_center_overlap",
            (A_map, B_map, C_map, D_map),
        )
        write_render(
            ints_Ls,
            (ax, da, A, bx, db, B, cx, dc, C, dx, dd, D),
            "four_center_overlap",
            doc_func,
            out_dir,
            c=False,
        )
        print()

    funcs = {
        "gto": gto,  # Cartesian Gaussian-type-orbital for density evaluation
        "ovlp": overlap,  # Overlap integrals
        "dpm": dipole,  # Linear moment (dipole) integrals
        "dqpm": diag_quadrupole,  # Diagonal part of the quadrupole tensor
        "qpm": quadrupole,  # Quadratic moment (quadrupole) integrals
        "kin": kinetic,  # Kinetic energy integrals
        "coul": coulomb,  # 1-electron Coulomb integrals
        "2c2e": _2center2electron,  # 2-center-2-electron integrals for density fitting (DF)
        # "3c2e": _3center2electron,  # 3-center-2-electron integrals for DF
        "3c2e_sph": _3center2electron_sph,  # Spherical 3-center-2-electron integrals for DF
        # "4covlp": fourcenter_overlap,  # Four center overlap integral
    }

    # Generate all possible integrals, when no 'keys' were supplied
    if len(keys) == 0:
        keys = funcs.keys()

    start = datetime.now()

    for key in keys:
        funcs[key]()

    duration = datetime.now() - start
    duration_hms = str(duration).split(".")[0]  # Only keep hh:mm:ss
    print(f"sympleint run took {duration_hms} h.")


if __name__ == "__main__":
    run()
