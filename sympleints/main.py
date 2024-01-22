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
# [8] https://arxiv.org/pdf/2007.12057.pdf
#     Fundamentals of Molecular Integrals Evaluation
#     Fermann, Valeev


import argparse
from enum import Enum
from datetime import datetime
import functools
import itertools as it
from math import prod
from pathlib import Path
import sys
import textwrap
import time
from typing import Optional

from jinja2 import Template
from sympy import __version__ as sympy_version
from sympy import (
    Array,
    cse,
    factorial2,
    flatten,
    permutedims,
    pi,
    simplify,
    sqrt,
    symbols,
    tensorcontraction as tc,
    tensorproduct as tp,
)

from sympleints import (
    __version__,
    canonical_order,
    get_center,
    get_map,
    shell_iter,
    get_timer_getter,
)
from sympleints.cart2sph import cart2sph_coeffs, cart2sph_nlms
from sympleints.config import L_MAX, L_AUX_MAX, PREC

# Integral definitions
from sympleints.defs.coulomb import (
    CoulombShell,
    TwoCenterTwoElectronShell,
    # ThreeCenterTwoElectronShell,
    ThreeCenterTwoElectronSphShell,
)

# from sympleints.defs.fourcenter_overlap import gen_fourcenter_overlap_shell
from sympleints.defs.gto import gen_gto3d_shell
from sympleints.defs.kinetic import gen_kinetic_shell
from sympleints.defs.multipole import (
    gen_diag_quadrupole_shell,
    gen_multipole_shell,
    gen_multipole_sph_shell,
)
from sympleints.defs.overlap import gen_overlap_shell, gen_prefactor_shell
from sympleints.Functions import Functions
from sympleints.helpers import L_MAP
from sympleints.l_iters import ll_iter, lllaux_iter

# Renderer
from sympleints.FortranRenderer import FortranRenderer
from sympleints.NumbaRenderer import NumbaRenderer
from sympleints.PythonRenderer import PythonRenderer
from sympleints.symbols import center_R, R, R_map, r, r_map, center_r, rP_map, rP2_map


KEYS = (
    "prefactor",
    "gto",
    "ovlp",
    "dpm",
    "dqpm",
    "qpm",
    "multi_sph",
    "kin",
    "coul",
    "2c2e",
    # "3c2e",  # not really practical
    "3c2e_sph",
)
Normalization = Enum("Normalization", ["PGTO", "CGTO", "NONE"])
normalization_map = {
    "pgto": Normalization.PGTO,
    "cgto": Normalization.CGTO,
    "none": Normalization.NONE,
}


# Pregenerate coefficients; now with mpmath enabled
CART2SPH = cart2sph_coeffs(max(L_MAX, L_AUX_MAX) + 2, zero_small=True, use_mp=True)
NLMS = cart2sph_nlms(max(L_MAX, L_AUX_MAX) + 2)


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
    return flatten(sph)


def get_spherical_quantum_numbers(L_tots):
    L_nlms = [NLMS[L] for L in L_tots]
    return list(it.product(*L_nlms))


@functools.cache
def norm_pgto(lmn, exponent):
    """Norm of a primitive Cartesian GTO with total angular momentum L = l + m + n."""
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


@functools.cache
def lmn_factors(L):
    """Angular momentum vector dependent part of the norm of a contracted GTO."""
    lmn_factors = list()
    for lmn in canonical_order(L):
        lmn_factor = prod([factorial2(2 * am - 1) for am in lmn])
        lmn_factor = 1 / sqrt(lmn_factor)
        lmn_factors.append(lmn_factor)
    return lmn_factors


def get_lmn_factors(Ls):
    all_lmn_factors = [lmn_factors(L) for L in Ls]
    exprs = [prod(ns) for ns in it.product(*all_lmn_factors)]
    return exprs


def apply_to_components(exprs, components, func):
    """Apply function func to cexprs in components of exprs.

    For n components the order (cexprs_0, cexprs_1, ..., cexprs_n) in exprs is expected.
    """
    nexprs = len(exprs)
    nexprs_per_component = nexprs // components
    mod_exprs = list()
    for i in range(components):
        comp_exprs = exprs[i * nexprs_per_component : (i + 1) * nexprs_per_component]
        mod_exprs.extend(func(comp_exprs))
    return mod_exprs


def integral_gen_for_L(
    int_func,
    Ls,
    exponents,
    contr_coeffs,
    name,
    maps,
    sph=False,
    normalization=Normalization.NONE,
    cse_kwargs=None,
    filter_func=None,
):
    time_str = time.strftime("%H:%M:%S")
    start = datetime.now()
    print(f"{time_str} - Processing {Ls} {name}")
    sys.stdout.flush()

    if maps is None:
        maps = list()
    if cse_kwargs is None:
        cse_kwargs = dict()
    if filter_func is None:

        def filter_func(*args):
            return True

    name_L = f"{name}_{''.join(map(str, Ls))}"
    get_timer = get_timer_getter(prefix=f"\t{name_L} ... ", width=45, logger=None)

    expect_nexprs = len(list(shell_iter(Ls)))
    # Actually create expressions by calling the passed function.
    # This is where the magic begins to happen!
    with get_timer("expression generation"):
        exprs, lmns = int_func(*Ls)

    with get_timer("multiplying contraction coefficients"):
        contr_coeff_prod = functools.reduce(
            lambda di, dj: di * dj, contr_coeffs[: len(Ls)], 1
        )
        exprs = [contr_coeff_prod * expr for expr in exprs]
    nexprs = len(exprs)
    assert len(exprs) % expect_nexprs == 0
    components = nexprs // expect_nexprs

    if normalization == Normalization.PGTO:
        with get_timer("multiplying GTO normalization factors"):
            pgto_norms = get_pgto_normalization(Ls, exponents)
            exprs = apply_to_components(
                exprs,
                components,
                lambda cexprs: [norm * expr for norm, expr in zip(pgto_norms, cexprs)],
            )
    elif normalization == Normalization.CGTO:
        with get_timer("multiplying lmn CGTO normalization factors"):
            lmn_factors = get_lmn_factors(Ls)
            exprs = apply_to_components(
                exprs,
                components,
                lambda cexprs: [norm * expr for norm, expr in zip(lmn_factors, cexprs)],
            )

    # Maybe do this later, after the CSE?
    if sph:
        with get_timer("Cartesian to spherical conversion"):
            exprs = apply_to_components(
                exprs, components, lambda cexprs: cart2spherical(Ls, cexprs)
            )
            # NOTE: Cartesian basis functions are characterized by their angular momenta
            # l, m and n, stored in the variable 'lmns'. After transformation to spherical
            # basis functions l, m, n are replaced by the quantum numbers n, l and m, which
            # are stored again  in 'lmns', overwriting the Cartesian values.
            # cart_lmns = lmns   # Original angular momenta could be saved to cart_lmns
            lmns = get_spherical_quantum_numbers(Ls)
            # Repeat updated list of quantum numbers the correct number of times.
            lmns = list(it.chain(*[lmns for _ in range(components)]))

    assert len(exprs) == len(
        lmns
    ), f"Got {len(exprs)} expressions, but {len(lmns)} angular momenta!"
    # Filter expressions, according to their quantum numbers.
    exprs, lmns = zip(
        *[(expr, lmns_) for expr, lmns_ in zip(exprs, lmns) if filter_func(*lmns_)]
    )

    # Common subexpression elimination
    _cse_kwargs = cse_kwargs.copy()
    # 'optimizations': 'basic' becomes too slow for higher angular momenta, at least
    # for 3-center integrals.
    if (max(Ls) > 3) or ((len(Ls) > 2) and max(Ls) > 2):
        try:
            del _cse_kwargs["optimizations"]
        except KeyError:
            pass
    with get_timer("common subexpression elimination"):
        repls, reduced = cse(list(exprs), order="none", **_cse_kwargs)

    # Replacement expressions, used to form the reduced expressions.
    with get_timer("simplifying & evaluating RHS"):
        for i, (lhs, rhs) in enumerate(repls):
            rhs = simplify(rhs)
            rhs = rhs.evalf(PREC)
            # Replace occurences of Ax, Ay, Az, ... with A[0], A[1], A[2], ...
            rhs = functools.reduce(lambda rhs, map_: rhs.xreplace(map_), maps, rhs)
            repls[i] = (lhs, rhs)

    # Reduced expression, i.e., the final integrals/expressions.
    with get_timer("simplifying & evaluating LHS"):
        for i, red in enumerate(reduced):
            red = simplify(red)
            red = red.evalf(PREC)
            reduced[i] = functools.reduce(
                lambda red, map_: red.xreplace(map_), maps, red
            )

    dur = datetime.now() - start
    print(f"\t{name_L} ... finished in {str(dur)} h")
    sys.stdout.flush()
    return Ls, (repls, reduced)


def integral_gen_getter(
    contr_coeffs, sph=False, normalization=Normalization.NONE, cse_kwargs=None
):
    def integral_gen(
        int_func,
        L_maxs,
        exponents,
        name,
        maps=None,
        sph=sph,
        filter_func=None,
        L_iter=None,
    ):
        if maps is None:
            maps = list()
        ranges = [range(L + 1) for L in L_maxs]

        if L_iter is None:
            L_iter = it.product(*ranges)
        L_iter = list(L_iter)

        def inner(Ls):
            return integral_gen_for_L(
                int_func,
                Ls,
                exponents,
                contr_coeffs,
                name,
                maps,
                sph,
                normalization,
                cse_kwargs,
                filter_func=filter_func,
            )

        return L_iter, inner

    return integral_gen


def make_header():
    argv = sys.argv
    args = parse_args(argv[1:])

    cmd = " ".join(sys.argv)
    tpl = Template(
        """
    Molecular integrals over Gaussian basis functions generated by sympleints.
    See https://github.com/eljost/sympleints for more information.

    sympleints version: {{ version }}
    sympy version: {{ sympy_version }}

    sympleints was executed with the following arguments:
    \t{{ cmd }}

    {% for k, v in args._get_kwargs() %}
    \t{{ k }} = {{ v }}
    {% endfor %}
    """,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    header = textwrap.dedent(
        tpl.render(version=__version__, sympy_version=sympy_version, args=args, cmd=cmd)
    ).strip()
    return header


def run(
    l_max: int,
    l_aux_max: int,
    sph: bool = False,
    normalization=Normalization.NONE,
    out_dir=".",
    prefix="ints",
    keys=None,
    cse_kwargs: Optional[dict] = None,
    boys_func: Optional[str] = None,
    cli=False,
):
    out_dir = Path(out_dir).absolute()

    if keys is None:
        keys = list()

    if cse_kwargs is None:
        cse_kwargs = {}

    if cli:
        header = make_header()
    else:
        header = """TODO: fix headers for API call!"""

    int_kind = "Spherical" if sph else "Cartesian"

    # Cartesian basis function centers A, B, C and D.
    center_A = get_center("A")
    center_B = get_center("B")
    center_C = get_center("C")
    center_D = get_center("D")
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

    integral_gen = integral_gen_getter(
        contr_coeffs=contr_coeffs,
        sph=sph,
        normalization=normalization,
        cse_kwargs=cse_kwargs,
    )
    renderers = [
        PythonRenderer(),
        # NumbaRenderer(),
        # FortranRenderer(),
    ]
    results = dict()

    def render_write(funcs):
        nonlocal out_dir

        fns = list()
        for renderer in renderers:
            lang = renderer.language.lower()
            lang_results = results.setdefault(lang, dict())
            rend_out_dir = out_dir / f"{prefix}_{lang}"
            if not rend_out_dir.exists():
                rend_out_dir.mkdir()
            fn = renderer.render_write(funcs, rend_out_dir)
            lang_results[funcs.name] = fn
            fns.append(fn)
        return fns

    #############
    # Prefactor #
    ############

    def prefactor():
        def doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            prefix = "Spherical" if sph else "Cartesian"
            return f"{prefix} prefactors for 3D ({shell_a}{shell_b}) overlap at distance RP."

        ls_exprs = integral_gen(
            lambda La_tot, Lb_tot: gen_prefactor_shell(La_tot, Lb_tot),
            (l_max, l_max),
            (ax, bx),
            "prefactor",
            (rP_map, rP2_map),
            L_iter=ll_iter(l_max),
        )

        prefactor_funcs = Functions(
            name="prefactors",
            l_max=l_max,
            coeffs=[da, db],
            exponents=[ax, bx],
            centers=[A, B],
            ls_exprs=ls_exprs,
            doc_func=doc_func,
            header=header,
            spherical=sph,
            primitive=True,
            hermitian=[0, 1],
            parallel=False,
            with_ref_center=False,
        )
        render_write(prefactor_funcs)

    #################
    # Cartesian GTO #
    #################

    def gto():
        def doc_func(L_tot):
            (La_tot,) = L_tot
            shell_a = L_MAP[La_tot]
            return (
                f"3D {int_kind} {shell_a}-Gaussian shell.\n"
                "Exponent ax, contraction coeff. da, centered at A, evaluated at R."
            )

        # This code can evaluate multiple points at a time
        ls_exprs = integral_gen(
            lambda La_tot: gen_gto3d_shell(La_tot, ax, A, R),
            (l_max,),
            (ax,),
            "gto",
        )
        name = ("sph" if sph else "cart") + "_gto3d"

        gto_funcs = Functions(
            name=name,
            l_max=l_max,
            coeffs=[da],
            exponents=[ax],
            centers=[A],
            ls_exprs=ls_exprs,
            doc_func=doc_func,
            header=header,
            spherical=sph,
        )
        render_write(gto_funcs)

    #####################
    # Overlap integrals #
    #####################

    def overlap():
        def doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return f"{int_kind} 3D ({shell_a}{shell_b}) overlap integral."

        ls_exprs = integral_gen(
            lambda La_tot, Lb_tot: gen_overlap_shell(La_tot, Lb_tot, ax, bx, A, B),
            (l_max, l_max),
            (ax, bx),
            "ovlp3d",
            (A_map, B_map),
            L_iter=ll_iter(l_max),
        )
        ovlp_funcs = Functions(
            name="ovlp3d",
            l_max=l_max,
            coeffs=[da, db],
            exponents=[ax, bx],
            centers=[A, B],
            ls_exprs=ls_exprs,
            ncomponents=1,
            doc_func=doc_func,
            header=header,
            spherical=sph,
            hermitian=[0, 1],
        )
        render_write(ovlp_funcs)

    ###########################
    # Dipole moment integrals #
    ###########################

    def dipole():
        def doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return (
                f"{int_kind} 3D ({shell_a}{shell_b}) dipole moment integrals.\n"
                "The origin is at R."
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

        ls_exprs = integral_gen(
            lambda La_tot, Lb_tot: gen_multipole_shell(
                La_tot, Lb_tot, ax, bx, A, B, 1, R
            ),
            (l_max, l_max),
            (ax, bx),
            "dipole moment",
            (A_map, B_map, R_map),
            L_iter=ll_iter(l_max),
        )

        dipole_funcs = Functions(
            name="dipole3d",
            l_max=l_max,
            coeffs=[da, db],
            exponents=[ax, bx],
            centers=[A, B],
            ls_exprs=ls_exprs,
            doc_func=doc_func,
            comment=dipole_comment,
            ncomponents=3,
            header=header,
            spherical=sph,
            hermitian=[0, 1],
        )
        render_write(dipole_funcs)

    ###########################################
    # Diagonal of quadrupole moment integrals #
    ###########################################

    def diag_quadrupole():
        def doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return (
                f"{int_kind} 3D ({shell_a}{shell_b}) quadrupole moment integrals\n"
                "for operators x², y² and z². The origin is at R."
            )

        diag_quadrupole_comment = """
        Diagonal of the quadrupole moment matrix with operators x², y², z².

        for rr in (xx, yy, zz):
            for bf_a in basis_functions_a:
                for bf_b in basis_functions_b:
                        quadrupole_integrals(bf_a, bf_b, rr)
        """

        ls_exprs = integral_gen(
            lambda La_tot, Lb_tot: gen_diag_quadrupole_shell(
                La_tot, Lb_tot, ax, bx, center_A, center_B, center_R
            ),
            (l_max, l_max),
            (ax, bx),
            "diag quadrupole moment",
            (A_map, B_map, R_map),
            L_iter=ll_iter(l_max),
        )

        diag_quadrupole_funcs = Functions(
            name="diag_quadrupole3d",
            l_max=l_max,
            coeffs=[da, db],
            exponents=[ax, bx],
            centers=[A, B],
            ls_exprs=ls_exprs,
            doc_func=doc_func,
            comment=diag_quadrupole_comment,
            ncomponents=3,
            header=header,
            spherical=sph,
            hermitian=[0, 1],
        )
        render_write(diag_quadrupole_funcs)

    ###############################
    # Quadrupole moment integrals #
    ###############################

    def quadrupole():
        def doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return (
                f"{int_kind} 3D ({shell_a}{shell_b}) quadrupole moment integrals.\n"
                "The origin is at R."
            )

        quadrupole_comment = r"""
        Quadrupole integrals contain the upper triangular part of the symmetric
        3x3 quadrupole matrix.

        / xx xy xz \\
        |    yy yz |
        \       zz /
        """

        ls_exprs = integral_gen(
            lambda La_tot, Lb_tot: gen_multipole_shell(
                # Le_tot = 2
                La_tot,
                Lb_tot,
                ax,
                bx,
                center_A,
                center_B,
                2,
                center_R,
            ),
            (l_max, l_max),
            (ax, bx),
            "quadrupole moment",
            (A_map, B_map, R_map),
            L_iter=ll_iter(l_max),
        )

        quadrupole_funcs = Functions(
            name="quadrupole3d",
            l_max=l_max,
            coeffs=[da, db],
            exponents=[ax, bx],
            centers=[A, B],
            ls_exprs=ls_exprs,
            doc_func=doc_func,
            comment=quadrupole_comment,
            ncomponents=6,
            header=header,
            spherical=sph,
            hermitian=[0, 1],
        )
        render_write(quadrupole_funcs)

    ################################################
    # Integrals for distributed multipole analysis #
    ################################################

    def multipole_sph(Le_tot=2):
        def doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return (
                f"Primitive {int_kind} 3D ({shell_a}{shell_b}) spherical multipole integrals.\n"
                "In contrast to the other multipole integrals, the origin R is calculated\n"
                "inside the function and is (possibly) unique for all primitive pairs."
            )

        ncomponents = sum([2 * L + 1 for L in range(Le_tot + 1)])
        ls_exprs = integral_gen(
            lambda La_tot, Lb_tot: gen_multipole_sph_shell(
                # La_tot, Lb_tot, ax, bx, center_A, center_B, center_R
                # Don't provide an origin R, as it will unique for all primitive pairs
                La_tot,
                Lb_tot,
                ax,
                bx,
                center_A,
                center_B,
                Le_tot=Le_tot,
            ),
            (l_max, l_max),
            (ax, bx),
            "multipole3d_sph",
            (A_map, B_map, R_map),
            L_iter=ll_iter(l_max),
        )

        sph_multipole_funcs = Functions(
            name="multipole3d_sph",
            l_max=l_max,
            coeffs=[da, db],
            exponents=[ax, bx],
            centers=[A, B],
            ls_exprs=ls_exprs,
            doc_func=doc_func,
            # TODO: add comment
            # comment=multipole_comment,
            ncomponents=ncomponents,
            header=header,
            spherical=sph,
            primitive=True,
            hermitian=[0, 1],
        )
        render_write(sph_multipole_funcs)

    ############################
    # Kinetic energy integrals #
    ############################

    def kinetic():
        def doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return f"{int_kind} 3D ({shell_a}{shell_b}) kinetic energy integral."

        ls_exprs = integral_gen(
            lambda La_tot, Lb_tot: gen_kinetic_shell(
                La_tot, Lb_tot, ax, bx, center_A, center_B
            ),
            (l_max, l_max),
            (ax, bx),
            "kinetic",
            (A_map, B_map),
            L_iter=ll_iter(l_max),
        )

        kinetic_funcs = Functions(
            name="kinetic3d",
            l_max=l_max,
            coeffs=[da, db],
            exponents=[ax, bx],
            centers=[A, B],
            ls_exprs=ls_exprs,
            ncomponents=1,
            doc_func=doc_func,
            header=header,
            spherical=sph,
            hermitian=[0, 1],
        )
        render_write(kinetic_funcs)

    #########################
    # 1el Coulomb Integrals #
    #########################

    def coulomb():
        def doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return f"{int_kind} ({shell_a}{shell_b}) 1-electron Coulomb integral."

        ls_exprs = integral_gen(
            lambda La_tot, Lb_tot: CoulombShell(
                La_tot, Lb_tot, ax, bx, center_A, center_B, center_R
            ),
            (l_max, l_max),
            (ax, bx),
            "coulomb3d",
            (A_map, B_map, R_map),
            L_iter=ll_iter(l_max),
        )
        coulomb_funcs = Functions(
            name="coulomb3d",
            l_max=l_max,
            coeffs=[da, db],
            exponents=[ax, bx],
            centers=[A, B],
            ls_exprs=ls_exprs,
            ncomponents=1,
            boys_func=boys_func,
            doc_func=doc_func,
            header=header,
            spherical=sph,
            hermitian=[0, 1],
        )
        render_write(coulomb_funcs)

    ###############################################
    # Two-center two-electron repulsion integrals #
    ###############################################

    def _2center2electron():
        def doc_func(L_tots):
            La_tot, Lb_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            return (
                f"{int_kind} ({shell_a}|{shell_b}) "
                "two-center two-electron repulsion integral."
            )

        ls_exprs = integral_gen(
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
            "int2c2e",
            (A_map, B_map),
            L_iter=ll_iter(l_aux_max),
        )
        _2c2e_funcs = Functions(
            name="int2c2e3d",  # Fortran does not like _2...
            l_max=l_aux_max,
            coeffs=[da, db],
            exponents=[ax, bx],
            centers=[A, B],
            ls_exprs=ls_exprs,
            ncomponents=1,
            boys_func=boys_func,
            doc_func=doc_func,
            header=header,
            spherical=sph,
            hermitian=[0, 1],
        )
        render_write(_2c2e_funcs)

    #################################################
    # Three-center two-electron repulsion integrals #
    #################################################

    """
    # NOT YET UPDATED!
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
            c=False,
            py_kwargs={"add_imports": boys_import},
        )
    """

    def _3center2electron_sph():
        def doc_func(L_tots):
            La_tot, Lb_tot, Lc_tot = L_tots
            shell_a = L_MAP[La_tot]
            shell_b = L_MAP[Lb_tot]
            shell_c = L_MAP[Lc_tot]
            doc_str = (
                f"{int_kind} ({shell_a}{shell_b}|{shell_c}) three-center "
                "two-electron repulsion integral."
            )
            if int_kind == "Cartesian":
                doc_str += (
                    "\nThese integrals MUST BE converted to spherical harmonics!\n"
                    "\nIntegral generation utilized Ahlrichs (truncated) vertical "
                    "recursion relation.\nThere, some terms are omitted, that would "
                    "cancel anyway, after Cartesian->Spherical transformation."
                )
            return doc_str

        ls_exprs = integral_gen(
            lambda La_tot, Lb_tot, Lc_tot: ThreeCenterTwoElectronSphShell(
                La_tot, Lb_tot, Lc_tot, ax, bx, cx, center_A, center_B, center_C
            ),
            (l_max, l_max, l_aux_max),
            (ax, bx, cx),
            "int3c2e3d_sph",
            (A_map, B_map, C_map),
            L_iter=lllaux_iter(l_max, l_aux_max),
        )
        int3c2e_funcs = Functions(
            name="int3c2e3d_sph",  # Fortran does not like _2...
            l_max=l_aux_max,
            coeffs=[da, db, dc],
            exponents=[ax, bx, cx],
            centers=[A, B, C],
            ls_exprs=ls_exprs,
            boys_func=boys_func,
            doc_func=doc_func,
            header=header,
            spherical=sph,
            hermitian=[0, 1],
        )
        render_write(int3c2e_funcs)

    #################################
    # Four-center overlap integrals #
    #################################

    """
    def fourcenter_overlap():
        raise Exception("Switch to new Functions syntax!")

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
                center_D,),
            (l_max, l_max, l_max, l_max),
            (ax, bx, cx, dx),
            "four_center_overlap",
            (A_map, B_map, C_map, D_map),
            L_iter=???
            hermitian=???,
        )
        write_render(
            ints_Ls,
            (ax, da, A, bx, db, B, cx, dc, C, dx, dd, D),
            "four_center_overlap",
            doc_func,
            c=False,
        )
    """

    funcs = {
        "prefactor": prefactor,
        "gto": gto,  # Cartesian Gaussian-type-orbital for density evaluation
        "ovlp": overlap,  # Overlap integrals
        "dpm": dipole,  # Linear moment (dipole) integrals
        "dqpm": diag_quadrupole,  # Diagonal part of the quadrupole tensor
        "qpm": quadrupole,  # Quadratic moment (quadrupole) integrals
        "multi_sph": multipole_sph,  # Integrals for distributed multipole analysis
        "kin": kinetic,  # Kinetic energy integrals
        "coul": coulomb,  # 1-electron Coulomb integrals
        "2c2e": _2center2electron,  # 2-center-2-electron density fitting integrals
        # "3c2e": _3center2electron,  # 3-center-2-electron integrals for DF
        "3c2e_sph": _3center2electron_sph,  # Sph. 3-center-2-electron DF integrals
        # "4covlp": fourcenter_overlap,  # Four center overlap integral
    }
    assert set(funcs.keys()) == set(
        KEYS
    ), "Did you just add a key to 'keys' but forgot to also add it to 'KEYS'?"

    # Generate all possible integrals, when no 'keys' were supplied
    negate_keys = list()
    if len(keys) == 0:
        keys = funcs.keys()
    elif any(negate_keys := [key[1:] for key in keys if "~" in key]):
        keys = [key for key in funcs.keys() if key not in negate_keys]

    for ngk in negate_keys:
        print(f"Skipping generation of '{ngk}'.")
    start = datetime.now()
    for key in keys:
        funcs[key]()
        print()
    duration = datetime.now() - start
    duration_hms = str(duration).split(".")[0]  # Only keep hh:mm:ss
    print(f"sympleint run took {duration_hms} h.")

    return results


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
    """
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write out generated integrals to the current directory, potentially "
        "overwriting the present modules.",
    )
    """
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory, where the generated integrals are written.",
    )
    parser.add_argument(
        "--prefix", default="ints", help="Prefix for the integral directories."
    )
    keys_str = f"({', '.join(KEYS)})"
    parser.add_argument(
        "--keys",
        nargs="+",
        help=f"Generate only certain expressions. Possible keys are: {keys_str}. "
        "If not given, all expressions are generated.",
    )
    parser.add_argument("--sph", action="store_true")
    parser.add_argument(
        "--opt-basic", action="store_true", help="Turn on basic optimizations in CSE."
    )
    parser.add_argument(
        "--boys-func",
        default="pysisyphus.wavefunction.ints.boys",
        help="Which Boys-function to use.",
    )
    parser.add_argument("--normalize", choices=normalization_map.keys(), default="none")

    return parser.parse_args(args)


def run_cli():
    args = parse_args(sys.argv[1:])

    l_max = args.lmax
    l_aux_max = args.lauxmax
    sph = args.sph
    normalization = normalization_map[args.normalize]
    out_dir = args.out_dir
    prefix = args.prefix
    keys = args.keys
    boys_func = args.boys_func

    cse_kwargs = None
    if args.opt_basic:
        cse_kwargs = {
            "optimizations": "basic",
        }

    return run(
        l_max=l_max,
        l_aux_max=l_aux_max,
        sph=sph,
        normalization=normalization,
        out_dir=out_dir,
        prefix=prefix,
        keys=keys,
        cse_kwargs=cse_kwargs,
        boys_func=boys_func,
        cli=True,
    )


if __name__ == "__main__":
    run_cli()
