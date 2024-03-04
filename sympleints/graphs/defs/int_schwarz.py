# [1] LIBRETA: Computerized Optimization and Code Synthesis
#     for ElectronRepulsion Integral Evaluation
#     Zhang, 2018


from sympleints.graphs.Integral import Integral
from sympleints.helpers import BFKind


def get_int_schwarz(L_tots):
    """(ab|ab) electron repulsion integrals for Schwarz screening.

    The fact that Bra (ab| and Ket |ab) are identical allows for some
    simplifications."""

    integral = Integral(
        "int_4c2e", L_tots, kinds=[BFKind.SPH, BFKind.SPH, BFKind.SPH, BFKind.SPH]
    )
    integral.add_base("2 * pi**2.5 / (px**2 * sqrt(2 * px)) * KAB**2 * boys(n, 0)")
    # VRR, buildup of A
    integral.add_transformation(
        name="vrr0",
        center_index=0,
        expr_raw=(
            "PA[pos] * Int(-a) "
            "+ (La[pos]-1)/(2*px) * (Int(-2a) - 1/2 * Int(-2a+n)) "
            "+ (Lc[pos])/(4*px) * Int(-a-c+n)"
        ),
    )
    # VRR2, buildup of C
    integral.add_transformation(
        name="vrr2",
        center_index=2,
        expr_raw=(
            "PA[pos] * Int(-c) "
            "+ (Lc[pos]-1)/(2*px) * (Int(-2c) - 1/2 * Int(-2c+n)) "
            "+ (La[pos])/(4*px) * Int(-a-c+n)"
        ),
    )
    # HRR, buildup of B
    integral.add_transformation(
        name="hrr1",
        center_index=1,
        expr_raw="Int(+a-b) + AB[pos] * Int(-b)",
        prefer_index=0,
        order=(1, 0, 2, 3),
    )
    # Cartesian2Spherical-transformation of B
    integral.add_transformation(name="c2s_b", center_index=1, c2s=True)
    # Cartesian2Spherical-transformation of A
    integral.add_transformation(
        name="c2s_a", center_index=0, c2s=True, order=(0, 1, 3, 2)
    )
    # HRR, buildup of D
    integral.add_transformation(
        name="hrr3",
        center_index=3,
        expr_raw="Int(+c-d) + AB[pos] * Int(-d)",
        prefer_index=2,
        order=(1, 0, 2, 3),
    )
    # Cartesian2Spherical-transformation of C
    integral.add_transformation(name="c2s_c", center_index=2, c2s=True)
    # Cartesian2Spherical-transformation of D
    integral.add_transformation(name="c2s_d", center_index=3, c2s=True)
    return integral
