# [1] EVALUATING MANY-ELECTRON MOLECULAR INTEGRALS FOR
#     QUANTUM CHEMISTRY
#     Womack, PhD thesis


from sympleints.graphs.Integral import Integral
from sympleints.helpers import BFKind


def get_int_4c2e(L_tots):
    """Electron repulsion integral.

    The ERIs are defined using chemist's notation.

    (ab|cd) = integral over x1 and x2
        a(x1) * b(x1) * 1/r_12 * c(x2) * d(x2)
    """

    integral = Integral(
        "int_4c2e", L_tots, kinds=[BFKind.SPH, BFKind.SPH, BFKind.SPH, BFKind.SPH]
    )
    integral.add_base(
        "2 * pi**2.5 / (px * qx * sqrt(px + qx)) * KAB * KCD * boys(n, boys_arg)"
    )
    # VRR, buildup of A
    integral.add_transformation(
        name="vrr0",
        center_index=0,
        expr_raw=(
            "PA[pos] * Int(-a) "
            "- qx / (px + qx) * PQ[pos] * Int(-a+n) "
            "+ (La[pos]-1)/(2*px) * (Int(-2a) - qx / (px + qx) * Int(-2a+n)) "
            "+ (Lc[pos])/(2 * (px + qx)) * Int(-a-c+n)"
        ),
    )
    # VRR2, buildup of C
    integral.add_transformation(
        name="vrr2",
        center_index=2,
        expr_raw=(
            "QC[pos] * Int(-c) "
            "- px / (px + qx) * PQ[pos] * Int(-c+n) "
            "+ (Lc[pos]-1)/(2*qx) * (Int(-2c) - px / (px + qx) * Int(-2c+n)) "
            "+ (La[pos])/(2 * (px + qx)) * Int(-a-c+n)"
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
        expr_raw="Int(+c-d) + CD[pos] * Int(-d)",
        prefer_index=2,
        order=(1, 0, 2, 3),
    )
    # Cartesian2Spherical-transformation of C
    integral.add_transformation(name="c2s_c", center_index=2, c2s=True)
    # Cartesian2Spherical-transformation of D
    integral.add_transformation(name="c2s_d", center_index=3, c2s=True)
    return integral
