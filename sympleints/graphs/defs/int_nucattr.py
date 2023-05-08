# [1] https://doi.org/10.1039/D1CP02805G
#     Efficient evaluation of electrostatic potential with computerized
#     optimized code
#     Zhang, Lu, 2021

from sympleints.graphs.Integral import Integral
from sympleints.helpers import BFKind


def get_nucattr_integral(L_tots):
    """2-center-1-electron nuclear attraction integral."""

    name = "int_nucattr"
    integral = Integral(name, L_tots, kinds=[BFKind.SPH, BFKind.SPH])
    integral.add_base("2 * pi / px * K * boys(n, px * R2PR)")
    # VRR, buildup of A
    integral.add_transformation(
        name="vrr",
        center_index=0,
        expr_raw=(
            "PA[pos] * Int(-a) "
            "+ (La[pos]-1)/(2*px) * (Int(-2a) - Int(-2a+n)) "
            "- PR[pos] * Int(-a+n)"
        ),
    )
    # HRR, buildup of B
    integral.add_transformation(
        name="hrr",
        center_index=1,
        expr_raw="Int(+a-b) + AB[pos] * Int(-b)",
        order=(1, 0),
    )
    # Cartesian2Spherical-transformation of B
    integral.add_transformation(
        name="c2s_b", center_index=1, c2s=True
    )  # , order=(1, 0))
    # Cartesian2Spherical-transformation of A
    integral.add_transformation(name="c2s_a", center_index=0, c2s=True, order=(0, 1))
    return integral
