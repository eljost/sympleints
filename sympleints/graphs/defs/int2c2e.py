# [1] EVALUATING MANY-ELECTRON MOLECULAR INTEGRALS FOR
#     QUANTUM CHEMISTRY
#     Womack, PhD thesis


from sympleints.graphs.Integral import Integral
from sympleints.helpers import BFKind


def get_int2c2e_integral(L_tots):
    """See C.2 Two-index Coulomb integrals in [1]."""

    name = "int2c2e"
    integral = Integral(name, L_tots, kinds=[BFKind.SPH, BFKind.SPH])
    integral.add_base("2 * pi**2.5 / (sqrt(px) * ax * bx) * boys(n, mu * R2AB)")
    # VRR, buildup of A
    integral.add_transformation(
        name="vrr",
        center_index=0,
        expr_raw=(
            "PA[pos] * Int(-a+n) "
            "+ (La[pos]-1)/(2*ax) * (Int(-2a) - bx/px * Int(-2a+n)) "
            "+ (Lb[pos])/(2*px) * Int(-a-b+n)"
        ),
    )
    # VRR2, buildup of B
    integral.add_transformation(
        name="vrr2",
        center_index=1,
        expr_raw=(
            "PB[pos] * Int(-b+n) "
            "+ (Lb[pos]-1)/(2*bx) * (Int(-2b) - ax/px * Int(-2b+n)) "
            "+ (La[pos])/(2*px) * Int(-a-b+n)"
        ),
    )
    # Cartesian2Spherical-transformation of B
    integral.add_transformation(name="c2s_b", center_index=1, c2s=True, order=(1, 0))
    # Cartesian2Spherical-transformation of A
    integral.add_transformation(name="c2s_a", center_index=0, c2s=True, order=(0, 1))
    return integral
