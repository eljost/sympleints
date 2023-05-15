from sympleints.graphs.Integral import Integral
from sympleints.helpers import BFKind


def get_int_3c2e(L_tots):
    integral = Integral("int_3c2e", L_tots, kinds=[BFKind.SPH, BFKind.SPH, BFKind.SPH])
    integral.add_base("theta * kappa * boys(n, alpha * R2PC)")
    # VRR, buildup of A
    integral.add_transformation(
        name="vrr",
        center_index=0,
        expr_raw=(
            "PA[pos] * Int(-a) - alpha/px * PC[pos] * Int(-a+n) "
            "+ (La[pos]-1)/(2*px) * (Int(-2a) - alpha/px * Int(-2a+n))"
        ),
        order=(2, 0, 1),
        L_target_func=lambda L_tots: L_tots[0] + L_tots[1],
    )
    # VRR2, buildup of C
    integral.add_transformation(
        name="vrr2",
        center_index=2,
        expr_raw="alpha / cx * PC[pos] * Int(-c+n) + La[pos]/(2*(px+cx)) * Int(-a-c+n)",
    )
    # Cartesian2Spherical-transformation of C
    integral.add_transformation(name="c2s_c", center_index=2, c2s=True)
    # HRR, buildup of B
    integral.add_transformation(
        name="hrr",
        center_index=1,
        expr_raw="Int(+a-b) + AB[pos] * Int(-b)",
        prefer_index=0,
        order=(1, 0, 2),
    )
    # TODO: avoid manual specification of order and final C2S transformations.
    # Add some kind of build() or finalize() method that adds the missing transformations
    # and sets the correct order and applies the missing C2S transformations.
    # Cartesian2Spherical-transformation of B
    # This could also include some sanity checks, as comparing the lengths of kinds
    # and order etc.
    integral.add_transformation(name="c2s_b", center_index=1, c2s=True)
    # Cartesian2Spherical-transformation of A
    integral.add_transformation(name="c2s_a", center_index=0, c2s=True, order=(0, 1, 2))
    return integral
