# [1]   https://doi.org/10.1016/0009-2614(91)80260-5
#       An efficient method of implementing the horizontal recurrence
#       relation in the evaluation of electron repulsion integrals using
#       Cartesian Gaussian functions
#       Ryu, Lee, Lindh, 1991

from sympleints.graphs.triangle_rec import get_pos_dict


def test_min_costs():
    L_max = 8
    pos_dict = get_pos_dict(L_max, strict=True)
    cost_dict = dict()
    for L_target in range(1, L_max + 1):
        pos_for_L_target = pos_dict[L_target]
        for L in range(1, L_target + 1):
            cost_dict.setdefault(L_target, list()).append(len(pos_for_L_target[L]))

    # Reference costs
    ref_cost_dict = {
        1: [3],
        2: [3, 6],
        # f-shell requires only 4 d-shells
        3: [3, 4, 10],
        # g-shell requires only 6 f-shells
        4: [3, 3, 6, 15],
        5: [3, 3, 5, 9, 21],
        # From the line one things could be improved.
        # Actually only 4-shells are required, not 5... Interestingly 7 and 12
        # agree again with values reported by Ryu et. al.
        6: [3, 3, 5, 7, 12, 28],
        7: [3, 3, 5, 8, 9, 15, 36],
        8: [3, 3, 5, 7, 9, 12, 19, 45],
    }
    assert cost_dict == ref_cost_dict
