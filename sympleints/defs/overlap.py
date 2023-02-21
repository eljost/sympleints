from sympleints.defs.multipole import gen_multipole_shell


def gen_overlap_shell(La_tot, Lb_tot, a, b, A, B):
    exprs, lmns = gen_multipole_shell(La_tot, Lb_tot, a, b, A, B)
    return exprs, lmns
