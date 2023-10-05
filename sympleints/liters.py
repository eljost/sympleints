import itertools as it


# As the OS recursion relations build up angular momentum from left to right
# it is most economical to increase angular momentum at the left-most index first.
# Basically the wrappers below are thin wrappers around itertools-functions to
# yield column-major indices.


def ll_iter(lmax):
    # Note the order (Lb, La), ensuring Lb >= La.
    for Lb, La in it.combinations_with_replacement(range(lmax + 1), 2):
        yield (La, Lb)


def lllaux_iter(lmax, lauxmax):
    # Note the order (Lb, La), ensuring Lb >= La.
    for Lb, La in it.combinations_with_replacement(range(lmax + 1), 2):
        for Lc in range(lauxmax + 1):
            yield (La, Lb, Lc)


def schwarz_iter(lmax):
    # Note the order (Lb, La), ensuring Lb >= La.
    # Yields total angular momenta for Schwarz integrals
    # (00|00), (10|10), (11|11) etc.
    for Lb, La in it.combinations_with_replacement(range(lmax + 1), 2):
        yield La, Lb, La, Lb
