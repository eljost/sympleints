from math import prod

import numpy as np
from scipy.special import factorial2 as sp_factorial2

from sympleints.helpers import canonical_order


def factorial2(n: int) -> int:
    """Scipy 1.11 decided that (-1)!! is not 1 anymore!

    Please see https://github.com/scipy/scipy/issues/18813."""
    if n == -1:
        return 1
    elif n < -1:
        raise Exception(f"Only supported negative argument is -1, but got {n}!")
    return sp_factorial2(n)


def get_lmn_factors(L: int) -> np.ndarray:
    lmns = canonical_order(L)
    lmn_factors = np.zeros(len(lmns))
    for i, lmn in enumerate(lmns):
        lmn_factors[i] = prod([factorial2(2 * am - 1) for am in lmn])
    lmn_factors = 1 / np.sqrt(lmn_factors)
    return lmn_factors


def norm_cgto_lmn(
    coeffs: np.ndarray, exps: np.ndarray, L: int
) -> tuple[np.ndarray, np.ndarray]:
    N = 0.0
    for i, expi in enumerate(exps):
        for j, expj in enumerate(exps):
            tmp = coeffs[i] * coeffs[j] / (expi + expj) ** (L + 1.5)
            tmp *= np.sqrt(expi * expj) ** (L + 1.5)
            N += tmp
    N = np.sqrt(exps ** (L + 1.5) / (np.pi**1.5 / 2**L * N))

    mod_coeffs = N * coeffs
    lmn_factors = get_lmn_factors(L)

    return mod_coeffs, lmn_factors
