from dataclasses import dataclass

import numpy as np

from sympleints.testing.normalization import norm_cgto_lmn


@dataclass
class Shell:
    L: int
    center: np.ndarray
    coeffs: np.ndarray
    exps: np.ndarray
    center_ind: int
    atomic_num: int

    def __post_init__(self):
        self.coeffs, _ = norm_cgto_lmn(self.coeffs, self.exps, self.L)

    @property
    def cart_size(self) -> int:
        return (self.L + 2) * (self.L + 1) // 2

    @property
    def sph_size(self) -> int:
        return 2 * self.L + 1
