from dataclasses import dataclass
from typing import Callable, List, Optional

from sympy import Symbol

from sympleints.symbols import R


@dataclass
class Functions:
    name: str
    l_max: int
    coeffs: List[Symbol]
    exponents: List[Symbol]
    centers: List[Symbol]
    ls_exprs: List
    header: str = ""
    doc_func: Optional[Callable] = None
    comment: str = ""
    boys: bool = False
    ncomponents: int = 0
    with_ref_center: bool = False
    full_name: Optional["str"] = None
    l_aux_max: Optional[int] = None
    spherical: bool = False

    def __post_init__(self):
        assert self.l_max >= 0
        assert len(self.coeffs) == len(self.exponents) == len(self.centers)

        self.ls_exprs = list(self.ls_exprs)

        if self.full_name is None:
            self.full_name = self.name

    @property
    def Ls(self):
        return ["La", "Lb", "Lc", "Ld"][: self.nbfs]

    def get_args(self, ref_center=False):
        args = list()
        for exponent, coeff, center in zip(self.exponents, self.coeffs, self.centers):
            args.extend([str(exponent), str(coeff), str(center)])
        return args

    @property
    def prim_args(self):
        args = list()
        for exponent, coeff, center in zip(self.exponents, self.coeffs, self.centers):
            args.extend([str(exponent), str(coeff), str(center)])
        return args

    @property
    def full_args(self):
        args = self.prim_args
        if self.with_ref_center:
            args += [self.ref_center]
        return args

    @property
    def ref_center(self):
        return R

    @property
    def L_args(self):
        return ["La", "Lb", "Lc", "Ld"][: self.nbfs]

    @property
    def ls(self):
        return [ls for ls, _ in self.ls_exprs]

    @property
    def nbfs(self):
        return len(self.coeffs)

    @property
    def ls_name_map(self):
        name = self.name
        return {
            ls: f"{name}_" + "".join([str(l) for l in ls]) for ls, _ in self.ls_exprs
        }

    @property
    def ndim(self):
        """Number of dimension in the final result array.

        Equal to the number of basis functions. Incremented by one when multiple components
        are returned, e.g. with linear moment integrals.
        """
        ndim = self.nbfs
        if self.ncomponents > 0:
            ndim += 1
        return ndim

    @property
    def cartesian(self):
        return not self.spherical
