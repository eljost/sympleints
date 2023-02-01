from dataclasses import dataclass
from typing import Callable, List, Optional

from sympy import Symbol


@dataclass
class Functions:
    name: str
    l_max: int
    coeffs: List[Symbol]
    exponents: List[Symbol]
    centers: List[Symbol]
    ls_exprs: List
    doc_func: Optional[Callable]
    add_imports: Optional[List[str]] = None
    comment: str = ""
    header: str = ""
    add_args: Optional[List] = None
    ncomponents: int = 0
    ref_center: Optional[Symbol] = None
    full_name: Optional["str"] = None
    l_aux_max: Optional[int] = None

    def __post_init__(self):
        assert self.l_max >= 0
        assert len(self.coeffs) == len(self.exponents) == len(self.centers)

        if self.full_name is None:
            self.full_name = self.name

        if self.add_args is None:
            self.add_args = list()

        if self.add_imports is None:
            self.add_imports = list()

    @property
    def args(self):
        args = list()
        for exponent, coeff, center in zip(self.exponents, self.coeffs, self.centers):
            args.extend([str(exponent), str(coeff), str(center)])
        if self.ref_center is not None:
            args.extend([str(self.ref_center)])

        args += [str(arg) for arg in self.add_args]
        return args

    @property
    def L_args(self):
        return ["La", "Lb", "Lc", "Ld"][:self.nbfs]

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
