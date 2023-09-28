from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from sympy import Symbol

from sympleints.symbols import R


ArgKind = Enum(
    "ArgKind", ("CONTR", "EXPO", "CENTER", "RESULT1", "RESULT2", "RESULT3", "RESULT4")
)


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
    with_ref_center: bool = True
    full_name: Optional["str"] = None
    l_aux_max: Optional[int] = None
    spherical: bool = False
    primitive: bool = False

    def __post_init__(self):
        assert self.l_max >= 0
        assert len(self.coeffs) == len(self.exponents) == len(self.centers)
        assert self.ncomponents >= 0

        # This realizes the generator containing the expressions
        self.ls_exprs = list(self.ls_exprs)

        if self.full_name is None:
            self.full_name = self.name

    @property
    def Ls(self):
        return ["La", "Lb", "Lc", "Ld"][: self.nbfs]

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
            args += [str(self.ref_center)]
        return args

    @property
    def prim_container_args(self):
        container_args = list()
        for exponent, coeff, center in zip(self.exponents, self.coeffs, self.centers):
            container_args.extend([f"{exponent}s", f"{coeff}s", f"{center}"])
        return container_args

    @property
    def full_container_args(self):
        container_args = self.prim_container_args
        if self.with_ref_center:
            container_args += [str(self.ref_center)]
        return container_args

    @property
    def prim_arg_kinds(self) -> list[ArgKind]:
        arg_kinds = list()

        # Every involved basis function contributes
        # one orbital exponent, one contraction coefficient and one center
        for _ in range(self.nbfs):
            arg_kinds += [ArgKind.EXPO, ArgKind.CONTR, ArgKind.CENTER]
        return arg_kinds

    @property
    def full_arg_kinds(self) -> list[ArgKind]:
        arg_kinds = self.prim_arg_kinds
        if self.with_ref_center:
            arg_kinds += [ArgKind.CENTER]
        return arg_kinds

    @property
    def result_kind(self) -> ArgKind:
        # Return RESULT2 when only 1 component is present
        # key = f"RESULT{self.ndim_act}"
        key = f"RESULT{self.ndim}"
        return ArgKind[key]

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

        Equal to the number of basis functions. Incremented by one when multiple
        components are returned, e.g. with linear moment integrals.
        """
        ndim = self.nbfs
        if self.ncomponents > 0:
            ndim += 1
        return ndim

    @property
    def ndim_act(self):
        # Return RESULT2 when only 1 component is present
        ndim = self.ndim
        ndim = ndim - 1 if self.ncomponents in (0, 1) else ndim
        return ndim

    @property
    def cartesian(self):
        return not self.spherical

    # def numba_func_type(self):
    # pass

    # def numba_driver_type(self):
