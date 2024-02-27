from dataclasses import dataclass
from enum import Enum
import itertools as it
from typing import Callable, List, Optional

from pathos.pools import ProcessPool
import psutil
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
    boys_func: Optional[str] = None
    ncomponents: int = 1
    with_ref_center: bool = True
    full_name: Optional["str"] = None
    l_aux_max: Optional[int] = None
    spherical: bool = False
    primitive: bool = False
    parallel: Optional[bool] = True
    hermitian: Optional[list[int]] = None

    def __post_init__(self):
        assert self.l_max >= 0
        assert len(self.coeffs) == len(self.exponents) == len(self.centers)
        assert self.ncomponents >= 1

        L_iter, inner = self.ls_exprs
        self.L_iter = L_iter
        if self.parallel:
            # This realizes the generator containing the expressions
            #
            # ... either in parallel
            # use all but 1 CPU cores
            nodes = max(1, psutil.cpu_count(logical=False) - 1)
            pool = ProcessPool(nodes=nodes)
            self.ls_exprs = pool.map(inner, L_iter)
        else:
            #
            # ... or in serial
            self.ls_exprs = [inner(Ls) for Ls in L_iter]

        if self.full_name is None:
            self.full_name = self.name

        if self.hermitian is None:
            self.hermitian = []

    @property
    def Ls(self):
        return ["La", "Lb", "Lc", "Ld"][: self.nbfs]

    def bf_args(self, i):
        """Basis function arguments: exponent, contr. coefficient and center."""
        return [str(lst[i]) for lst in (self.exponents, self.coeffs, self.centers)]

    def prim_args_for_bf_inds(self, bf_inds):
        return list(it.chain(*[self.bf_args(i) for i in bf_inds]))

    @property
    def prim_args(self):
        return self.prim_args_for_bf_inds(range(self.nbfs))

    def full_args_for_bf_inds(self, bf_inds):
        args = self.prim_args_for_bf_inds(bf_inds)
        if self.with_ref_center:
            args += [str(self.ref_center)]
        return args

    @property
    def full_args(self):
        return self.full_args_for_bf_inds(range(self.nbfs))

    @property
    def prim_container_args(self):
        return [f"{arg}s" for arg in self.prim_args]

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
        return self.L_iter

    @property
    def nbfs(self):
        # Return number of total angular momenta
        return len(self.ls_exprs[0][0])

    @property
    def ls_name_map(self):
        name = self.name
        return {ls: f"{name}_" + "".join([str(l) for l in ls]) for ls in self.ls}

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
