from collections import OrderedDict
from typing import List, Optional, Tuple

from sympy.codegen.ast import Assignment

from sympleints.graphs.AngMoms import name_for_L_tots_kinds
from sympleints.graphs.Transform import parse_raw_expr, RecurRel, Cart2Sph
from sympleints.helpers import BFKind


class Integral:
    colors = (
        "red",
        "green",
        "blue",
        "orange",
        "black",
        "hotpink",
        "gold",
        "tan",
        "springgreen",
    )

    def __init__(
        self,
        name: str,
        L_tots: Tuple[int],
        kinds: Optional[List[BFKind]] = None,
        order=None,
        with_aux: bool = True,
    ):
        self.name = name
        self._L_tots = L_tots
        self._ninds = len(self.L_tots)
        if kinds is None:
            kinds = [BFKind.CART] * self.ncenters
        self.kinds = list(kinds)
        if order is None:
            order = tuple(range(self.ncenters))
        self.order = order
        self.with_aux = bool(with_aux)

        self.target_name = name_for_L_tots_kinds(
            self.L_tots, kinds=kinds, with_aux=with_aux
        )
        # Always start from Cartesian functions. self.cur_kinds can differ from
        # self.kinds, when integrals over spherical functions are required.
        self.cur_kinds = [BFKind.CART] * self.ncenters

        # Even though python dicts are ordered by default from >= 3.7 on,
        # we make it explicit here by using an OrderedDict.
        self._transformations = OrderedDict()

    @property
    def L_tots(self):
        return self._L_tots

    @property
    def ncenters(self):
        return self._ninds

    @property
    def ninds(self):
        return self._ninds + int(self.with_aux)

    @property
    def rrs(self):
        return list(self._transformations.values())

    def add_base(self, base_raw):
        self._base = parse_raw_expr(base_raw)

    def make_rr(
        self,
        name: str,
        center_index: int,
        expr_raw: str,
        **kwargs,
    ):
        rr_expr = parse_raw_expr(expr_raw)

        transf = RecurRel(
            name=name,
            center_index=center_index,
            kinds=self.cur_kinds,
            ninds=self.ninds,
            L_tots=self.L_tots,
            rr_expr=rr_expr,
            **kwargs,
        )
        return transf

    def make_c2s(self, name, center_index: int, **kwargs):
        new_kinds = self.cur_kinds.copy()
        assert new_kinds[center_index] == BFKind.CART
        new_kinds[center_index] = BFKind.SPH
        transf = Cart2Sph(
            name=name,
            center_index=center_index,
            kinds=new_kinds,
            ninds=self.ninds,
            L_tots=self.L_tots,
            **kwargs,
        )
        return transf, new_kinds

    def add_transformation(
        self,
        name: str,
        center_index: int,
        expr_raw: Optional[str] = None,
        c2s: bool = False,
        order=None,
        **kwargs,
    ):
        if order is None:
            # Try to use order from previous transformation
            try:
                prev_key = tuple(self._transformations.keys())[-1]
                order = self._transformations[prev_key].order
            except IndexError:
                order = tuple(range(self.ninds))

        _edge_attrs = {
            "color": self.colors[len(self.rrs) % len(self.colors)],
        }
        if "edge_attrs" not in kwargs:  # is None:
            edge_attrs = {}
        else:
            edge_attrs = kwargs["edge_attrs"]
        _edge_attrs.update(edge_attrs)
        kwargs["edge_attrs"] = _edge_attrs

        if expr_raw is not None:
            transf = self.make_rr(name, center_index, expr_raw, order=order, **kwargs)
        elif c2s:
            transf, new_kinds = self.make_c2s(name, center_index, order=order, **kwargs)
            # The current kinds have to be updated after transformation
            self.cur_kinds = new_kinds
        else:
            raise Exception("Either 'expr_raw' must be provided or 'c2s' must be True!")

        self._transformations[name] = transf
        return transf

    @property
    def opt_transforms(self):
        """When optimizing the RR we go from top to bottom, so we reverse the order."""
        return self.rrs[::-1]

    def get_base_expr(self, ang_moms, compr_arr_map, index_map):
        lhs = compr_arr_map[ang_moms.as_key()][index_map[ang_moms]]
        rhs = self._base.expr
        if self.with_aux:
            rhs = rhs.subs(self._base.n, ang_moms.aux.n)
        return Assignment(lhs, rhs)

    @property
    def shell_size(self):
        shell_size = 1
        size_funcs = {
            BFKind.CART: lambda L: (L + 2) * (L + 1) // 2,
            BFKind.SPH: lambda L: 2 * L + 1,
        }
        for L, kind in zip(self.L_tots, self.kinds):
            shell_size *= size_funcs[kind](L)
        return shell_size
