#!/usr/bin/env python

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

import sympy
from sympy.codegen.ast import Assignment


class IndexedSlice(sympy.Symbol):
    """Symbol representing an array slice.

    Requires '_print_IndexedSlice'-method in code printer, to be printed
    correctly.
    """

    def __new__(cls, name: str, start_index: int, end_index: int) -> IndexedSlice:
        aug_name = f"{name}_{start_index}-{end_index}"
        obj = sympy.Symbol.__new__(cls, aug_name)
        obj.org_name = name
        obj._start_index = start_index
        # 'end_index' must not be confused with the 'stop' parameter of a slice object.
        # With 'start_index' = 0 and 'end_index' = 3 the corresponding slice object is
        # slice(0, 4) or '0:4'.
        # This difference must be taken into account when printing the symbol.
        obj._end_index = end_index
        return obj

    @staticmethod
    def from_start_and_end_indexed(
        start_indexed: sympy.Indexed, end_indexed: sympy.Indexed
    ) -> IndexedSlice:
        for indexed in (start_indexed, end_indexed):
            assert indexed.is_Indexed and len(indexed.indices) == 1
        assert start_indexed.base.name == end_indexed.base.name
        start_index = start_indexed.indices[0]
        # The end index in python is exclusive
        end_index = end_indexed.indices[0]
        return IndexedSlice(start_indexed.base.name, start_index, end_index)

    @staticmethod
    def from_end_indexed_and_distance(
        end_indexed: sympy.Indexed, distance: int
    ) -> IndexedSlice:
        assert distance >= 0
        assert end_indexed.is_Indexed and len(end_indexed.indices) == 1
        end_index = end_indexed.indices[0]
        # The end index in python is exclusive
        start_index = end_index - distance
        return IndexedSlice(end_indexed.base.name, start_index, end_index)

    @property
    def start_index(self) -> int:
        return self._start_index

    @property
    def end_index(self) -> int:
        return self._end_index

    def _sympystr(self, printer, *args):
        # Add one to end_index, as the 'stop' value in python slices is exclusive.
        return f"{self.name}[{self.start_index}:{self.end_index+1}]"


def get_indexed(expr: sympy.Expr, ninds: Optional[int] = None) -> List[sympy.Indexed]:
    """Extract Indexed with a certain amount of indices from sympy expression."""
    indexed: List[sympy.Indexed] = [fs for fs in expr.free_symbols if fs.is_Indexed]
    if ninds is not None:
        indexed = [i for i in indexed if len(i.indices) == ninds]
    return indexed


def get_subs(org_symbs: List[sympy.Expr], repl_symbs: List[sympy.Expr]) -> Dict:
    """Prepare dict for variable substitution in sympy expression."""
    # There may be no Indexed symbols in the expression, resulting in different lengths
    # for org_symbs and repl_symbs, so we can't use 'strict=True'.
    return {org: repl for org, repl in zip(org_symbs, repl_symbs, strict=False)}


def expressions_can_merge(
    expr: Assignment, prev_expr: sympy.Expr, indexed_filter: Optional[str] = None
) -> Tuple[bool, List[sympy.Indexed], List[sympy.Indexed]]:
    """Determine if 'expr' and 'prev_expr' are contiguous and can be merged."""
    if prev_expr is None:
        return False, [], []

    # Current expression
    lhs_indexed = get_indexed(expr.lhs, 1)
    rhs_indexed = get_indexed(expr.rhs, 1)

    if rhs_indexed and indexed_filter is not None:
        filter_re = re.compile(indexed_filter)
        rhs_indexed = [
            indexed for indexed in rhs_indexed if filter_re.match(indexed.name)
        ]

    # Decremebnt indices by one
    lhs_decr = [indexed.base[indexed.indices[0] - 1] for indexed in lhs_indexed]
    rhs_decr = [indexed.base[indexed.indices[0] - 1] for indexed in rhs_indexed]
    lhs_subs = get_subs(lhs_indexed, lhs_decr)
    rhs_subs = get_subs(rhs_indexed, rhs_decr)
    # Substitute decrement indices into the expression and compare with
    # previous expression.
    expr_subbed = Assignment(expr.lhs.xreplace(lhs_subs), expr.rhs.xreplace(rhs_subs))
    can_merge = expr_subbed == prev_expr
    # Or substract expressions?
    # can_merge = sympy.simplify(expr_subbed - prev_expr) == 0
    return can_merge, lhs_indexed, rhs_indexed


def merge(
    expr: sympy.Expr,
    prev_expr: sympy.Expr,
    start_expr: sympy.Expr,
    indexed_filter: Optional[str] = None,
) -> Tuple[bool, sympy.Expr | IndexedSlice]:
    can_merge, lhs_indexed, rhs_indexed = expressions_can_merge(
        expr, prev_expr, indexed_filter=indexed_filter
    )

    # Do nothing if we can't merge
    if not can_merge:
        return False, expr
    # When we can merge we construct the appropriate IndexedSlice objects
    else:
        # LHS is easy. There is only one symbol, which makes the mapping
        # between start_expr and expr unambiguous.
        start_lhs_indexed = get_indexed(start_expr.lhs, 1)
        distance = lhs_indexed[0].indices[0] - start_lhs_indexed[0].indices[0]
        # Substitute Indexd with IndexedSlice objects
        lhs_slice = IndexedSlice.from_start_and_end_indexed(
            start_lhs_indexed[0], lhs_indexed[0]
        )
        mod_lhs = expr.lhs.xreplace({lhs_indexed[0]: lhs_slice})

        # RHS is more complicated, because the same array may appear multiple times
        # with difference indices. To solve this problem we rely on the fact that
        # our slices are contiguous and that we know the (index) distance of the
        # current expression to the start_expr. So the 'end_index' of the IndexSlice
        # object is just index of the respective Indexed object and the 'start_index'
        # is calculate as the difference of 'end_index' and distance.

        rhs_subs = dict()
        for rhsi in rhs_indexed:
            rhs_subs[rhsi] = IndexedSlice.from_end_indexed_and_distance(rhsi, distance)
        mod_rhs = expr.rhs.xreplace(rhs_subs)
        mod_expr = Assignment(mod_lhs, mod_rhs)

    return True, mod_expr


def merge_expressions(exprs: List[sympy.Expr], **kwargs) -> List[sympy.Expr]:
    merged_exprs = list()
    prev_expr = None
    start_expr = None
    for expr in exprs:
        # print(f"{i=}, {expr=}, {prev_expr=}")
        merged, mod_expr = merge(expr, prev_expr, start_expr, **kwargs)
        # print(f"\t{merged=:> 6b}, {mod_expr=}, {start_expr=}")
        if merged:
            # Replace last item with merged expression
            merged_exprs[len(merged_exprs) - 1] = mod_expr
        else:
            start_expr = expr
            merged_exprs.append(mod_expr)
        prev_expr = expr
    return merged_exprs
