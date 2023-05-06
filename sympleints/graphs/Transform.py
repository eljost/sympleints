import abc
from dataclasses import dataclass
import re
from typing import Dict, Optional, Tuple

import networkx as nx
import sympy
from sympy.codegen.ast import Assignment

from sympleints.graphs.AngMoms import AngMoms, CartAngMom, SphAngMom
from sympleints.helpers import BFKind, shell_iter

from pysisyphus.wavefunction.cart2sph import expand_sph_quantum_numbers


KEY_MAP = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "n": -1,  # Must be modified by number centers
}
MOD_RE = re.compile(r"([+-]{1})(\d?)([abcdn]{1})")


def masks_for_modifier(modifier, ninds):
    res = MOD_RE.findall(modifier)
    assert res
    mask = [0] * ninds
    for sign, num, key in res:
        if num == "":
            num = 1
        num = int(num)
        assert num > 0
        num = -num if (sign == "-") else num
        index = KEY_MAP[key]
        if key == "n":
            index += ninds
        mask[index] = num
    return mask


def nodes_by_level(G, index, drop_aux=True):
    grouped = dict()
    for node in G:
        compressed = node.compressed(drop_aux)
        key = compressed[index]
        val = node.drop_aux() if drop_aux else node
        grouped.setdefault(key, list()).append(val)
    return grouped


def unique_compressed_in_G(G):
    return set([node.drop_aux().compressed_unique() for node in G.nodes()])


def unique_kinded_tot_ang_moms_in_G(G: nx.DiGraph):
    return tuple(set([node.to_kinded_tot_ang_moms() for node in G.nodes()]))


def collapse_to_key_graph(G):
    Gc = nx.DiGraph()
    for edge in G.edges():
        from_, to_ = edge
        Gc.add_edge(from_.as_key(), to_.as_key())
    return Gc


@dataclass
class RRExpr:
    raw_expr: str
    expr: sympy.Expr
    modifiers: Tuple[str]
    ints: Tuple[sympy.Symbol]
    angmom_vars: Tuple[sympy.Symbol]
    angmom_subs: Tuple[sympy.Symbol]
    pos_vec_vars: Tuple[sympy.Symbol]
    pos_vec_subs: Tuple[sympy.IndexedBase]
    c2s_ind: Optional[int] = None

    # Sympy symbol of auxiliary index n
    n = sympy.Symbol("n")


def parse_raw_expr(raw_expr: str, sub_key="Int") -> RRExpr:
    sub_re = re.compile(rf"{sub_key}\((.+?)\)")
    center_inds = "abcd"
    angmom_re = re.compile(rf"(L[{center_inds}])\[pos\]")
    pos_re = re.compile(r"([ABCDPQR]{1,2})\[pos\]")
    c2s_re = re.compile(r"c2s\(([abcd]{{1}})\)")

    if bool(c2s_re.search(raw_expr)):
        c2s_mobj = c2s_re.fullmatch(raw_expr)
        assert c2s_mobj, "c2s(arg) must be given alone!"
        c2s_ind = list(center_inds).index(c2s_mobj.group(1))
    else:
        c2s_ind = None

    # Extract modifier for recursion relations
    modifiers = tuple(sub_re.findall(raw_expr))

    # Replace Integral reference with dummy variables
    ints = list()
    expr_subbed = raw_expr
    for i, _ in enumerate(modifiers):
        repl = f"{sub_key}{i}"
        ints.append(repl)
        expr_subbed = sub_re.sub(repl, expr_subbed, count=1)

    # Replace references to angular momenta
    # La[pos] -> La_pos
    angmoms = angmom_re.findall(expr_subbed)
    angmom_inds = list()
    for angmom in angmoms:
        repl = f"{angmom}_pos"
        expr_subbed = angmom_re.sub(repl, expr_subbed, count=1)
        angmom_ind = center_inds.index(angmom[1])
        angmom_inds.append(angmom_ind)

    # Replace 'pos' dependent references to vectors
    # PA[pos] -> PA_pos
    pos_vecs = pos_re.findall(expr_subbed)
    for pos_vec in pos_vecs:
        repl = f"{pos_vec}_pos"
        expr_subbed = pos_re.sub(repl, expr_subbed, count=1)

    # Create actual sympy expressions/symbols
    expr = sympy.sympify(expr_subbed)
    # It seems we can't indicate real=True/integer=True here, as this will mess up
    # the subs later in the RR.
    ints = sympy.symbols(ints)
    angmom_vars = [sympy.Symbol(f"{am}_pos") for am in angmoms]
    pos_vec_vars = [sympy.Symbol(f"{pv}_pos") for pv in pos_vecs]
    pos_vec_subs = [sympy.IndexedBase(pv, shape=3) for pv in pos_vecs]
    res = RRExpr(
        raw_expr=raw_expr,
        expr=expr,
        modifiers=modifiers,
        ints=ints,
        angmom_vars=angmom_vars,
        angmom_subs=angmom_inds,
        pos_vec_vars=pos_vec_vars,
        pos_vec_subs=pos_vec_subs,
        c2s_ind=c2s_ind,
    )
    return res


class Transform(metaclass=abc.ABCMeta):
    is_c2s = False

    def __init__(
        self,
        *,
        name: str,
        center_index,
        kinds,
        ninds,
        L_tots,
        L_target_func=None,
        order=None,
        edge_attrs: Optional[Dict] = None,
    ):
        self.name = name
        self.center_index = center_index
        self.kinds = kinds
        self.ninds = ninds
        self.L_tots = L_tots
        # L_target may be higher than self.L_tots[self.center_index] because
        # some recursion relations build up a higher angular momentum at a given
        # index, that is later reduced by another recursion relation.
        if L_target_func is None:
            L_target_func = lambda L_tots: L_tots[self.center_index]
        self.L_target_func = L_target_func
        self.L_target = self.L_target_func(self.L_tots)
        if order is None:
            order = range(len(kinds))
        self.order = order
        assert self.center_index in self.order
        if edge_attrs is None:
            edge_attrs = dict()
        self._edge_attrs = edge_attrs

        self._edge_attrs["rr"] = self.name
        self.ncenters = len(self.order)

    @property
    def edge_attrs(self):
        return self._edge_attrs

    @abc.abstractmethod
    def apply(
        self, ang_moms, reduce_index=None, drop_none=True
    ) -> Tuple[Tuple[AngMoms, ...], int]:
        pass

    @abc.abstractmethod
    def get_expr(self, ang_moms, pos, compr_arr_map, index_map):
        pass

    @abc.abstractmethod
    def target_node_iter(self, G: nx.DiGraph, array_index_map: Dict):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class RecurRel(Transform):
    def __init__(
        self,
        *,
        rr_expr,
        prefer_index=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rr_expr = rr_expr
        self.prefer_index = prefer_index

        self.masks = [
            masks_for_modifier(mod, self.ninds) for mod in self.rr_expr.modifiers
        ]
        self.pos_vec_subs = dict()
        # Pre-populate position vector substitutes
        for pos in range(3):
            pvs = [pv[pos] for pv in self.rr_expr.pos_vec_subs]
            self.pos_vec_subs[pos] = list(zip(self.rr_expr.pos_vec_vars, pvs))
        # TODO: pre-populate angular momentum?

    def apply(self, ang_moms, reduce_index=None, drop_none=True):
        if reduce_index is None:
            reduce_index = ang_moms.get_reduce_index(
                self.center_index,
                prefer_index=self.prefer_index,
                L_target=self.L_target,
                name=self.name,
            )
        mod_ang_moms = tuple()
        for mask in self.masks:
            tmp = ang_moms.apply_mask(mask, reduce_index)
            if (tmp is not None) or (not drop_none):
                mod_ang_moms += (tmp,)
        return mod_ang_moms, reduce_index

    def get_expr(self, ang_moms, pos, compr_arr_map, index_map):
        # Left hand side of the assignment
        # lhs = compr_arr_map[ang_moms.compressed()][index_map[ang_moms_na]]
        lhs = compr_arr_map[ang_moms.as_key()][index_map[ang_moms]]

        # Build up right hand side of the assignment, including substitutions
        #   1.) Apply RR to get AngMoms objects in expression
        rhs_ang_moms, _ = self.apply(ang_moms, reduce_index=pos, drop_none=False)
        ints = self.rr_expr.ints  # Dummy variables to be substituted
        #  Replace missing/None AngMoms with Zero, to they cancel from the expression
        rhs_symbols = [sympy.S.Zero] * len(ints)
        for i, rhs_am in enumerate(rhs_ang_moms):
            if rhs_am is not None:
                rhs_symbols[i] = compr_arr_map[rhs_am.as_key()][index_map[rhs_am]]
        # Substitute in integrals
        ints_sub = zip(ints, rhs_symbols)
        expr = self.rr_expr.expr.subs(ints_sub)

        # Substitute in coordinate arrays to replace PA[pos] etc.
        expr = expr.subs(self.pos_vec_subs[pos])

        # Substitute in angular momenta to replace La[pos] etc.
        ang_mom_subs = zip(
            self.rr_expr.angmom_vars,
            [ang_moms[i][pos] for i in self.rr_expr.angmom_subs],
        )
        rhs = expr.subs(ang_mom_subs)

        return Assignment(lhs, rhs)

    def target_node_iter(self, G, key_index_map):
        # Determine resolution order by collapsing the nodes to their keys.
        Gkey = collapse_to_key_graph(G)
        # Topological sort gives os the required evaluation order.
        node_gen = reversed(list(nx.topological_sort(Gkey)))
        target_keys = [node for node in node_gen if Gkey.out_degree(node) > 0]

        for key in target_keys:
            yield from key_index_map[key]


def get_c2s_map(L_tot: int):
    """Map between LHS transformed SphAngMom, dependent RHS CartAngMoms and factors."""
    c2s_map = dict()
    for raw_ang_moms in shell_iter((L_tot,), kinds=(BFKind.SPH,)):
        assert len(raw_ang_moms) == 1
        Lm = raw_ang_moms[0]
        sph_am = SphAngMom(*Lm)
        # with_lmn_factors=True the (l, m, n) angular momentum vector dependent
        # factors are multiplied onto the C2S-coefficients. Please see
        # '$sympleints_root/ressources/cgto_normalization.ipynb' how normalization
        # is handled in sympleints.
        factors, cart_lmns = expand_sph_quantum_numbers(Lm, with_lmn_factors=True)
        rhs_ams = list()
        for cart_lmn in cart_lmns:
            cart_am = CartAngMom(*cart_lmn)
            rhs_ams.append(cart_am)
        c2s_map[sph_am] = (factors, rhs_ams)
    return c2s_map


class Cart2Sph(Transform):
    is_c2s = True

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Create C2S map of correct length and kinds
        self._map = get_c2s_map(self.L_target)

    def apply(self, ang_moms, reduce_index=None, drop_none=True):
        # Compared to actual recursion relations the Cartesian->Spherical transformation
        # is unambiguous.
        # Just select from precomputed maps according to total angular momentum at the
        # center to be transformed.

        # Only try to transform spherical indices. Return early otherwise.
        if not ang_moms.is_spherical(self.center_index):
            return list(), reduce_index
        sph_ang_mom = ang_moms[self.center_index]
        _, cart_ang_moms = self._map[sph_ang_mom]
        mod_ang_moms = list()
        for cart_ang_mom in cart_ang_moms:
            tmp_ang_mom = ang_moms.with_ang_mom_at(cart_ang_mom, self.center_index)
            mod_ang_moms.append(tmp_ang_mom)
        return mod_ang_moms, reduce_index

    def get_expr(self, ang_moms, pos, compr_arr_map, index_map):
        # Left hand side of the assignment
        lhs = compr_arr_map[ang_moms.as_key()][index_map[ang_moms]]

        sph_ang_mom = ang_moms[self.center_index]
        factors, cart_ang_moms = self._map[sph_ang_mom]

        summands = list()
        for fact, cam in zip(factors, cart_ang_moms):
            source_ang_moms = ang_moms.with_ang_mom_at(cam, self.center_index)
            summand = (
                fact
                * compr_arr_map[source_ang_moms.as_key()][index_map[source_ang_moms]]
            )
            summands.append(summand)
        rhs = sum(summands)
        assignment = Assignment(lhs, rhs)
        return assignment

    def target_node_iter(self, G, key_index_map):
        # Only generate expressions for nodes that are spherical at the given index
        sph_keys = [
            key for key in key_index_map if key[self.center_index].kind == BFKind.SPH
        ]
        for key in sph_keys:
            yield from key_index_map[key]
