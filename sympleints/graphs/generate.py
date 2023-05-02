from collections.abc import Sequence
from dataclasses import dataclass
import itertools as it
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
from sympy import IndexedBase
from sympy.codegen.ast import Assignment


from sympleints.graphs.AngMoms import AngMoms, LKind
from sympleints.graphs.Integral import Integral
from sympleints.graphs.merge_exprs import merge_expressions
from sympleints.graphs.optimize import get_G_fn, opt_integral_transforms
from sympleints.helpers import get_order_funcs_for_kinds


def get_index_map_for_graph(
    G,
    order: Sequence[int],
):
    """Map 2d/3d ang_moms to flat 1d index.

    This is where the magic happens.
    """

    nodes_for_key = dict()
    for node in G.nodes():
        nodes_for_key.setdefault(node.as_key(drop_aux=False), list()).append(node)
    keys = nodes_for_key.keys()

    index_map = dict()
    key_index_map = dict()
    size_map = dict()

    for key in keys:
        key_index_map[key] = list()
        _lkinds = [entry for entry in key if isinstance(entry, LKind)]
        *lkinds, aux_n = key
        assert len(lkinds) == len(_lkinds)
        L_tots = [lkind.L for lkind in lkinds]
        kinds = [lkind.kind for lkind in lkinds]

        order_funcs = get_order_funcs_for_kinds(kinds)

        orders = [ord_func(L) for ord_func, L in zip(order_funcs, L_tots)]
        orders = [orders[i] for i in order]
        org_order = [0] * len(order)
        for i, index in enumerate(order):
            org_order[index] = i

        i = 0
        all_ang_moms = list(it.product(*orders))
        all_nodes = [AngMoms.from_iterables(am + ((aux_n,),)) for am in all_ang_moms]
        present_nodes = nodes_for_key[key]
        # Loop over all nodes for given key in graph
        for node in all_nodes:
            node_org_order = node.with_order(org_order)
            if node_org_order in present_nodes:
                index_map[node_org_order] = i
                i += 1
                key_index_map.setdefault(key, list()).append(node_org_order)
        # Store array size
        size_map[key] = i
    return index_map, key_index_map, size_map


def resort_array(source_arr, source_arr_index_map, target_index_map, work_arr):
    assignments = list()
    # Store copy of source array
    for i in range(len(source_arr_index_map)):
        assignments.append(Assignment(work_arr[i], source_arr[i]))
    # Current order in aim
    for i, lmn in enumerate(source_arr_index_map):
        index = target_index_map[lmn]
        assignments.append(Assignment(source_arr[index], work_arr[i]))
    return assignments


def array_names_from_G(G):
    ang_mom_map = dict()
    name_map = dict()
    for node in G.nodes():
        key = node.as_key(drop_aux=False)
        name = node.name()
        ang_mom_map.setdefault(key, set()).add(name)
        name_map[name] = key  # node.as_key(drop_aux=False)
    return ang_mom_map, name_map


def source_node_iter(G):
    for node, degree in G.out_degree():
        if degree == 0:
            yield node


@dataclass
class GeneratedIntegral:
    integral: Integral
    L_tots: Tuple[int, ...]
    array_defs: Dict
    assignments: List
    shell_size: int
    target_array_name: str

    @property
    def key(self):
        return "".join(map(str, self.L_tots))


def generate_integral(L_tots: Sequence[int], integral: Integral) -> GeneratedIntegral:
    key = "".join(map(str, L_tots))

    base = Path(".")
    fns = [base / get_G_fn(integral, transf, key) for transf in integral.rrs]
    for fn in fns:
        if not fn.exists():
            opt_integral_transforms(L_tots, integral, with_aux=integral.with_aux)
            break

    array_defs = dict()
    compr_arr_map = dict()

    def make_array(name, size, compressed):
        if name in array_defs:
            ref_arr = array_defs[name]
            assert (
                size == ref_arr.shape[0]
            ), "Tried to recreate array with different size!"
            arr = ref_arr
            assert compr_arr_map[compressed] is ref_arr
        else:
            arr = IndexedBase(name, shape=size)
            array_defs[name] = arr
            compr_arr_map[compressed] = arr
        return arr

    def make_arrays(keys_names, name_map, size_map):
        for key, names in keys_names.items():
            size = size_map[key]
            for name in names:
                make_array(name, size, name_map[name])

    prev_order = None
    prev_key_index_map = None
    max_tmp_size = 0
    resorted = False
    assignments = dict()

    zip_ = list(zip(integral.rrs, fns))
    for i, (transform, G_fn) in enumerate(zip_):
        print(f"Processing '{transform.name}'")
        G = nx.read_gexf(G_fn, node_type=AngMoms.from_str)
        print("\t ... read graph from file")

        names, name_map = array_names_from_G(G)
        index_map, key_index_map, size_map = get_index_map_for_graph(
            G,
            order=transform.order,  # filter=transform.center_index
        )
        make_arrays(names, name_map, size_map)
        # Storage for the assignments of the current transformation
        cur_assignments = list()

        # When the order changes between two transformations arrays have to be resorted.
        if (prev_order is not None) and prev_order != transform.order:
            resorted = True
            # Determine source nodes of current transformation ...
            source_nodes = list(source_node_iter(G))
            # ... and group them by the array they belong to (according to their key).
            source_nodes_by_key = dict()
            for sn in source_nodes:
                source_nodes_by_key.setdefault(sn.as_key(), list()).append(sn)

            # Resort source arrays one by one
            for key in source_nodes_by_key.keys():
                nodes = key_index_map[key]
                # Update size of temporary array. The actual array will be created
                # at the end, when the required maximum size is known.
                max_tmp_size = max(max_tmp_size, len(nodes))
                # Determine sympy.IndexedBase array object that belongs to the given key
                source_arr = compr_arr_map[key]
                # Original node ordering that we want to modify
                arr_inds = prev_key_index_map[key]
                cur_assignments.extend(
                    resort_array(
                        source_arr,
                        source_arr_index_map=arr_inds,
                        target_index_map=index_map,
                        work_arr=IndexedBase("tmp", shape=(max_tmp_size,)),
                    )
                )

        # Generate base expressions in the first cycle
        if i == 0:
            for node in source_node_iter(G):
                expr = integral.get_base_expr(node, compr_arr_map, index_map)
                cur_assignments.append(expr)
        # Generate expressions for all target nodes in transformation
        for node in transform.target_node_iter(G, key_index_map):
            try:
                pos = G.nodes[node]["pos"]
            except KeyError:
                pos = None  # Dummy for C2S, where no pos is present
            # TODO: also provide previous node, to compare indices and check if both
            # expressions can be merged.
            expr = transform.get_expr(node, pos, compr_arr_map, index_map)
            cur_assignments.append(expr)
        # Merge expressions for more succinct code. This is also where the magic happens.
        cur_assignments = merge_expressions(cur_assignments, indexed_filter="work_|tmp")
        assignments[transform.name] = cur_assignments
        prev_order = transform.order
        prev_key_index_map = key_index_map

    # After all transformations are carried out, we know the maximum required size
    # for the tmp array.
    if resorted:
        make_array("tmp", max_tmp_size, compressed="tmp")

    gen_integral = GeneratedIntegral(
        integral=integral,
        L_tots=tuple(L_tots),
        array_defs=array_defs,
        assignments=assignments,
        shell_size=integral.shell_size,
        target_array_name=integral.target_name,
    )
    return gen_integral
