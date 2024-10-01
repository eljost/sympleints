from collections.abc import Sequence
import time
from typing import Dict, Optional, Tuple

import networkx as nx

from sympleints.graphs.AngMoms import AngMoms, AngMomMap, AuxIndex
from sympleints.graphs.helpers import dump_graph
from sympleints.graphs.Integral import Integral
from sympleints.graphs.Transform import Transform
from sympleints.helpers import shell_iter, get_path_in_cache_dir


def apply_transform(
    G: nx.DiGraph,
    ang_moms: AngMoms,
    transf: Transform,
    edge_attrs: Optional[Dict] = None,
) -> None:
    """Apply transformation to given node."""
    if edge_attrs is None:
        edge_attrs = transf.edge_attrs
    G.add_node(ang_moms)

    other_nodes, pos = transf.apply(ang_moms)
    if pos is not None:
        nx.set_node_attributes(G, values={ang_moms: {"pos": pos}})
    for other_node in other_nodes:
        G.add_edge(ang_moms, other_node, **edge_attrs)
        # Try to further reduce the expression
        apply_transform(G, other_node, transf, edge_attrs=edge_attrs)


def get_shell_graph(
    L_tots: Tuple[int, ...],
    transf: Transform,
    with_aux: bool = False,
    edge_attrs: Optional[Dict] = None,
) -> nx.DiGraph:
    """Generate all nodes in a given shell and apply a transformation to them.

    Thin wrapper around 'apply_transfrom'."""
    G = nx.DiGraph()
    edge_attrs = transf.edge_attrs

    for raw_ang_moms in shell_iter(L_tots, kinds=transf.kinds):
        # Create AngMoms objects with correct BFKinds
        args = [AngMomMap[kind](*am) for kind, am in zip(transf.kinds, raw_ang_moms)]
        if with_aux:
            args += [AuxIndex(0)]
        ang_moms = AngMoms(args)
        apply_transform(G, ang_moms, transf, edge_attrs=edge_attrs)
    return G


def graphs_from_transforms(
    L_tots: Tuple[int, ...], transforms: Sequence[Transform], with_aux: bool = False
) -> Dict[str, nx.DiGraph]:
    """Successively apply multiple transformations to produce a shell."""
    names = [transf.name for transf in transforms]
    name0, *rest_names = names
    transf0, *rest_transforms = transforms
    G = get_shell_graph(
        L_tots,
        transf0,
        with_aux=with_aux,
    )
    results = {
        name0: G,
    }
    prev_rr_sources = [n for n, d in G.out_degree() if d == 0]
    for name, transf in zip(rest_names, rest_transforms):
        G_tmp = nx.DiGraph()
        for ang_moms in prev_rr_sources:
            apply_transform(
                G_tmp,
                ang_moms,
                transf,
            )
        results[name] = G_tmp
        G = nx.compose(G_tmp, G)
        prev_rr_sources = [n for n, d in G.out_degree() if d == 0]
    results["total"] = G
    return results


def get_G_fn(integral, transf, key):
    return f"{integral.name}_G_opt_{key}_{transf.name}.xml.tar.gz"


def opt_integral_transforms(
    L_tots: Tuple[int, ...],
    integral: Integral,
    with_aux: bool,
    max_cycles: int = 50,
    do_plot=False,
) -> None:
    """Try to determine the graph w/ smallest number of intermediates by brute force."""
    assert max_cycles > 0
    min_ind = 1e8  # High number to start with
    min_counter = 0
    # G_min = None
    min_results = dict()
    lkey = "".join(map(str, L_tots))
    start = time.time()
    transforms = integral.opt_transforms
    for i in range(max_cycles):
        results = graphs_from_transforms(L_tots, transforms, with_aux=with_aux)
        G = results["total"]
        if len(G) < min_ind:
            print(
                f"Found new minimum graph with size {len(G)} in cycle {i+1}/{max_cycles}."
            )
            # dump_graph(G, f"min_{min_counter}")
            min_ind = len(G)
            # G_min = G
            min_results = results
            min_counter += 1
    dur = time.time() - start
    cycle_dur = dur / max_cycles
    print(f"{cycle_dur:.2f} s / cycle")
    # dump_graph(G_min, "G_min")
    # nx.write_gexf(G_min, "G_min.xml")
    # for rr_key, G in min_results.items():
    for transf in transforms:
        G = min_results[transf.name]
        fn = get_G_fn(integral, transf, lkey)
        # TODO: remove special characters, e.g., parantheses
        nx.write_gexf(G, get_path_in_cache_dir(fn))
        print(f"Wrote '{fn}'.")
    if do_plot and max(L_tots) <= 3:
        dump_graph(min_results["total"], f"G_min_{lkey}")
    else:
        print(f"Skipped image generation for '{L_tots}'")
