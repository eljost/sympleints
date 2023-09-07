from pathlib import Path
import networkx as nx


def dump_graph(G, key, out_dir=None):
    """Dump graph G to PNG and SVG."""
    if out_dir is None:
        out_dir = Path(".")

    AG = nx.nx_agraph.to_agraph(G)
    AG.graph_attr.update(rankdir="RL")
    AG.layout("dot")
    for ext in ("png", "svg"):
        fn = f"{key}.{ext}"
        AG.draw(out_dir / fn)
        print(f"Wrote graph to '{fn}'")
    return G
