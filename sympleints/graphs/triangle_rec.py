from dataclasses import dataclass
from collections.abc import Sequence
import functools
import json
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from sympleints.helpers import canonical_order
from sympleints.logger import logger

TPoint = Tuple[int, int, int]


def get_plane(L: int) -> NDArray[int]:
    return np.array(canonical_order(L), dtype=int)


def plot(Ls: Optional[Sequence[int]] = None, L_max: Optional[int] = None):
    """Basic 3d plot of planes with given Ls."""
    if Ls is None:
        Ls = range(5)
    points = list()
    for L in Ls:
        points.extend(get_plane(L))
    points = np.array(points, dtype=int)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.view_init(elev=45.0, azim=45.0)
    ax.scatter(
        *points.T,
        s=85,
        depthshade=False,
    )
    if L_max is not None:
        ax.set_xlim(0, L_max)
        ax.set_ylim(0, L_max)
        ax.set_zlim(0, L_max)
    ax.set_xlabel("$l_x$")
    ax.set_ylabel("$l_y$")
    ax.set_zlabel("$l_z$")
    return fig, ax


# At most 6 points can be generated from a given point.
# These are actually 3 steps in pos. and neg. direction.
strategy = np.array(
    (
        (1, 1, -2),
        (-1, -1, 2),
        #
        (2, -1, -1),
        (-2, 1, 1),
        #
        (1, -2, 1),
        (-1, 2, -1),
    )
)


def visit_from(start: TPoint):
    """Get set of points reachable from 'start' in the same plane."""
    new_points = start + strategy
    new_points = new_points[np.all(new_points >= 0, axis=1)]
    return set(map(tuple, new_points))


@functools.singledispatch
def points_below(points: Sequence[TPoint], initial=False):
    """Generate points in plane below, reachable from points in plane above.

    If all points in plane L should be checked it is more convenient to just call this
    function with an integer, instead of an sequence. Then, all points in L are generated
    automatically."""
    Ls = [sum(point) for point in points]
    L0 = Ls[0]
    assert all([Li == L0 for Li in Ls])

    # Initial step from L to L-1
    if L0 > 1:
        step_down = (-2, 1, 0)
    else:
        step_down = (-1, 0, 0)
    assert sum(step_down) == -1
    step_down = np.array(step_down, dtype=int)

    # Using all points for the initial step down does not seem to help ...
    if initial:
        start_points = canonical_order(L0 - 1)
    else:
        start_points = [lmn + step_down for lmn in points]
        start_points = [start for start in start_points if all(start >= 0)]

    for start in start_points:
        visited = {
            tuple(start),
        }
        to_visit = visit_from(start)
        # Loop until there are no new points to be visited
        while to_visit:
            start = to_visit.pop()
            visited.add(start)
            to_visit.update(visit_from(start) - visited)
        below = np.array(list(visited), dtype=int)
        yield below


@points_below.register
def _(L: int, **kwargs):
    """Generated all points in plane L."""
    points = canonical_order(L)
    yield from points_below(points, **kwargs)


def points_reachable(points: Sequence[TPoint]) -> set:
    """Set of reachable points in plane above the plane of the given points."""
    reachable = set()
    for point in points:
        x, y, z = point
        # We can go up either in x, y or z-direction
        reachable |= {(x + 1, y, z), (x, y + 1, z), (x, y, z + 1)}
    return reachable


blw_kwargs = {
    "s": 110,
    "depthshade": False,
}
trg_kwargs = {
    "s": 135,
    "depthshade": False,
}


@dataclass
class PlanePoints:
    L: int
    points: Sequence[TPoint]
    target_points: Optional[Sequence[TPoint]] = None
    children: Optional[List] = None

    @property
    def edges(self) -> set:
        L = self.L
        return {edge for edge in ((L, 0, 0), (0, L, 0), (0, 0, L)) if min(edge) >= 0}

    def plot(self, L_target, L_max=None, prefix=""):
        L = self.L
        fig, ax = plot((L + 1,), L_max=L_max)
        arr = np.array(tuple(self.points), dtype=int)
        ax.scatter(
            *arr.T,
            c="white",
            label=f"Points$_{{L={L}}}$",
            edgecolor="red",
            linewidth=2,
            **blw_kwargs,
            # s=125,
        )
        target_str = ""
        if self.target_points is not None:
            target_arr = np.array(tuple(self.target_points), dtype=int)
            ax.scatter(
                *target_arr.T,
                c="green",
                label=f"Points$_{{L={L+1}}}$",
                **trg_kwargs,
            )
            target_str = f" {len(target_arr)} target points,"
        ax.legend()
        title = (
            f"Target L={L_target}, L={L+1},{target_str} {len(self.points)} points below"
        )
        fig.suptitle(title)
        fig.savefig(f"{prefix}target_L{L_target}_level_L{L+1}.png")
        fig.tight_layout()
        plt.close()


# def required_points_below(L: int, points_L: Sequence[TPoint]):
def required_points_below(parent_plane: PlanePoints):
    # Base case
    if parent_plane.L == 0:
        return []

    unique_below = set()
    points_L = parent_plane.points
    L = parent_plane.L

    # Reachable points from plane L
    initial = parent_plane.target_points == None
    for below_L in points_below(points_L, initial=initial):
        below_set = frozenset(map(tuple, below_L))
        unique_below.add(below_set)
        # pp = PlanePoints(L - 1, below_set)  # TODO: del
        # pp.plot(L_target=L, L_max=4)  # TODO: del
    logger.debug(f"\t... found {len(unique_below)} variant(s)")

    # Add the triangle corners from L-1, as they are always required.
    tmp = set()
    edges_below = {
        edge for edge in ((L - 1, 0, 0), (0, L - 1, 0), (0, 0, L - 1)) if min(edge) >= 0
    }
    # Augment every variant with edges
    for below_L in unique_below:
        below_L = frozenset(below_L | edges_below)
        tmp.add(below_L)
    if len(unique_below) == 0:
        tmp = {frozenset(edges_below)}
    # tmp will at least contain the edges now.
    unique_below = tmp

    # Determine reachable points in plane L from found points in plane L - 1
    # and augment L - 1 point if not all points in L are reachable.
    #
    # points in plane L.
    in_L = set(points_L)
    tmp = set()
    for i, below_L in enumerate(unique_below):
        below_L = set(below_L)
        reachable = points_reachable(below_L)
        unreachable = in_L - reachable
        was_unreachable = bool(unreachable)
        logger.debug(f"\t... variant {i}: {len(unreachable)} unreachable points")
        while unreachable:
            # Select one point that we want to reach in plane L
            to_reach = unreachable.pop()
            # Determine first non-zero index, along which we can step down
            first_non_zero = [j for j, am in enumerate(to_reach) if am > 0][0]
            reach_from = np.array(to_reach)
            reach_from[first_non_zero] -= 1
            # Add new point in plane L-1 to our set and recalculate unreachable points
            below_L.add(tuple(reach_from))
            unreachable = in_L - points_reachable(below_L)
        if was_unreachable:
            logger.debug(f"\t\t... augmented variant {i}")
        tmp.add(frozenset(below_L))
    if tmp:
        unique_below = tmp

    # Sometimes we can further prune the points. As we use the proposed step down of
    # (-2, 1, 0) we sometimes step away too far.
    tmp = set()
    for below_L in unique_below:
        prune_check = below_L.copy()
        to_prune = set()
        for point in prune_check:
            tmp_copy = set(below_L)
            tmp_copy.remove(point)
            reachable = points_reachable(tmp_copy)
            # If in_L is fully contained in reachable we can prune the point
            if reachable >= in_L:
                to_prune.add(point)
                below_L = tmp_copy
        tmp.add(frozenset(below_L))
    unique_below = tmp

    # A last check that all points are reachable. TODO: the reachable check is already
    # done in the pruning. Maybe put the printed message somewhere else.
    n_Lminus = len(canonical_order(L))
    for i, below_L in enumerate(unique_below):
        assert len(in_L - points_reachable(below_L)) == 0
        ratio = len(below_L) / n_Lminus
        logger.debug(
            f"\t... variant {i} requires only {ratio:.2%} of all points in plane {L-1}"
        )
    unique_below = tuple(unique_below)
    min_len = min(map(len, unique_below))
    logger.debug(f"\t... minimum number of required points is {min_len}")
    unique_below_min_len = list(filter(lambda arg: len(arg) == min_len, unique_below))
    plane_points = [
        PlanePoints(L=L - 1, points=below_L, target_points=points_L)
        for below_L in unique_below_min_len
    ]
    for pp in plane_points:
        pp.children = required_points_below(pp)
    return plane_points


def min_cost_path(node):
    if not node.children:
        return [node], len(node.points)
    else:
        # list of (path, cost) pairs
        options = [min_cost_path(c) for c in node.children]
        path, cost = min(options, key=lambda option: option[1])
        path.append(node)
        return path, cost + len(node.points)


def get_pos_dict(L_max, strict=False):
    reduce_at_pos = dict()
    for L_target in range(0, L_max + 1):
        # for L_target in (L_max,):
        logger.debug(f"Processing target L={L_target}")
        points = canonical_order(L_target)
        parent_plane = PlanePoints(L_target, points)
        child_planes = required_points_below(parent_plane)
        if len(child_planes) == 0:
            continue
        logger.debug(
            f"Found {len(child_planes)} variants in L={L_target-1} for L_target={L_target}."
        )
        paths, costs = zip(*[min_cost_path(cp) for cp in child_planes])
        min_cost = min(costs)
        logger.debug(f"Minimum cost is {min_cost}")
        for i, path in enumerate(paths):
            # Plot first path with minimum cost
            if costs[i] == min_cost:
                # From small L to bigger L
                for p in path:
                    # p.plot(L_target, L_max, prefix=f"path_{i}_")
                    logger.debug(
                        f"\tL={p.L+1}, ({len(p.target_points)}, {len(p.points)})"
                    )
                aug_min_path = [
                    parent_plane,
                ] + path[::-1]
                break  # Only plot first path

        # Populate final table
        # Loop over all planes in minimum cost path
        tmp0 = dict()
        for p in aug_min_path:
            tmp1 = dict()
            # Loop over all points in current plane and determine, which points in the
            # upper plane are reachable.
            for point in p.points:
                arr = np.array(point, dtype=int)
                # Another approach would be to only include reachable points that are also
                # target points of a given plane; not all points in the target plane.
                # This would be much more strict, as requesting a position for node that
                # is not needed/not in the dictionary would produce a KeyError.
                reachable_from_point = points_reachable((point,))
                if strict:
                    target_points = set(p.target_points) if p.target_points else set()
                    reachable_from_point = reachable_from_point & target_points
                for pr in reachable_from_point:
                    # tmp[pr] = point
                    pos = int((pr - arr).argmax())
                    # tmp[pr] = (point, pos)
                    tmp1[pr] = pos
                    # key = ",".join(map(str, pr))  # JSON does not support tuple keys
                    # tmp1[key] = pos
            tmp0[p.L + 1] = tmp1
        reduce_at_pos[L_target] = tmp0
        del aug_min_path
    # TODO: flatten the final dict and remove the second level of 'L' keys?!
    # The angular mometa are unique so we don't need the different 'L' keys ...
    return reduce_at_pos


def convert_for_json(reduce_at_pos):
    converted = dict()
    for L_target, for_L in reduce_at_pos.items():
        tmp0 = dict()
        for L, am_map in for_L.items():
            tmp1 = dict()
            for key, pos in am_map.items():
                json_key = ",".join(map(str, key))  # JSON does not support tuple keys
                tmp1[json_key] = pos
            tmp0[L] = tmp1
        converted[L_target] = tmp0
    return converted


def dump_json(reduce_at_pos, out_fn="reduce_at_pos.json"):
    from pprint import pprint

    pprint(reduce_at_pos)
    for_json = convert_for_json(reduce_at_pos)
    pprint(for_json)
    with open(out_fn, "w") as handle:
        json.dump(for_json, handle)
    print(f"Dumped to '{out_fn}")


if __name__ == "__main__":
    L_max = 8
    reduce_at_pos = get_pos_dict(L_max)
    dump_json(reduce_at_pos)
