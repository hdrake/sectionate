"""
Walking a section across a grid's physical corner points.

The walk moves on the corner *node* graph from `sectionate.topology`, not on native
`(face, j, i)` indices. That is what makes it grid-agnostic: a periodic wrap, a
bipolar fold seam and a rotated tile boundary are all just edges, and the walk needs
no notion of "right/left/up/down", no wrap-around arithmetic, and no test for whether
two indices happen to be the same corner. There are no seam twins to admit, because
in node space a seam has no twins.

Native indices come back at the very end, in `native_path`, which is a projection
rather than a step of the algorithm.
"""

import numpy as np


def find_closest_corner(lon, lat, topology):
    """
    The corner node nearest to a geographic position.

    Ties are broken by the node's canonical native index, so a grid that stores one
    position for many corners -- a bipolar cap's singular meridian -- snaps to a
    definite one of them rather than to whichever the search happened to reach first.
    """
    from .section import distance_on_unit_sphere

    nlon, nlat, known = topology._positions()
    d = np.where(known, distance_on_unit_sphere(nlon, nlat, lon, lat), np.inf)
    best = np.flatnonzero(d == d.min())
    if best.size == 1:
        return int(best[0])
    nat = topology.node_native[best]
    order = np.lexsort((nat[:, 2], nat[:, 1], nat[:, 0]))
    return int(best[order[0]])


def infer_grid_path(node1, node2, topology, curve="great circle"):
    """
    The chain of corner nodes that best approximates `curve` between two corners.

    Three rules, in order, and none of them knows what kind of grid it is on:

    1. enumerate the current corner's neighbours, dropping the one just came from;
    2. admit those that get strictly closer to the end point;
    3. among the admitted, step to the one lying closest to the requested curve,
       breaking near-ties deterministically so the path does not depend on the
       platform or on which end you start from.

    Corners that are distinct but coincident are handled as a group (see
    `CornerTopology.plateau_labels`): the walk decides where to *leave* such a group
    by looking at the neighbours outside it, then crosses it by the shortest chain.
    Without that, the metric is flat across the group and the route through it would
    be settled by node numbering -- which on a tripolar grid's polar fan decides
    which sector a closed section encloses.

    Parameters
    ----------
    node1, node2 : int
        Start and end corner nodes.
    topology : sectionate.topology.CornerTopology
    curve : str
        The curve this segment follows; see `sectionate.grid_section`.

    Returns
    -------
    list of int
        Corner nodes from `node1` to `node2` inclusive, each adjacent to the next.
    """
    from .section import (
        WALK_DEVIATION_ATOL, distance_on_unit_sphere, spherical_angle,
    )

    lon, lat, known = topology._positions()
    plateau = topology.plateau_labels()
    if not (known[node1] and known[node2]):
        raise ValueError(
            "A section endpoint snapped to a corner whose position is unknown. That "
            "happens only for a corner stored on no face and surrounded by fewer "
            "than three tracer cells, on the outermost row of a staggered grid."
        )

    lon1, lat1 = lon[node1], lat[node1]
    lon2, lat2 = lon[node2], lat[node2]

    # Same metrics as ever: `progress` says whether a candidate gets us nearer the
    # end point, `deviation` how far it strays from the requested curve. Both are
    # symmetric in the two end points, so the path does not depend on travel
    # direction, and both are in radians so one tie tolerance suits either curve.
    if curve == "great circle":
        def progress(n):
            return distance_on_unit_sphere(lon[n], lat[n], lon2, lat2)

        def deviation(n):
            return (spherical_angle(lon2, lat2, lon1, lat1, lon[n], lat[n])
                    + spherical_angle(lon1, lat1, lon2, lat2, lon[n], lat[n]))
    else:
        def progress(n):
            return np.sin(np.deg2rad((lon[n] - lon2) / 2.0)) ** 2

        def deviation(n):
            return (np.deg2rad(abs(lat[n] - lat1))
                    + np.deg2rad(abs(lat[n] - lat2)))

    def order_key(n):
        f, j, i = topology.node_native[n]
        return (int(f), int(j), int(i), int(n))

    members = _plateau_members(topology, plateau)

    path = [int(node1)]
    here, came_from = int(node1), -1
    # Enough steps to cross every corner of the grid once; a walk that needs more
    # is not converging and should say so rather than spin.
    nstep_max = topology.n_nodes + 1
    steps = 0

    while here != node2:
        if steps > nstep_max:
            raise RuntimeError(
                "The section walk did not reach its end point. This means the two "
                "end points are not connected through the grid's corner graph -- "
                "check that the grid's `face_connections` and axis boundary "
                "conditions describe the topology you expect."
            )
        steps += 1

        group = members[plateau[here]]
        if node2 in group:
            path.extend(_shortest_chain(topology, here, node2, group)[1:])
            break

        if len(group) == 1:
            cands = [
                int(m) for m in topology.neighbors(here)
                if m != came_from and known[m]
            ]
            step_from = here
        else:
            # Leave the group by whichever outside neighbour is best; the entry and
            # exit are what matter, and the chain between them is just bookkeeping.
            cands, exit_via = [], {}
            for via in group:
                for m in topology.neighbors(via):
                    m = int(m)
                    if plateau[m] == plateau[here] or not known[m]:
                        continue
                    if m not in exit_via:
                        exit_via[m] = int(via)
                        cands.append(m)
            step_from = None

        if not cands:
            raise RuntimeError(
                "The section walk reached a corner with nowhere to go: every "
                "neighbour is the corner it came from, or has no known position."
            )

        p_here = progress(here)
        admitted = [m for m in cands if progress(m) < p_here]
        if admitted:
            devs = [(deviation(m), m) for m in admitted]
            devs = [(d, m) for d, m in devs if np.isfinite(d)] or devs
            best = min(d for d, _ in devs)
            nxt = min(
                (m for d, m in devs if d <= best + WALK_DEVIATION_ATOL),
                key=order_key,
            )
        else:
            # Nothing gets closer (closing a fold, or hemmed in by a wall): take the
            # neighbour nearest the end point instead.
            nxt = min(
                cands,
                key=lambda m: (
                    float(distance_on_unit_sphere(lon[m], lat[m], lon2, lat2)),
                    order_key(m),
                ),
            )

        if step_from is None:
            via = exit_via[nxt]
            path.extend(_shortest_chain(topology, here, via, group)[1:])
            step_from = via
        path.append(nxt)
        came_from, here = step_from, nxt

    return path


def _plateau_members(topology, plateau):
    """Nodes of each plateau, as a dict keyed by plateau label."""
    cache = getattr(topology, "_plateau_members", None)
    if cache is not None:
        return cache
    order = np.argsort(plateau, kind="stable")
    labels = plateau[order]
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, labels.size]
    out = {}
    for lo, hi in zip(starts[:-1], starts[1:]):
        out[int(labels[lo])] = [int(n) for n in order[lo:hi]]
    topology._plateau_members = out
    return out


def _shortest_chain(topology, start, end, allowed):
    """Fewest-edges path from `start` to `end` staying inside `allowed`."""
    if start == end:
        return [start]
    allowed = set(allowed)
    prev = {start: None}
    queue = [start]
    while queue:
        nxt = []
        for n in queue:
            for m in topology.neighbors(n):
                m = int(m)
                if m in prev or m not in allowed:
                    continue
                prev[m] = n
                if m == end:
                    chain = [m]
                    while prev[chain[-1]] is not None:
                        chain.append(prev[chain[-1]])
                    return chain[::-1]
                nxt.append(m)
        queue = nxt
    raise RuntimeError(
        f"Corners {start} and {end} sit at the same place but are not connected "
        "through the corners that share it."
    )


def native_path(topology, nodes):
    """
    Write a chain of corner nodes as native `(face, j, i)` indices.

    A seam crossing has to change frame somewhere: the corner it crosses at is
    stored twice, once on each side, and only one of those is index-adjacent to what
    comes next. So where consecutive nodes have no index-adjacent pair of
    representations, the shared corner is emitted *twice* -- once in each frame --
    and the step between the two copies spans no cell and carries no velocity face.

    Doing this while writing the path out, rather than repairing the indices
    afterwards, is what makes it correct in general. A section that runs *along* a
    seam for a while visits several corners that are each stored twice, and there is
    no way to tell from the index list alone where the frame change belongs; here
    the walk still knows.

    Returns
    -------
    (corners, node_of_corner) : (np.ndarray of shape (n, 3), np.ndarray)
        Native `(face, j, i)` per emitted corner, and the node each one is a
        representation of.
    """
    reps = [_native_reps(topology, n) for n in nodes]
    for n, r in zip(nodes, reps):
        if len(r) == 0:
            topology.native_of([n])       # raises, with the position in the message

    out, out_nodes = [], []
    cur = None
    for k, (n, cands) in enumerate(zip(nodes, reps)):
        if cur is None:
            nxt = reps[k + 1] if k + 1 < len(reps) else None
            cur = _pick_first(cands, nxt)
            out.append(cur)
            out_nodes.append(n)
            continue
        adj = [r for r in cands if _index_adjacent(cur, r)]
        if adj:
            cur = adj[0]
        else:
            # No step in index space from where we stand. If the corner we are
            # standing on has another representation from which there is one, emit
            # that too: the crossing is then written as an ordinary step plus a
            # hand-off between two spellings of one corner, which spans no cell.
            prev_node = out_nodes[-1]
            switch = None
            for r_prev in _native_reps(topology, prev_node):
                for r in cands:
                    if _index_adjacent(r_prev, r):
                        switch, cur = r_prev, r
                        break
                if switch is not None:
                    break
            if switch is not None:
                out.append(switch)
                out_nodes.append(prev_node)
            else:
                # A staggered grid does not store the corner on the face carrying
                # this face's velocity, so no two of their indices are adjacent.
                # The step is still a real edge -- the node ids say so, and that is
                # what the face attribution reads -- it simply changes frame.
                nxt = reps[k + 1] if k + 1 < len(reps) else None
                cur = _pick_first(cands, nxt)
        out.append(cur)
        out_nodes.append(n)

    return np.asarray(out, dtype=np.int64), np.asarray(out_nodes, dtype=np.int64)


def _native_reps(topology, node):
    """Representations of `node` that a native index array actually stores."""
    t, nyq, nxq = topology.t, topology.nyq, topology.nxq
    reps = topology.reps_of(node)
    f, J, I = reps[:, 0], reps[:, 1], reps[:, 2]
    ok = (J >= t) & (J < t + nyq) & (I >= t) & (I < t + nxq)
    return [(int(a), int(b) - t, int(c) - t) for a, b, c in reps[ok]]


def _index_adjacent(a, b):
    return a[0] == b[0] and abs(a[1] - b[1]) + abs(a[2] - b[2]) == 1


def _pick_first(cands, nxt):
    """Start in a frame the next corner can be reached from, where there is one."""
    if nxt:
        for r in cands:
            if any(_index_adjacent(r, s) for s in nxt):
                return r
    return cands[0]
