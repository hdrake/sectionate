"""
The physical corner-point topology of a structured grid.

A C-grid stores its vorticity ("corner") points in per-face index arrays, and one
*physical* corner can appear in several of them: the two ends of a periodic axis,
the mirrored columns of a bipolar fold seam, a corner shared by two tiles. Any code
that attributes transports, or that decides whether two steps of a section are the
same point, needs to know which indices denote the same corner.

This module answers that question **from the grid's declared metadata alone**.

Every identification a structured grid can express is an *affine integer map* on the
corner lattice, uniform along the whole seam it belongs to:

    periodic X (Nxc cells)   (J, I)   == (J, I + Nxc)
    bipolar fold on Y        (Nyc, I) == (Nyc, mirror(I))
    a `face_connections` gluing        an affine map onto the neighbouring face's
                                       seam line, carrying its rotation/reversal

The corner topology is then simply the quotient of the corner lattice by the group
these maps generate: a *node* is one physical corner, and every slot identified with
it is one of its representations. Nothing is inferred from coordinates, so two
corners that happen to coincide geographically -- the 41 rows of a tripolar grid's
bipolar pole, say -- stay distinct if the metadata says they are distinct, and a
corner whose stored coordinates are wrong is still identified correctly.

Working in this currency has a second payoff: an identification that
`face_connections` cannot express (its schema maps a whole tile edge to a whole tile
edge, so it cannot describe an edge glued to several partners over different index
ranges -- which is exactly the shape of the lat-lon-cap grid's southern boundary
fold) is still an ordinary affine map, and can simply be handed in. See
`Identification`.

The lattice used throughout is the **'outer'** one: `(Nyc + 1, Nxc + 1)` corner slots
per face, whatever the grid's native staggering, so that every corner of every cell
has a slot even where the native arrays drop a row. `native_of` maps a slot back to
the native index that stores it, where one does.
"""

import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

try:  # xgcm >= 0.10.1
    from xgcm.padding import _is_fold_padding, _seam_partner_indices, _resolve_pivot
except ImportError:  # pragma: no cover - keeps `import sectionate` working
    _is_fold_padding = _seam_partner_indices = _resolve_pivot = None

from .gridutils import corner_position, get_facedim, get_geo_corners


def _faced(da, dims, facedim):
    """`da`'s values with a leading face axis, real or synthetic."""
    if facedim is None:
        return da.transpose(*dims).values[None, ...]
    return da.transpose(facedim, ..., *dims).values



def _lonlat_to_xyz(lon, lat):
    """Positions on the unit sphere, so that comparing them is free of the
    longitude wrap and of the pole's degenerate longitudes."""
    lon = np.deg2rad(np.asarray(lon, dtype=float))
    lat = np.deg2rad(np.asarray(lat, dtype=float))
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


class Identification:
    """
    One declared statement that two lines of corner slots are the same points.

    Both lines are given as explicit slot-index arrays, so an identification is not
    restricted to whole tile edges and can express a partial gluing, a seam that
    reverses, or a line glued to itself.

    Parameters
    ----------
    a, b : array-like of (face, J, I)
        Corresponding corner slots on the 'outer' lattice. ``a[k]`` and ``b[k]`` are
        the same physical corner.
    name : str, optional
        Where this identification came from, used in error messages.
    """

    def __init__(self, a, b, name=""):
        self.a = np.asarray(a, dtype=np.int64).reshape(-1, 3)
        self.b = np.asarray(b, dtype=np.int64).reshape(-1, 3)
        if self.a.shape != self.b.shape:
            raise ValueError(
                f"Identification {name!r} pairs {self.a.shape[0]} slots with "
                f"{self.b.shape[0]}; the two sides must have the same length."
            )
        self.name = name

    def __repr__(self):
        return f"<Identification {self.name!r}: {self.a.shape[0]} slot pairs>"


def _seam_or_fill(padding):
    """
    The padding to use when a halo must contain a *real* neighbour or nothing.

    Only a genuine seam supplies a neighbouring cell: a periodic wrap, a bipolar
    fold, or a `face_connections` gluing (which `xgcm` applies regardless of this
    value). Every other boundary is a wall and is padded with NaN, so that nothing
    downstream mistakes a wall for a connection. In particular ``"extend"`` must not
    survive: it *replicates* the edge cell, inventing a neighbour where there is
    none.
    """
    if padding == "periodic":
        return "periodic"
    if _is_fold_padding is not None and _is_fold_padding(padding):
        return padding
    return "fill"



class CornerTopology:
    """
    The physical corner points of a grid, and how its stored corners map onto them.

    Built entirely from declared metadata -- each axis' `padding` (periodic wrap,
    bipolar fold, or wall), the grid's `face_connections`, and any explicit
    `identifications` handed in. Geographic coordinates are never consulted to decide
    identity; see `validate_positions` for the separate check that they *agree* with
    it.

    Parameters
    ----------
    grid : xgcm.Grid
    identifications : list of Identification, optional
        Extra declared identifications, for topology the grid's own metadata cannot
        express. Each is applied exactly as given.

    Attributes
    ----------
    n_nodes : int
        Number of distinct physical corner points.
    node_id : np.ndarray, shape (nf, nqy, nqx)
        The node each corner slot of the 'outer' lattice belongs to.
    node_native : np.ndarray, shape (n_nodes, 3)
        Canonical native `(face, j, i)` storing each node, or `(-1, -1, -1)` where the
        node is stored on no face.
    """

    def __init__(self, grid, identifications=None):
        self.grid = grid
        self.facedim = get_facedim(grid)
        self.single_tile = self.facedim is None
        self.pos = corner_position(grid)

        cd = _corner_coord_dict(grid)
        self.coords = cd
        ds = grid._ds
        Yq, Xq = cd["Y"]["corner"], cd["X"]["corner"]
        nyq, nxq = ds.sizes[Yq], ds.sizes[Xq]
        self.nyq, self.nxq = nyq, nxq

        # Native corner (j, i) lives at 'outer' slot (j + t, i + t): a 'right'
        # staggering drops the low corner row/column, 'left' the high one, and
        # 'outer' stores both.
        self.t = 1 if self.pos == "right" else 0
        # Cell counts follow from the corner array and the staggering.
        self.Nyc = nyq - 1 if self.pos == "outer" else nyq
        self.Nxc = nxq - 1 if self.pos == "outer" else nxq
        self.nqy, self.nqx = self.Nyc + 1, self.Nxc + 1
        self.nf = 1 if self.single_tile else ds.sizes[self.facedim]

        if self.t + nyq > self.nqy or self.t + nxq > self.nqx:
            raise ValueError(
                f"This grid declares its vorticity points at the {self.pos!r} "
                f"position, but its corner arrays are {nyq} x {nxq} against "
                f"{self.Nyc} x {self.Nxc} tracer cells. An 'outer' grid stores one "
                "more corner than cell on each axis, 'left' and 'right' exactly as "
                "many. Check that the corner coordinates and the tracer coordinates "
                "come from the same grid."
            )

        self._identifications = list(identifications or [])
        self._build()

    # ------------------------------------------------------------------ slots

    @property
    def n_slots(self):
        return self.nf * self.nqy * self.nqx

    def slot(self, f, J, I):
        """Flat index of corner slot `(f, J, I)` on the 'outer' lattice."""
        return (np.asarray(f) * self.nqy + np.asarray(J)) * self.nqx + np.asarray(I)

    def unslot(self, s):
        """Inverse of `slot`."""
        s = np.asarray(s)
        I = s % self.nqx
        J = (s // self.nqx) % self.nqy
        f = s // (self.nqx * self.nqy)
        return f, J, I

    # ------------------------------------------------------------- generators

    def _axis_generators(self):
        """
        Identifications declared by a single grid's own axis boundary conditions:
        a periodic wrap, and a bipolar fold.

        Only meaningful for a single-tile grid. On a multi-tile grid the per-axis
        padding applies to whichever face edges `face_connections` leaves unglued,
        and the connections themselves carry the topology, so the wrap is not a
        whole-lattice identification and is not applied here.
        """
        out = []
        if not self.single_tile:
            return out

        nqy, nqx, Nyc, Nxc = self.nqy, self.nqx, self.Nyc, self.Nxc
        J = np.arange(nqy)
        I = np.arange(nqx)

        for ax, n_cells, size in (("X", Nxc, nqx), ("Y", Nyc, nqy)):
            if ax not in self.grid.axes:
                continue
            padding = self.grid.axes[ax].padding
            if padding == "periodic":
                # The first and last lines of the lattice are one line of corners.
                if ax == "X":
                    a = np.stack([np.zeros_like(J), J, np.zeros_like(J)], -1)
                    b = np.stack([np.zeros_like(J), J, np.full_like(J, nqx - 1)], -1)
                else:
                    a = np.stack([np.zeros_like(I), np.zeros_like(I), I], -1)
                    b = np.stack([np.zeros_like(I), np.full_like(I, nqy - 1), I], -1)
                out.append(Identification(a, b, name=f"periodic {ax}"))

        out.extend(self._fold_generators())
        return out

    def _fold_generators(self):
        """
        The identification a declared bipolar fold makes along its seam row.

        The seam row is the *high* edge of the fold axis, and it is glued to itself:
        corner `I` is the same point as its mirror about the pole. The mirror is
        taken straight from `xgcm`'s own fold padding, so a fold means exactly the
        same thing here as it does when `xgcm` fills a halo across it.
        """
        out = []
        if _is_fold_padding is None:
            return out
        folds = getattr(self.grid, "_folds", None) or {}
        for fold_axis, info in folds.items():
            seam_axis = info["seam_axis"]
            pivot = _resolve_pivot(info["pivot"], fold_axis, seam_axis)
            if fold_axis == "Y" and seam_axis == "X":
                seam_len, fold_hi = self.nqx, self.nqy - 1
                mk = lambda K, M: (
                    np.stack([np.zeros_like(K), np.full_like(K, fold_hi), K], -1),
                    np.stack([np.zeros_like(M), np.full_like(M, fold_hi), M], -1),
                )
            elif fold_axis == "X" and seam_axis == "Y":
                seam_len, fold_hi = self.nqy, self.nqx - 1
                mk = lambda K, M: (
                    np.stack([np.zeros_like(K), K, np.full_like(K, fold_hi)], -1),
                    np.stack([np.zeros_like(M), M, np.full_like(M, fold_hi)], -1),
                )
            else:  # pragma: no cover - xgcm only builds X/Y folds
                raise NotImplementedError(
                    f"A fold on axis {fold_axis!r} with seam axis {seam_axis!r} is "
                    "not supported."
                )
            # The lattice is 'outer' whatever the native staggering, so the mirror
            # is the 'outer' one.
            partner = _seam_partner_indices("outer", pivot["seam"], seam_len)
            K = np.arange(seam_len)
            a, b = mk(K, partner.astype(K.dtype))
            out.append(Identification(a, b, name=f"fold {fold_axis}"))
        return out

    # ------------------------------------------- multi-tile seam identifications

    def _face_connection_generators(self):
        """
        The identification each `face_connections` gluing makes, read off the
        declaration itself.

        A gluing says four things: which face, which of its axes, which end of that
        axis, and whether the tangential index runs the same way. The first three are
        stated outright (`reverse` means the two faces meet on the *same* side of
        their axes, which is how `xgcm` checks the links back). The fourth follows
        from the first three, because a grid is a single oriented surface: crossing
        the seam flips the outward direction, and the rotation that does so carries
        the tangential direction with it. Nothing is measured, so a seam that is
        rotated, reversed, or runs into a wall or a pole is resolved the same way as
        any other.
        """
        if self.single_tile:
            return []

        fc = self.grid._face_connections[self.facedim]
        nqy, nqx = self.nqy, self.nqx

        def line(axis, side):
            """Corner slots along a face's edge, in tangential order."""
            if axis == "X":
                fixed = 0 if side == 0 else nqx - 1
                return np.arange(nqy), (lambda k: (k, fixed))
            fixed = 0 if side == 0 else nqy - 1
            return np.arange(nqx), (lambda k: (fixed, k))

        out = []
        for f, axes in fc.items():
            for axis, links in axes.items():
                for side, link in enumerate(links):
                    if link is None:
                        continue
                    f2, axis2, rev = link
                    # `reverse` glues the two faces on the same side of their axes.
                    side2 = side if rev else 1 - side
                    ks, slot = line(axis, side)
                    ks2, slot2 = line(axis2, side2)
                    if ks.size != ks2.size:
                        raise ValueError(
                            f"Face {f}'s {axis} edge is glued to face {f2}'s "
                            f"{axis2} edge, but they hold {ks.size} and {ks2.size} "
                            "corners. Two edges that meet must be the same length."
                        )
                    if not _tangential_is_forward(axis, side, axis2, side2):
                        ks2 = ks2[::-1]
                    a = np.array([(f,) + slot(int(k)) for k in ks], dtype=np.int64)
                    b = np.array([(f2,) + slot2(int(k)) for k in ks2], dtype=np.int64)
                    out.append(
                        Identification(a, b, name=f"face {f} {axis}[{side}] -> face {f2}")
                    )
        return out

    # ------------------------------------------------------------------ build

    def _build(self):
        gens = self._axis_generators()
        gens += self._face_connection_generators()
        gens += self._identifications
        self.identifications = gens

        n = self.n_slots
        if gens:
            rows = np.concatenate([self.slot(*g.a.T) for g in gens])
            cols = np.concatenate([self.slot(*g.b.T) for g in gens])
            bad = (rows < 0) | (rows >= n) | (cols < 0) | (cols >= n)
            if bad.any():
                k = int(np.flatnonzero(bad)[0])
                raise ValueError(
                    "An identification names a corner slot outside this grid's "
                    f"{self.nf} x {self.nqy} x {self.nqx} corner lattice "
                    f"(flat slot {int(rows[k])} <-> {int(cols[k])})."
                )
            graph = coo_matrix(
                (np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n)
            )
            self.n_nodes, labels = connected_components(
                graph, directed=False, return_labels=True
            )
        else:
            self.n_nodes, labels = n, np.arange(n, dtype=np.int64)

        self.labels = labels.astype(np.int64)
        self.node_id = self.labels.reshape(self.nf, self.nqy, self.nqx)
        self._check_no_face_folds_onto_itself(gens)
        self._resolve_native()

    def _check_no_face_folds_onto_itself(self, gens):
        """
        Catch a set of identifications that folds a face onto itself by accident.

        Two different corners of one face are the same point only where the grid says
        so -- a periodic wrap closing a single tile, or a fold seam glued to its own
        mirror. Anywhere else it is a contradiction: it would merge, say, two opposite
        vertices of a cube into one corner, and everything downstream would then
        attribute their velocity faces to the wrong place. Rather than resolve to
        something quietly wrong, say which face and which corners.
        """
        self_glued = {
            int(f) for g in gens
            for f in np.intersect1d(g.a[:, 0], g.b[:, 0])
        }
        for f in range(self.nf):
            if f in self_glued:
                continue
            lab = self.node_id[f].ravel()
            order = np.argsort(lab, kind="stable")
            s = lab[order]
            dup = np.flatnonzero(s[1:] == s[:-1])
            if dup.size:
                k = order[dup[0]]
                k2 = order[dup[0] + 1]
                J1, I1 = divmod(int(k), self.nqx)
                J2, I2 = divmod(int(k2), self.nqx)
                raise ValueError(
                    f"The declared topology makes corners (J={J1}, I={I1}) and "
                    f"(J={J2}, I={I2}) of face {f} the same physical point, but "
                    "nothing glues that face to itself. Two seams of this face must "
                    "disagree about how they line up -- check the axis and `reverse` "
                    "flags of its `face_connections` entries."
                )

    def _resolve_native(self):
        """
        Pick, for each node, the canonical native `(face, j, i)` that stores it.

        A node may be stored once (the common case), on several faces (a seam twin),
        or on none at all -- a corner that exists physically but falls outside every
        face's native array, such as a cube vertex or the high row a 'left'
        staggering drops. Those get `(-1, -1, -1)`.
        """
        t, nyq, nxq = self.t, self.nyq, self.nxq
        f, J, I = np.meshgrid(
            np.arange(self.nf), np.arange(self.nqy), np.arange(self.nqx), indexing="ij"
        )
        native = (J >= t) & (J < t + nyq) & (I >= t) & (I < t + nxq)
        # Pack (f, j, i) into one integer so the canonical choice is a plain minimum.
        packed = np.where(native, (f * nyq + (J - t)) * nxq + (I - t), -1)

        best = np.full(self.n_nodes, np.iinfo(np.int64).max, dtype=np.int64)
        ok = packed >= 0
        np.minimum.at(best, self.labels[ok.ravel()], packed[ok].ravel())

        node_native = np.full((self.n_nodes, 3), -1, dtype=np.int64)
        found = best < np.iinfo(np.int64).max
        b = best[found]
        node_native[found, 0] = b // (nyq * nxq)
        node_native[found, 1] = (b // nxq) % nyq
        node_native[found, 2] = b % nxq
        self.node_native = node_native
        self.node_is_stored = found

    # ------------------------------------------------------------ derived data

    @property
    def _reps_csr(self):
        """CSR-style `(offsets, slots)` listing every slot of each node."""
        if getattr(self, "_reps", None) is None:
            order = np.argsort(self.labels, kind="stable")
            offsets = np.searchsorted(self.labels[order], np.arange(self.n_nodes + 1))
            self._reps = (offsets, order)
        return self._reps

    def reps_of(self, node):
        """The `(face, J, I)` slots of `node`, as an (n, 3) array."""
        offsets, order = self._reps_csr
        s = order[offsets[node]:offsets[node + 1]]
        return np.stack(self.unslot(s), -1)

    @property
    def node_reps(self):
        """Every slot of every node, as a list of `(face, J, I)` tuples per node."""
        if getattr(self, "_reps_list", None) is None:
            self._reps_list = [
                [tuple(int(x) for x in slot) for slot in self.reps_of(n)]
                for n in range(self.n_nodes)
            ]
        return self._reps_list

    @property
    def node_adj(self):
        """The neighbouring nodes of each node, as a list of sets."""
        if getattr(self, "_adj_sets", None) is None:
            self._build_adjacency()
            self._adj_sets = [
                set(int(m) for m in self.neighbors(n)) for n in range(self.n_nodes)
            ]
        return self._adj_sets

    @property
    def valence(self):
        """Number of distinct neighbouring nodes of each node."""
        self._build_adjacency()
        return np.diff(self._adj_offsets)

    def neighbors(self, node):
        """The nodes sharing a grid edge with `node`."""
        self._build_adjacency()
        return self._adj[self._adj_offsets[node]:self._adj_offsets[node + 1]]

    def _build_adjacency(self):
        if getattr(self, "_adj", None) is not None:
            return
        nid = self.node_id
        pairs = []
        # Edges of the lattice, in each face's own frame. Identified slots make the
        # same physical edge appear more than once; duplicates are removed below.
        pairs.append(np.stack([nid[:, :, :-1].ravel(), nid[:, :, 1:].ravel()], -1))
        pairs.append(np.stack([nid[:, :-1, :].ravel(), nid[:, 1:, :].ravel()], -1))
        e = np.concatenate(pairs)
        e = e[e[:, 0] != e[:, 1]]                      # zero-length: not an edge
        lo = np.minimum(e[:, 0], e[:, 1])
        hi = np.maximum(e[:, 0], e[:, 1])
        key = np.unique(lo.astype(np.int64) * self.n_nodes + hi)
        lo, hi = key // self.n_nodes, key % self.n_nodes
        # undirected: store both orientations
        a = np.concatenate([lo, hi])
        b = np.concatenate([hi, lo])
        order = np.argsort(a, kind="stable")
        self._adj = b[order]
        self._adj_offsets = np.searchsorted(a[order], np.arange(self.n_nodes + 1))
        self._edges = np.stack([lo, hi], -1)

    @property
    def edges(self):
        """Every grid edge, as an (n_edges, 2) array of node pairs (lo, hi)."""
        self._build_adjacency()
        return self._edges

    # ------------------------------------------------------------- positions

    def _positions(self):
        """
        One geographic position per node: `(node_lon, node_lat, known)`.

        A node that is natively stored takes the position of its canonical
        representation, so every representation of a corner reports the *same*
        position however the grid's arrays spell them. A node stored on no face --
        a cube vertex, or the row a 'left' staggering drops -- is placed at the
        centroid of the tracer cells that meet there, which is well defined because
        the cells around a corner are exactly what the node is. Where fewer than
        three cells survive there is nothing to average and the position is left
        unknown rather than guessed.
        """
        if getattr(self, "_pos_cache", None) is not None:
            return self._pos_cache

        lon = np.full(self.n_nodes, np.nan)
        lat = np.full(self.n_nodes, np.nan)

        geo = get_geo_corners(self.grid)
        cd = self.coords
        Yq, Xq = cd["Y"]["corner"], cd["X"]["corner"]
        nat_lon = _faced(geo["X"], (Yq, Xq), self.facedim).astype(float)
        nat_lat = _faced(geo["Y"], (Yq, Xq), self.facedim).astype(float)

        stored = self.node_is_stored
        f, j, i = self.node_native[stored].T
        lon[stored] = nat_lon[f, j, i]
        lat[stored] = nat_lat[f, j, i]

        missing = np.flatnonzero(~stored)
        if missing.size:
            centers = self._cell_centers()
            if centers is not None:
                clon, clat = centers
                for n in missing:
                    cells = self._cells_around(n)
                    if len(cells) < 3:
                        continue
                    fc, jc, ic = np.array(cells).T
                    v = _lonlat_to_xyz(clon[fc, jc, ic], clat[fc, jc, ic]).mean(0)
                    norm = np.linalg.norm(v)
                    if norm == 0.0:
                        continue
                    v = v / norm
                    lat[n] = np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0)))
                    lon[n] = np.rad2deg(np.arctan2(v[1], v[0]))

        self._pos_cache = (lon, lat, np.isfinite(lon) & np.isfinite(lat))
        return self._pos_cache

    @property
    def node_lon(self):
        return self._positions()[0]

    @property
    def node_lat(self):
        return self._positions()[1]

    @property
    def node_position_known(self):
        return self._positions()[2]

    def _cell_centers(self):
        """Tracer-cell centre coordinates as `(lon, lat)` arrays, or None."""
        cd = self.coords
        if cd["X"]["center"] is None or cd["Y"]["center"] is None:
            return None
        Yc, Xc = cd["Y"]["center"], cd["X"]["center"]
        ds = self.grid._ds
        out = []
        for axis, want in (("X", "lon"), ("Y", "lat")):
            hit = [
                ds.coords[c] for c in ds.coords
                if want in c.lower() and Xc in ds.coords[c].dims and Yc in ds.coords[c].dims
            ]
            if not hit:
                return None
            out.append(_faced(hit[0], (Yc, Xc), self.facedim).astype(float))
        return out[0], out[1]

    def _cells_around(self, node):
        """
        The tracer cells that meet at `node`, as `(face, j, i)`.

        Taken as the union over the node's representations, so a corner on a seam
        collects the cells from both sides without any halo: each representation
        sees the cells that are in its own face.
        """
        cells = set()
        for f, J, I in self.reps_of(node):
            for dJ, dI in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
                jc, ic = J + dJ, I + dI
                if 0 <= jc < self.Nyc and 0 <= ic < self.Nxc:
                    cells.add((int(f), int(jc), int(ic)))
        return sorted(cells)

    # -------------------------------------------------------------- plateaus

    def plateau_labels(self, tolerance_m=1e-3):
        """
        Group corners that are distinct but sit at the same place.

        A grid may separate corners that geography does not: a bipolar cap's
        singular meridian is one point that a whole column of corners maps to, with
        zero-length velocity faces between them. Keeping them distinct is what lets a
        closed section say which of the fan's sectors it passed between -- but it
        also leaves a walk with a *flat* metric over the whole column, where greedy
        descent has nothing to choose on and the path would be decided by node
        numbering.

        So the walk treats such a group as one place to arrive at and leave, and
        this labels them. It is geometry used to navigate, never to decide identity.
        """
        key = round(float(tolerance_m), 12)
        cache = getattr(self, "_plateau_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        lon, lat, known = self._positions()
        e = self.edges
        a, b = e[:, 0], e[:, 1]
        ok = known[a] & known[b]
        xa = _lonlat_to_xyz(lon[a], lat[a])
        xb = _lonlat_to_xyz(lon[b], lat[b])
        chord = np.full(a.shape, np.inf)
        chord[ok] = np.linalg.norm(xa[ok] - xb[ok], axis=-1) * 6.371e6
        deg = chord < tolerance_m
        if deg.any():
            g = coo_matrix(
                (np.ones(int(deg.sum()), dtype=np.int8), (a[deg], b[deg])),
                shape=(self.n_nodes, self.n_nodes),
            )
            _, labels = connected_components(g, directed=False, return_labels=True)
        else:
            labels = np.arange(self.n_nodes, dtype=np.int64)
        labels = labels.astype(np.int64)
        self._plateau_cache = (key, labels)
        return labels

    # ------------------------------------------------------ edges -> velocities

    def edge_velocities(self, node_a, node_b):
        """
        Every native storage of the velocity face between two adjacent corner nodes.

        Each entry is ``(var, face, j, i, to_cell, from_cell, step, slots)``: the
        velocity component and its native index, the global tracer-cell ids its
        positive direction points to and from *in its own face's frame*, the step
        ``(dJ, dI)`` from `node_a` to `node_b` in that same frame, and the pair of
        corner slots it was read from. Reading a face in the frame it is stored in is
        what lets a rotated or reversed seam be crossed with no vector rotation at
        all.

        The velocity and the step travel together on purpose. A seam that folds a row
        onto itself stores one physical edge twice, in mirrored frames and therefore
        with opposite sign -- the duplicated, sign-flipped row of an ORCA-style fold.
        Taking the velocity from one spelling and the direction of travel from the
        other would make a section that runs out and back along such a row fail to
        cancel, so a caller must use one entry whole rather than mix two.

        A seam face is usually stored once, on whichever face's low edge it is; a
        face along a boundary fold is stored twice; a face on an edge no array covers
        is stored not at all, and the list is empty.
        """
        t, Nyc, Nxc = self.t, self.Nyc, self.Nxc
        reps_b = {}
        for f, J, I in self.reps_of(node_b):
            reps_b.setdefault(int(f), []).append((int(J), int(I)))

        out = []
        for f, Ja, Ia in self.reps_of(node_a):
            f, Ja, Ia = int(f), int(Ja), int(Ia)
            for Jb, Ib in reps_b.get(f, ()):
                if abs(Ja - Jb) + abs(Ia - Ib) != 1:
                    continue

                def gid(jc, ic):
                    if 0 <= jc < Nyc and 0 <= ic < Nxc:
                        return (f * Nyc + jc) * Nxc + ic
                    return None

                step = (Jb - Ja, Ib - Ia)
                slots = ((f, Ja, Ia), (f, Jb, Ib))
                if Ia == Ib:      # a vertical lattice edge: an X-direction velocity
                    jc, I = min(Ja, Jb), Ia
                    i = I - t
                    if 0 <= i < self.nxq and 0 <= jc < Nyc:
                        out.append(
                            ("U", f, jc, i, gid(jc, I), gid(jc, I - 1), step, slots)
                        )
                else:             # a horizontal lattice edge: a Y-direction velocity
                    J, ic = Ja, min(Ia, Ib)
                    j = J - t
                    if 0 <= j < self.nyq and 0 <= ic < Nxc:
                        out.append(
                            ("V", f, j, ic, gid(J, ic), gid(J - 1, ic), step, slots)
                        )
        return out

    @property
    def face_handedness(self):
        """
        Whether each face's `+i, +j` axes are right-handed on the sphere, as `+-1`.

        A velocity face's sign relative to a section is a question about *sides* --
        does the stored velocity point to the left of the way we are going? -- and in
        the frame the velocity is stored in that is pure index arithmetic. The only
        thing the frame cannot tell you is whether it is mirrored relative to the
        world, so that is measured once per face here, from one cell, rather than per
        edge from the section's own geometry. A section edge whose two corners
        coincide (a polar fan) then still gets a definite answer, where a cross
        product of two zero-length steps would not.
        """
        if getattr(self, "_handed", None) is not None:
            return self._handed
        geo = get_geo_corners(self.grid)
        cd = self.coords
        Yq, Xq = cd["Y"]["corner"], cd["X"]["corner"]
        lon = _faced(geo["X"], (Yq, Xq), self.facedim).astype(float)
        lat = _faced(geo["Y"], (Yq, Xq), self.facedim).astype(float)
        # Measured on the sphere, not in a flat (east, north) frame: a local frame
        # degenerates at a pole, where a single cell would then report the opposite
        # handedness to the rest of its face. `(e_i x e_j) . r` is the same question
        # asked in a way the pole does not break.
        p = _lonlat_to_xyz(lon[:, :-1, :-1], lat[:, :-1, :-1])
        ex = _lonlat_to_xyz(lon[:, :-1, 1:], lat[:, :-1, 1:]) - p
        ey = _lonlat_to_xyz(lon[:, 1:, :-1], lat[:, 1:, :-1]) - p
        cross = np.sum(np.cross(ex, ey) * p, axis=-1)

        handed = np.ones(self.nf, dtype=np.int64)
        for f in range(self.nf):
            c = cross[f][np.isfinite(cross[f])]
            c = c[np.abs(c) > 1e-12]        # degenerate cells say nothing either way
            if c.size == 0:
                continue
            pos, neg = int((c > 0).sum()), int((c < 0).sum())
            if pos and neg:
                raise ValueError(
                    f"Face {f} is not consistently oriented: {pos} of its cells run "
                    f"one way round and {neg} the other. A face is a chart of a piece "
                    "of the sphere, so every cell of it must have the same handedness; "
                    "cells that do not are inside out, and the velocity faces around "
                    "them would be signed backwards. Check the corner coordinates -- "
                    "this usually means two rows of them cross."
                )
            handed[f] = 1 if pos else -1
        self._handed = handed
        return handed

    def cells_of_edge(self, node_a, node_b):
        """The (up to two) tracer cells an edge separates, as global ids."""
        shared = set(self._cells_around(node_a)) & set(self._cells_around(node_b))
        return {(f * self.Nyc + j) * self.Nxc + i for f, j, i in shared}

    def padded_transports(self, u, v):
        """
        Extend native staggered transports onto the full 'outer' lattice.

        A staggered grid stores one velocity face once, so a face's own arrays stop
        short of two of its edges; the values are not missing, they are held by
        whichever face does store them. Each slot the native arrays do not cover is
        therefore filled by asking the corner graph which physical edge it is and
        reading that edge's stored value, re-signed for the receiving face's own axis
        direction.

        Nothing is rotated: every value is read in the frame it was written in, so
        there is no chance of getting wrong how a rotated or reversed seam maps `u`
        onto `v`. And it takes plain arrays rather than a `{axis: component}` mapping.

        An edge stored on *no* face -- an open wall, or a junction a staggering
        leaves out, such as a cubed sphere's un-stored vertices -- is filled with
        0.0. For a wall that is the true transport. For an un-stored junction the
        edge exists but nothing holds its value, so 0.0 is the conservative choice
        rather than a truth: it keeps cell convergence exactly conservative and
        confines the unknown flux to those edges, where a halo pad would spread its
        `fill_value` through the sum instead.

        Parameters
        ----------
        u, v : xr.DataArray or np.ndarray
            Native X- and Y-direction transports, dims ``([face,] Y, X)`` with X/Y at
            (corner, center) for `u` and (center, corner) for `v`.

        Returns
        -------
        (Uo, Vo) : np.ndarray
            Shapes ``(nf, Nyc, Nxc + 1)`` and ``(nf, Nyc + 1, Nxc)``.
        """
        nf, Nyc, Nxc, t = self.nf, self.Nyc, self.Nxc, self.t
        U = np.nan_to_num(np.asarray(getattr(u, "values", u), dtype=float))
        V = np.nan_to_num(np.asarray(getattr(v, "values", v), dtype=float))
        if self.single_tile:
            U, V = U[None, ...], V[None, ...]

        def gid(f, j, i):
            return (f * Nyc + j) * Nxc + i if (0 <= j < Nyc and 0 <= i < Nxc) else None

        def value(nA, nB, want_to, want_from):
            """
            That physical edge's stored transport, signed for this slot.

            The two are compared by which tracer cell each points *at*. Only one side
            need be identifiable: an edge separates exactly two cells, so agreeing on
            one of them fixes the direction. That matters here, because the face that
            stores a seam edge names only the cell on its own side -- the other lies
            on the face we are filling, which its frame cannot see.
            """
            for var, f2, j2, i2, to_cell, from_cell, _step, _slots in (
                self.edge_velocities(nA, nB)
            ):
                val = float((U if var == "U" else V)[f2, j2, i2])
                for stored, sign in ((to_cell, 1.0), (from_cell, -1.0)):
                    if stored is None:
                        continue
                    if stored == want_to:
                        return sign * val
                    if stored == want_from:
                        return -sign * val
            return 0.0

        Uo = np.zeros((nf, Nyc, Nxc + 1))
        Uo[:, :, t:t + self.nxq] = U
        Vo = np.zeros((nf, Nyc + 1, Nxc))
        Vo[:, t:t + self.nyq, :] = V

        for f in range(nf):
            for I in (I for I in (0, Nxc) if not (t <= I < t + self.nxq)):
                for jc in range(Nyc):
                    nA = int(self.node_id[f, jc, I])
                    nB = int(self.node_id[f, jc + 1, I])
                    # `+x` points from the cell on the slot's low side to the high
                    own = gid(f, jc, I if I == 0 else I - 1)
                    other = (self.cells_of_edge(nA, nB) - {own} - {None})
                    other = next(iter(other)) if other else None
                    Uo[f, jc, I] = (
                        value(nA, nB, own, other) if I == 0
                        else value(nA, nB, other, own)
                    )
            for J in (J for J in (0, Nyc) if not (t <= J < t + self.nyq)):
                for ic in range(Nxc):
                    nA = int(self.node_id[f, J, ic])
                    nB = int(self.node_id[f, J, ic + 1])
                    own = gid(f, J if J == 0 else J - 1, ic)
                    other = (self.cells_of_edge(nA, nB) - {own} - {None})
                    other = next(iter(other)) if other else None
                    Vo[f, J, ic] = (
                        value(nA, nB, own, other) if J == 0
                        else value(nA, nB, other, own)
                    )
        return Uo, Vo

    # ----------------------------------------------------------- native output

    def native_of(self, nodes):
        """
        The canonical native `(face, j, i)` of each node.

        Raises where a node is stored on no face: it is a real corner, and a section
        may legitimately pass through it, but it has no native index and inventing
        one would put a wrong number into an index array. Callers that can work in
        node ids should do so.
        """
        nodes = np.asarray(nodes)
        bad = np.flatnonzero(~self.node_is_stored[nodes])
        if bad.size:
            n = int(nodes[bad[0]])
            lon, lat, known = self._positions()
            # Name the slot, not just the position: a corner stored on no face often
            # has too few cells around it to place, so the position is usually the
            # thing that is *not* available -- which makes for a useless message
            # exactly where this fires.
            slots = ", ".join(
                f"(face={f}, J={J}, I={I})" for f, J, I in self.reps_of(n)[:3]
            )
            where = (
                f", near lon={lon[n]:.3f}, lat={lat[n]:.3f}" if known[n]
                else ", whose position the grid does not determine either"
            )
            raise ValueError(
                f"Corner node {n} -- corner slot {slots}{where} -- is a real grid "
                "corner, but this grid stores "
                "it on no face, so it has no native (face, j, i) index to report.\n\n"
                "A 'left' staggering drops each face's high corner row and a 'right' "
                "one its low row, so a corner falling only on dropped rows is absent "
                "from every array -- which happens along a seam where two faces meet "
                "on the same side of their axes, and at a junction such as a cubed "
                "sphere's un-stored vertices. The section itself is valid and its "
                "node ids describe it exactly; it is only the native indices that "
                "cannot.\n\n"
                "Sections are routed around such corners wherever a stored "
                "alternative exists, so reaching this means the path has no other "
                "way through. Provide the grid in 'outer' staggering, which stores "
                "every corner, or work with the section's node ids."
            )
        return self.node_native[nodes]

    def validate_positions(self, tolerance_m=1.0):
        """
        Check that the coordinates agree with the topology, without ever letting
        them decide it.

        Every representation of a node is the same physical corner, so they should
        carry the same stored position. Where they do not, the grid's metadata and
        its coordinates disagree -- a declared fold whose corner arrays do not carry
        the fold, say -- and that is worth reporting loudly, because everything
        downstream that measures a distance will be wrong even though the topology
        is right.
        """
        geo = get_geo_corners(self.grid)
        cd = self.coords
        Yq, Xq = cd["Y"]["corner"], cd["X"]["corner"]
        nat_lon = _faced(geo["X"], (Yq, Xq), self.facedim).astype(float)
        nat_lat = _faced(geo["Y"], (Yq, Xq), self.facedim).astype(float)

        t, nyq, nxq = self.t, self.nyq, self.nxq
        f, J, I = np.meshgrid(
            np.arange(self.nf), np.arange(self.nqy), np.arange(self.nqx), indexing="ij"
        )
        native = (J >= t) & (J < t + nyq) & (I >= t) & (I < t + nxq)
        xyz = np.full((self.nf, self.nqy, self.nqx, 3), np.nan)
        fn, Jn, In = f[native], J[native], I[native]
        xyz[native] = _lonlat_to_xyz(
            nat_lon[fn, Jn - t, In - t], nat_lat[fn, Jn - t, In - t]
        )

        lab = self.labels.reshape(self.nf, self.nqy, self.nqx)[native]
        pts = xyz[native]
        order = np.argsort(lab, kind="stable")
        lab, pts = lab[order], pts[order]
        starts = np.flatnonzero(np.diff(lab)) + 1
        worst, worst_node = 0.0, -1
        for lo, hi in zip(np.r_[0, starts], np.r_[starts, lab.size]):
            if hi - lo < 2:
                continue
            block = pts[lo:hi]
            d = np.linalg.norm(block - block[0], axis=1).max() * 6.371e6
            if d > worst:
                worst, worst_node = float(d), int(lab[lo])
        if worst > tolerance_m:
            raise ValueError(
                f"The corner coordinates disagree with the declared topology: node "
                f"{worst_node} is stored more than once, and its copies are "
                f"{worst:.3g} m apart. The topology itself is taken from the grid's "
                "metadata and is unaffected, but distances measured along a section "
                "through this corner will not be meaningful. Check that the "
                "coordinates carry the periodicity or fold the grid declares."
            )
        return worst


def declare_identifications(grid, identifications):
    """
    Attach corner identifications a grid's own metadata cannot express, and cache
    them so everything downstream sees them.

    `face_connections` maps a whole tile edge to a whole tile edge, so it cannot say
    that one edge is glued to several partners over different index ranges. That is
    the shape of a lat-lon-cap grid's southern boundary, where the domain's edge
    folds onto itself across four tile edges, and it is why such a grid is usually
    published declaring those edges as walls -- leaving one line of physical corners
    stored twice with nothing to say they are the same.

    Parameters
    ----------
    grid : xgcm.Grid
    identifications : list of Identification

    Returns
    -------
    CornerTopology
    """
    ct = CornerTopology(grid, identifications=identifications)
    grid._sectionate_corner_topology = ct
    return ct


def corner_topology(grid, identifications=None):
    """
    The `CornerTopology` of `grid`, cached on the grid instance.

    Resolving the topology touches every corner of the grid once, so it is worth
    reusing across the several places in a section's life that need it.
    """
    if identifications:
        return CornerTopology(grid, identifications=identifications)
    cached = getattr(grid, "_sectionate_corner_topology", None)
    if cached is None:
        cached = CornerTopology(grid)
        try:
            grid._sectionate_corner_topology = cached
        except AttributeError:  # pragma: no cover - a grid that forbids attributes
            pass
    return cached


# Each edge of a face, as (outward direction, tangential direction) in that face's
# own (di, dj) basis.
_EDGE_FRAME = {
    ("X", 0): ((-1, 0), (0, 1)),
    ("X", 1): ((1, 0), (0, 1)),
    ("Y", 0): ((0, -1), (1, 0)),
    ("Y", 1): ((0, 1), (1, 0)),
}


def _rot90(v, k):
    for _ in range(k % 4):
        v = (-v[1], v[0])
    return v


def _tangential_is_forward(axis, side, axis2, side2):
    """
    Whether two glued edges run the same way along the seam.

    A grid is one oriented surface, so the map between two faces that meet is a
    rotation, not a reflection. Crossing the seam turns one face's outward direction
    into the other's inward direction, and there is exactly one rotation that does
    that; where it sends the tangential direction is then not a choice. So a
    90-degree gluing reverses the seam even when nothing declares a reversal, and a
    same-side gluing reverses it too -- both fall out of the same rule rather than
    needing a case each.
    """
    out1, tan1 = _EDGE_FRAME[(axis, side)]
    out2, tan2 = _EDGE_FRAME[(axis2, side2)]
    inward2 = (-out2[0], -out2[1])
    k = next(k for k in range(4) if _rot90(out1, k) == inward2)
    return _rot90(tan1, k) == tan2





def _corner_coord_dict(grid):
    """
    `coord_dict` that tolerates a grid with no tracer-center coordinates.

    Corner identity needs only the corner arrays and the declared topology, so a
    corner-only grid is perfectly well described; it simply cannot say anything about
    tracer cells or velocities later.
    """
    corner_pos = corner_position(grid)
    out = {}
    for ax in ("X", "Y"):
        coords = grid.axes[ax].coords
        out[ax] = {
            "corner": coords[corner_pos],
            "center": coords.get("center"),
        }
    return out
