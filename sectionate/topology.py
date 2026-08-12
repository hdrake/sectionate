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
from xgcm.padding import pad as _module_pad

try:  # xgcm >= 0.10.1
    from xgcm.padding import _is_fold_padding, _seam_partner_indices, _resolve_pivot
except ImportError:  # pragma: no cover - keeps `import sectionate` working
    _is_fold_padding = _seam_partner_indices = _resolve_pivot = None

from .gridutils import (
    corner_position, get_facedim, get_geo_corners, _pad_axes,
)


def _faced(da, dims, facedim):
    """`da`'s values with a leading face axis, real or synthetic."""
    if facedim is None:
        return da.transpose(*dims).values[None, ...]
    return da.transpose(facedim, ..., *dims).values


def _flat_step(lon0, lat0, lon1, lat1):
    """Displacement in a local flat (east, north) frame, in degrees, with longitudes
    scaled by cos(lat) and wrapped across the dateline."""
    dlon = ((lon1 - lon0 + 180.0) % 360.0) - 180.0
    return dlon * np.cos(np.deg2rad(0.5 * (lat0 + lat1))), (lat1 - lat0)


def _lonlat_to_xyz(lon, lat):
    """Positions on the unit sphere, so that comparing them is free of the
    longitude wrap and of the pole's degenerate longitudes."""
    lon = np.deg2rad(np.asarray(lon, dtype=float))
    lat = np.deg2rad(np.asarray(lat, dtype=float))
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


# Directions on a face's own lattice, as (dJ, dI) steps.
_STEPS = {"right": (0, 1), "left": (0, -1), "up": (1, 0), "down": (-1, 0)}

# The four sides of a face, each as (axis, which_end).
_SIDES = ("X_low", "X_high", "Y_low", "Y_high")


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


def seam_halo(grid, field, width=1, fill_value=np.nan):
    """
    Pad `field` by `width` cells, filling only across genuine seams.

    A periodic wrap, a bipolar fold and a `face_connections` gluing all supply real
    neighbouring values; every other boundary is a wall and is filled with
    `fill_value` (NaN by default). This is the halo any topology-aware sweep over
    tracer cells wants: `"extend"` padding would replicate the edge cell and make a
    wall look like a connection.

    Parameters
    ----------
    grid : xgcm.Grid
    field : xr.DataArray
        Any field whose dims include the horizontal axes to pad.
    width : int or dict, optional
        Halo width, per axis if a dict.
    fill_value : float, optional

    Returns
    -------
    xr.DataArray
    """
    axes = _pad_axes(grid, field.dims)
    padding = {ax: _seam_or_fill(grid.axes[ax].padding) for ax in axes}
    if isinstance(width, dict):
        widths = {ax: width.get(ax, (0, 0)) for ax in axes}
        widths = {
            ax: (w, w) if np.isscalar(w) else tuple(w) for ax, w in widths.items()
        }
    else:
        widths = {ax: (width, width) for ax in axes}
    return _module_pad(field, grid, widths, padding=padding, fill_value=fill_value)


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

    def _cell_neighbor_arrays(self):
        """
        Global tracer-cell ids, and each cell's neighbour across its four edges.

        Padded one cell along a single axis at a time: a halo filled across two seams
        at once (the diagonal corners of a two-axis pad) is a pad of a pad and is not
        reliable, and nothing here needs it.
        """
        cd, ds = self.coords, self.grid._ds
        Yc, Xc = cd["Y"]["center"], cd["X"]["center"]
        if Yc is None or Xc is None:
            raise ValueError(
                "This grid has `face_connections` but no tracer-center coordinates "
                "on its X and Y axes. The corner identifications across a tile seam "
                "are derived from which tracer cells meet across it, so the center "
                "coordinates are required. Declare them on the `xgcm.Grid`."
            )
        nf, Nyc, Nxc = self.nf, self.Nyc, self.Nxc
        cid = xr.DataArray(
            np.arange(nf * Nyc * Nxc, dtype=float).reshape(nf, Nyc, Nxc),
            dims=(self.facedim, Yc, Xc),
        )
        axes = _pad_axes(self.grid, cid.dims)
        padding = {ax: _seam_or_fill(self.grid.axes[ax].padding) for ax in axes}
        one = {ax: (1, 1) if ax == "X" else (0, 0) for ax in axes}
        GX = _module_pad(cid, self.grid, one, padding=padding, fill_value=np.nan)
        one = {ax: (1, 1) if ax == "Y" else (0, 0) for ax in axes}
        GY = _module_pad(cid, self.grid, one, padding=padding, fill_value=np.nan)
        GX = GX.transpose(self.facedim, Yc, Xc).values      # (nf, Nyc, Nxc+2)
        GY = GY.transpose(self.facedim, Yc, Xc).values      # (nf, Nyc+2, Nxc)
        return GX, GY

    def _face_connection_generators(self):
        """
        The identification each `face_connections` gluing makes, as an affine map.

        The map is *fitted* from tracer cells, which pad reliably across any seam
        however it is rotated or reversed, and then applied along the whole seam
        line -- including its two end corners, where the cells needed to see the
        correspondence directly have run out. That is what lets a seam that meets a
        wall, or a pole, still identify its corners.
        """
        if self.single_tile:
            return []

        GX, GY = self._cell_neighbor_arrays()
        Nyc, Nxc, nqy, nqx = self.Nyc, self.Nxc, self.nqy, self.nqx

        def cell_neighbor(f, jc, ic, d):
            if d == "left":
                return GX[f, jc, ic]
            if d == "right":
                return GX[f, jc, ic + 2]
            if d == "down":
                return GY[f, jc, ic]
            return GY[f, jc + 2, ic]

        def decode(g):
            g = int(g)
            return g // (Nyc * Nxc), (g // Nxc) % Nyc, g % Nxc

        out = []
        for f in range(self.nf):
            for side in _SIDES:
                # cells along this side, in seam order, and the neighbour beyond it
                if side == "X_low":
                    cells = [(f, p, 0) for p in range(Nyc)]
                    outward, seam_corner = "left", lambda p: (p, 0)
                elif side == "X_high":
                    cells = [(f, p, Nxc - 1) for p in range(Nyc)]
                    outward, seam_corner = "right", lambda p: (p, nqx - 1)
                elif side == "Y_low":
                    cells = [(f, 0, p) for p in range(Nxc)]
                    outward, seam_corner = "down", lambda p: (0, p)
                else:
                    cells = [(f, Nyc - 1, p) for p in range(Nxc)]
                    outward, seam_corner = "up", lambda p: (nqy - 1, p)

                nbr = [cell_neighbor(*c, outward) for c in cells]
                # Which partner (face, back-direction) each position glues to. A
                # wall pads NaN and glues to nothing.
                partner = []
                for c, g in zip(cells, nbr):
                    if not np.isfinite(g):
                        partner.append(None)
                        continue
                    f2, jc2, ic2 = decode(g)
                    own = (c[0] * Nyc + c[1]) * Nxc + c[2]
                    back = [
                        d for d in _STEPS
                        if cell_neighbor(f2, jc2, ic2, d) == own
                    ]
                    partner.append((f2, jc2, ic2, back[0]) if len(back) == 1 else None)

                for run in _contiguous_runs(partner):
                    p0, p1 = run                       # inclusive cell range
                    if p1 == p0:
                        raise NotImplementedError(
                            f"Face {f} is glued to another face across a single "
                            f"cell on its {side} edge. The direction the seam runs "
                            "in cannot be read from one cell alone; declare this "
                            "gluing explicitly with an `Identification`."
                        )
                    pairs = []
                    for p in range(p0, p1):
                        a = seam_corner(p + 1)
                        b = _shared_seam_corner(
                            partner[p], partner[p + 1], Nyc, Nxc, nqy, nqx
                        )
                        if b is None:
                            break
                        pairs.append((a, b))
                    if not pairs:
                        continue
                    ident = _affine_extend(
                        partner, run, pairs, seam_corner, nqy, nqx, f, side
                    )
                    if ident is not None:
                        out.append(ident)
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
    def node_reps(self):
        """CSR-style `(offsets, slots)` listing every slot of each node."""
        if getattr(self, "_reps", None) is None:
            order = np.argsort(self.labels, kind="stable")
            offsets = np.searchsorted(self.labels[order], np.arange(self.n_nodes + 1))
            self._reps = (offsets, order)
        return self._reps

    def reps_of(self, node):
        """The `(face, J, I)` slots of `node`, as an (n, 3) array."""
        offsets, order = self.node_reps
        s = order[offsets[node]:offsets[node + 1]]
        return np.stack(self.unslot(s), -1)

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

        Each entry is ``(var, face, j, i, to_cell, from_cell)``: the velocity
        component and its native index, plus the global tracer-cell ids its positive
        direction points to and from *in its own face's frame*. Reading a face in the
        frame it is stored in is what lets a rotated or reversed seam be crossed with
        no vector rotation at all.

        A seam face is usually stored once, on whichever face's low edge it is; a
        face across a boundary fold can be stored twice; a face on an edge no array
        covers is stored not at all, and the list is empty.
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

                if Ia == Ib:      # a vertical lattice edge: an X-direction velocity
                    jc, I = min(Ja, Jb), Ia
                    i = I - t
                    if 0 <= i < self.nxq and 0 <= jc < Nyc:
                        out.append(("U", f, jc, i, gid(jc, I), gid(jc, I - 1)))
                else:             # a horizontal lattice edge: a Y-direction velocity
                    J, ic = Ja, min(Ia, Ib)
                    j = J - t
                    if 0 <= j < self.nyq and 0 <= ic < Nxc:
                        out.append(("V", f, j, ic, gid(J, ic), gid(J - 1, ic)))
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
        handed = np.ones(self.nf, dtype=np.int64)
        for f in range(self.nf):
            found = False
            for j in range(self.nyq - 1):
                for i in range(self.nxq - 1):
                    p, px, py = (lat[f, j, i], lat[f, j, i + 1], lat[f, j + 1, i])
                    if not (np.isfinite(p) and np.isfinite(px) and np.isfinite(py)):
                        continue
                    ex = _flat_step(lon[f, j, i], p, lon[f, j, i + 1], px)
                    ey = _flat_step(lon[f, j, i], p, lon[f, j + 1, i], py)
                    cross = ex[0] * ey[1] - ex[1] * ey[0]
                    if abs(cross) > 1e-12:
                        handed[f] = 1 if cross > 0 else -1
                        found = True
                        break
                if found:
                    break
        self._handed = handed
        return handed

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
            where = (
                f" (near lon={lon[n]:.3f}, lat={lat[n]:.3f})" if known[n] else ""
            )
            raise ValueError(
                f"Corner node {n}{where} is a real grid corner but is stored on no "
                "face, so it has no native (face, j, i) index. This happens where a "
                "staggering drops the row an edge falls on, and at junctions such as "
                "a cubed sphere's un-stored vertices. Work with the section's node "
                "ids instead of its native indices here."
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


def _contiguous_runs(partner):
    """
    Maximal runs of seam positions glued to the same face from the same side.

    A tile edge usually glues to exactly one partner over its whole length, but
    nothing here assumes that: a wall in the middle, or two partners over different
    stretches, simply yields more than one run.
    """
    runs, start = [], None
    for p, q in enumerate(partner):
        key = None if q is None else (q[0], q[3])
        prev = None if start is None else (partner[start][0], partner[start][3])
        if key is None:
            if start is not None:
                runs.append((start, p - 1))
                start = None
        elif start is None:
            start = p
        elif key != prev:
            runs.append((start, p - 1))
            start = p
    if start is not None:
        runs.append((start, len(partner) - 1))
    return [r for r in runs if r[1] > r[0]]


def _shared_seam_corner(pa, pb, Nyc, Nxc, nqy, nqx):
    """
    The corner on the neighbouring face's seam line shared by two adjacent cells.

    `pa`/`pb` are `(face, j, i, back_direction)` for two cells that are adjacent
    along the seam. They share an edge, whose two corners are one step apart across
    the seam line; the one *on* the line is the image of the corner the two cells
    share on this side.
    """
    if pa is None or pb is None or pa[0] != pb[0] or pa[3] != pb[3]:
        return None
    _, ja, ia, d = pa
    _, jb, ib, _ = pb
    if d in ("left", "right"):
        # the seam line runs along j, so the two cells must differ in j
        if ia != ib or abs(ja - jb) != 1:
            return None
        return (max(ja, jb), 0 if d == "left" else nqx - 1)
    if ja != jb or abs(ia - ib) != 1:
        return None
    return (0 if d == "down" else nqy - 1, max(ia, ib))


def _affine_extend(partner, run, pairs, seam_corner, nqy, nqx, f, side):
    """
    Turn one verified corner correspondence into the whole seam line's map.

    The *direction* the neighbouring face's indices run in comes from its cells,
    which are known at every position of the run; the *offset* comes from a corner
    pair. Together they give an exact integer map, which is then applied to every
    corner of the run -- including its two ends, where there is no next cell to read
    the correspondence from directly. That extension is the whole point: it is what
    identifies the corners where a seam runs into a wall or a pole.
    """
    p0, p1 = run
    f2 = partner[p0][0]
    d2 = partner[p0][3]
    along_j = d2 in ("left", "right")

    def cell_coord(q):
        return q[1] if along_j else q[2]

    # direction of travel on the neighbouring face, from its cells
    dc = cell_coord(partner[p1]) - cell_coord(partner[p0])
    if abs(dc) != (p1 - p0):
        return None
    s = 1 if dc > 0 else -1

    # offset, from a verified corner pair
    (Ja, Ia), (Jb, Ib) = pairs[0]
    K = Ja if side.startswith("X") else Ia
    K2 = Jb if along_j else Ib
    c = K2 - s * K

    Ks = np.arange(p0, p1 + 2)
    K2s = s * Ks + c
    n2 = nqy if along_j else nqx
    if K2s.min() < 0 or K2s.max() >= n2:
        return None

    a = np.array([(f,) + seam_corner(int(k)) for k in Ks], dtype=np.int64)
    if along_j:
        fixed = 0 if d2 == "left" else nqx - 1
        b = np.stack([np.full_like(K2s, f2), K2s, np.full_like(K2s, fixed)], -1)
    else:
        fixed = 0 if d2 == "down" else nqy - 1
        b = np.stack([np.full_like(K2s, f2), np.full_like(K2s, fixed), K2s], -1)
    return Identification(a, b, name=f"face {f} {side} -> face {f2}")


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
