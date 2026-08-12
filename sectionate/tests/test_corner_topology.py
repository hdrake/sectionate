"""
Corner identity is decided by the grid's declared topology, never by coordinates.

Every expectation here is derived from what the grid *declares*, independently of
how `sectionate` happens to compute it: a periodic axis of `Nxc` cells has `Nxc`
distinct corner columns whatever `Nxc` is; a corner-pivot fold glues its seam row to
its own mirror, which pairs all but the two fixed points; a cubed sphere has
`6N^2 + 2` corner points. Where an expectation cannot be derived that way it is
stated as an invariant (a refinement relation, a valence that matches the cells
actually meeting at a point) rather than as a recorded number.
"""

import os
import sys
import time

import numpy as np
import pytest
import xarray as xr
import xgcm

from sectionate.topology import CornerTopology, Identification
from sectionate.gridutils import outer_topology

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _single_tile_grid(nx, ny, x_padding="periodic", y_padding="fill"):
    """A minimal 'outer' single-tile grid with the requested axis topology."""
    xq = np.arange(nx + 1, dtype=float) * (360.0 / nx)
    yq = np.linspace(-40.0, 40.0, ny + 1)
    xh = 0.5 * (xq[:-1] + xq[1:])
    yh = 0.5 * (yq[:-1] + yq[1:])
    lon_c, lat_c = np.meshgrid(xq, yq)
    lon, lat = np.meshgrid(xh, yh)
    ds = xr.Dataset(
        coords={
            "xh": xh, "yh": yh, "xq": xq, "yq": yq,
            "geolon": (("yh", "xh"), lon), "geolat": (("yh", "xh"), lat),
            "geolon_c": (("yq", "xq"), lon_c), "geolat_c": (("yq", "xq"), lat_c),
        }
    )
    return xgcm.Grid(
        ds,
        coords={"X": {"center": "xh", "outer": "xq"},
                "Y": {"center": "yh", "outer": "yq"}},
        padding={"X": x_padding, "Y": y_padding},
        autoparse_metadata=False,
    )


# --------------------------------------------------------------------- the axis


@pytest.mark.parametrize("nx", [1, 2, 3, 4, 5, 8, 17])
def test_periodic_axis_of_any_width_has_one_corner_column_per_cell(nx):
    """
    A periodic axis identifies its first and last corner column and nothing else,
    so `Nxc` cells leave exactly `Nxc` distinct corner columns -- for every width.

    Narrow axes are the point. Deciding identity from the ring of tracer cells around
    a corner cannot get this right: on a 2-cell-wide periodic axis all three corner
    columns of a row are surrounded by the same two cells, so a cell-set test merges
    corners that the grid says are distinct.
    """
    ny = 4
    ct = CornerTopology(_single_tile_grid(nx, ny))
    assert ct.n_nodes == (ny + 1) * nx
    # the wrap itself, and only the wrap
    assert (ct.node_id[0, :, 0] == ct.node_id[0, :, nx]).all()
    interior = ct.node_id[0, :, :nx]
    assert len(np.unique(interior)) == interior.size


def test_a_wall_row_still_identifies_its_periodic_twins():
    """
    Periodicity is a property of the axis, so it holds on the boundary rows too.

    A corner on a walled edge touches only two tracer cells, and two cells share an
    edge rather than meeting at a point -- so nothing about the cells around it can
    reveal that the row's two ends are the same corner. The declaration can.
    """
    ct = CornerTopology(_single_tile_grid(6, 4, y_padding="fill"))
    for row in (0, ct.nqy - 1):
        assert ct.node_id[0, row, 0] == ct.node_id[0, row, ct.nqx - 1]
        assert len(np.unique(ct.node_id[0, row, :])) == ct.Nxc


def test_a_walled_axis_identifies_nothing():
    ct = CornerTopology(_single_tile_grid(6, 4, x_padding="fill"))
    assert ct.n_nodes == ct.n_slots


def test_doubly_periodic_grid():
    nx, ny = 5, 4
    ct = CornerTopology(_single_tile_grid(nx, ny, "periodic", "periodic"))
    assert ct.n_nodes == nx * ny


# --------------------------------------------------------------------- the fold


def _xgcm_supports_fold():
    try:
        from xgcm.padding import _is_fold_padding  # noqa: F401
    except ImportError:
        return False
    return True


fold_only = pytest.mark.skipif(
    not _xgcm_supports_fold(), reason="requires an xgcm with north-fold padding"
)


@fold_only
@pytest.mark.parametrize("nx", [4, 6, 8, 12])
def test_corner_pivot_fold_pairs_its_seam_row_about_two_fixed_points(nx):
    """
    A corner-pivot fold glues the seam row to itself by `I -> -I (mod Nxc)`.

    That map has exactly two fixed points -- the two poles of the bipolar cap, at
    `I = 0` and `I = Nxc/2` for even `Nxc` -- and pairs off everything else. So the
    seam row keeps `Nxc/2 + 1` distinct corners out of `Nxc`, and the node count
    falls by the number of pairs.
    """
    ny = 4
    grid = _single_tile_grid(nx, ny, "periodic", {"fold": "corner"})
    ct = CornerTopology(grid)
    n_pairs = (nx - 2) // 2
    assert ct.n_nodes == (ny + 1) * nx - n_pairs
    seam = ct.node_id[0, ct.nqy - 1, :nx]
    assert len(np.unique(seam)) == nx // 2 + 1
    # the two fixed points really are fixed
    assert seam[0] == ct.node_id[0, ct.nqy - 1, 0]
    assert len(np.unique([seam[i] for i in (0, nx // 2)])) == 2


@fold_only
def test_fold_identity_ignores_the_corner_coordinates():
    """
    Identity comes from the declaration, so corrupting the coordinates cannot move
    it. This is the property the whole design turns on: a grid whose stored corner
    positions do not carry the fold symmetry it declares is still resolved correctly.
    """
    grid = _single_tile_grid(8, 4, "periodic", {"fold": "corner"})
    ref = CornerTopology(grid).node_id.copy()
    rng = np.random.default_rng(0)
    ds = grid._ds.copy(deep=True)
    ds["geolon_c"] = ds["geolon_c"] + rng.normal(0.0, 2.0, ds["geolon_c"].shape)
    ds["geolat_c"] = ds["geolat_c"] + rng.normal(0.0, 2.0, ds["geolat_c"].shape)
    jittered = xgcm.Grid(
        ds,
        coords={"X": {"center": "xh", "outer": "xq"},
                "Y": {"center": "yh", "outer": "yq"}},
        padding={"X": "periodic", "Y": {"fold": "corner"}},
        autoparse_metadata=False,
    )
    assert (CornerTopology(jittered).node_id == ref).all()


# ------------------------------------------------------- explicit declarations


def test_explicit_identification_merges_what_it_names():
    """
    Topology a grid's own metadata cannot express is still an ordinary affine map,
    and can be declared directly -- which is how a lat-lon-cap grid's southern
    boundary fold, glued to several partners over different index ranges, is stated.
    """
    grid = _single_tile_grid(6, 4, x_padding="fill")
    plain = CornerTopology(grid)
    J = np.arange(plain.nqy)
    ident = Identification(
        np.stack([np.zeros_like(J), J, np.zeros_like(J)], -1),
        np.stack([np.zeros_like(J), J, np.full_like(J, plain.nqx - 1)], -1),
        name="hand-declared wrap",
    )
    ct = CornerTopology(grid, identifications=[ident])
    assert ct.n_nodes == plain.n_nodes - plain.nqy
    assert (ct.node_id[0, :, 0] == ct.node_id[0, :, ct.nqx - 1]).all()


def test_identification_outside_the_lattice_raises():
    grid = _single_tile_grid(6, 4)
    bad = Identification([[0, 0, 0]], [[0, 0, 999]], name="nonsense")
    with pytest.raises(ValueError, match="outside this grid"):
        CornerTopology(grid, identifications=[bad])


def test_identification_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        Identification([[0, 0, 0], [0, 0, 1]], [[0, 0, 2]], name="ragged")


# ------------------------------------------------------------------ multi-tile


@pytest.fixture(scope="module")
def cube_grid():
    sys.path.insert(0, os.path.dirname(__file__))
    from test_cube_left_grid import cube_left_grid
    grid, _psi = cube_left_grid()
    return grid


def test_cubed_sphere_has_the_corner_count_euler_demands(cube_grid):
    """
    A cubed sphere of `N x N` cells per face has `6N^2 + 2` distinct corner points:
    `6N^2` of degree 4 and the 8 cube vertices of degree 3. That count follows from
    the polyhedron, not from any implementation.
    """
    ct = CornerTopology(cube_grid)
    N = ct.Nxc
    assert ct.n_nodes == 6 * N * N + 2
    v = ct.valence
    assert (np.unique(v) <= 4).all()
    assert int((v == 3).sum()) == 8


def test_multitile_identity_never_exceeds_the_old_coordinate_based_one(cube_grid):
    """
    Where the previous implementation merged two slots from cell identity, this one
    must agree; where it merged them from *coordinates*, this one is allowed to keep
    them apart. So the new partition must refine the old one, never coarsen it.
    """
    _assert_refines(CornerTopology(cube_grid), outer_topology(cube_grid))


def _assert_refines(ct, ot):
    new = ct.node_id.ravel()
    old = ot.node_id.ravel()
    keep = old >= 0
    order = np.lexsort((old[keep], new[keep]))
    n, o = new[keep][order], old[keep][order]
    # within each new node, every old label must be the same
    boundary = np.flatnonzero(np.diff(n)) + 1
    for lo, hi in zip(np.r_[0, boundary], np.r_[boundary, n.size]):
        assert len(np.unique(o[lo:hi])) == 1, (
            f"new node {n[lo]} spans old nodes {np.unique(o[lo:hi])}: "
            "the declared topology merged corners the cell-identity one kept apart"
        )


# ------------------------------------------------------------------- real grids


_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_MOM6 = os.path.join(_DATA, "MOM6_global_example_sigma2_budgets_v0_0_6.nc")
_ECCO = os.path.join(_DATA, "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc")


@fold_only
@pytest.mark.skipif(not os.path.exists(_MOM6), reason="needs the MOM6 example grid")
def test_real_tripolar_grid_matches_its_declaration():
    ds = xr.open_dataset(_MOM6)
    grid = xgcm.Grid(
        ds,
        coords={"X": {"center": "xh", "outer": "xq"},
                "Y": {"center": "yh", "outer": "yq"}},
        padding={"X": "periodic", "Y": {"fold": "corner"}},
        autoparse_metadata=False,
    )
    ct = CornerTopology(grid)
    n_pairs = (ct.Nxc - 2) // 2
    assert ct.n_nodes == (ct.Nyc + 1) * ct.Nxc - n_pairs

    # The southern edge is a wall, so its corners have three edges each; every
    # interior corner has four; the two fold pivots have two.
    counts = dict(zip(*[c.tolist() for c in np.unique(ct.valence, return_counts=True)]))
    assert counts[3] == ct.Nxc          # the walled southern row
    assert counts[2] == 2               # the two poles of the bipolar cap
    assert max(counts) == 4

    # Geographically coincident corners stay distinct where the grid says they are.
    # The bipolar cap's singular meridian stores one position for a whole column of
    # corners; each is still its own node, so a section through the cap knows which
    # of the fan's sectors it passed between.
    lon = ds["geolon_c"].values
    lat = ds["geolat_c"].values
    col = ct.node_id[0, 140:181, 0]
    assert len(np.unique(col)) == col.size
    assert np.allclose(lon[140:181, 0], lon[140, 0])
    assert np.allclose(lat[140:181, 0], lat[140, 0])


@pytest.mark.skipif(not os.path.exists(_ECCO), reason="needs the ECCO LLC90 grid")
def test_llc90_identity_refines_the_coordinate_based_one():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
    from load_example_ECCO_grid import load_ECCO_LLC90_grid

    grid = load_ECCO_LLC90_grid()
    ct = CornerTopology(grid)
    _assert_refines(ct, outer_topology(grid))
    # every tile edge that `face_connections` names is resolved; the four it leaves
    # `None` are the domain's southern boundary and stay unglued.
    assert len(ct.identifications) == 13 * 4 - 4
    assert (ct.valence <= 4).all()


# ----------------------------------------------------------------- performance


@pytest.mark.slow
def test_topology_build_scales_to_production_resolution():
    """
    The corner topology is built for every grid on every section, so it has to be
    affordable at the resolutions people actually run. Identity is O(seam) and
    labelling O(cells), both vectorized, so the cost is linear in corner count.
    """
    def build(nx, ny):
        grid = _single_tile_grid(nx, ny)
        t0 = time.perf_counter()
        ct = CornerTopology(grid)
        ct.valence
        return time.perf_counter() - t0, ct.n_slots

    t_small, n_small = build(360, 270)          # ~1e5 slots
    t_big, n_big = build(1440, 1080)            # ~1.6e6 slots, a quarter degree

    assert t_big < 15.0, f"a quarter-degree grid took {t_big:.1f}s to resolve"
    # linear, not quadratic: 16x the slots must not cost anywhere near 16^2 the time
    assert t_big / max(t_small, 1e-3) < 60.0
