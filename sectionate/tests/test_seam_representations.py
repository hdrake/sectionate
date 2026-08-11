"""
Tests for the seam representation contract of `uvindices_from_qindices`.

A seam gives one physical corner more than one index, so a section crossing it can be
written two ways: *seam-explicit*, carrying both indices (the step between them spans no
grid cell), or *twin-free*, carrying only one. Both describe the same path, and sectionate
accepts both -- which is what lets a section arrive from `grid_section`, from a mask traced
on the grid (`regionate.boundaries`, whose single-tile back-end deliberately keeps both
coincident corners at a seam junction), or from saved indices reloaded off disk.

The contract these pin down:

* **input is liberal** -- either representation is accepted and yields the *same* faces;
* **output is strict** -- the returned faces are all real velocity faces, so a step between
  two indices of one physical corner emits nothing;
* **"q" relates the two** -- faces and corner steps are therefore not one-to-one, and "q"
  says which corner step each face came from.
"""
import warnings

import numpy as np
import pytest

from sectionate.section import grid_section, distance_on_unit_sphere, COINCIDENT_TOLERANCE_M
from sectionate.transports import convergent_transport, uvindices_from_qindices
from sectionate.gridutils import get_geo_corners

from test_section_fold import (
    _pinched_fold_grid,
    _add_streamfunction_transports,
    _xgcm_supports_fold,
)
from test_section_multitile import _two_face_transport_grid

pytestmark = pytest.mark.skipif(
    not _xgcm_supports_fold(),
    reason="installed xgcm lacks the bipolar north-fold boundary",
)


def _corner_coords(grid, i_c, j_c, f_c=None):
    geo = get_geo_corners(grid)
    glon, glat = np.asarray(geo["X"].values), np.asarray(geo["Y"].values)
    if f_c is None:
        return glon[j_c, i_c], glat[j_c, i_c]
    return glon[f_c, j_c, i_c], glat[f_c, j_c, i_c]


def _coincident_steps(grid, i_c, j_c, f_c=None):
    """Indices k where corners k and k+1 are the same physical point."""
    lon, lat = _corner_coords(grid, i_c, j_c, f_c)
    d = distance_on_unit_sphere(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return set(np.flatnonzero(d < COINCIDENT_TOLERANCE_M).tolist())


def _to_twin_free(grid, i_c, j_c):
    """Delete the second index of every coincident consecutive pair."""
    drop = {k + 1 for k in _coincident_steps(grid, i_c, j_c)}
    keep = [k for k in range(len(i_c)) if k not in drop]
    return np.asarray(i_c)[keep], np.asarray(j_c)[keep]


def _wrap_section(grid):
    """A section crossing the periodic-X seam, which `grid_section` returns seam-explicit."""
    geo = get_geo_corners(grid)
    glon, glat = np.asarray(geo["X"].values), np.asarray(geo["Y"].values)
    i_c, j_c, _, _ = grid_section(
        grid,
        [float(glon[1, 1]), float(glon[1, 13])],
        [float(glat[1, 1]), float(glat[1, 13])],
    )
    return i_c, j_c


def test_both_seam_representations_give_the_same_faces():
    """The two ways of writing a seam crossing name identical velocity faces.

    `grid_section` returns the periodic-X crossing seam-explicit, so deleting the duplicate
    corner produces the twin-free form of the very same path. Nothing downstream may depend
    on which one it was handed.
    """
    grid = _pinched_fold_grid()
    i_x, j_x = _wrap_section(grid)
    assert _coincident_steps(grid, i_x, j_x), "fixture must actually cross the seam"

    i_f, j_f = _to_twin_free(grid, i_x, j_x)
    assert len(i_f) < len(i_x)

    explicit = uvindices_from_qindices(grid, i_x, j_x)
    free = uvindices_from_qindices(grid, i_f, j_f)

    for key in ("var", "i", "j", "Xinc", "Yinc"):
        np.testing.assert_array_equal(
            explicit[key], free[key], err_msg=f"{key} differs between representations"
        )


def _mirror_pair_boundary():
    """The boundary of two cells that are mirror images across the fold seam.

    This is the shape a mask tracer produces for a region joined only through the fold: the
    loop hands off between the seam corner's two indices twice, and each hand-off is a
    coincident pair. On `_pinched_fold_grid(16, 6)` the seam row is j=5, mirrored under
    i <-> 16-i, so cells (4, 1) and (4, 14) are a mirror pair sharing the seam face.
    """
    j_c = np.array([5, 4, 4, 5, 5, 4, 4, 5, 5])
    i_c = np.array([1, 1, 2, 2, 14, 14, 15, 15, 1])
    return i_c, j_c


def test_mask_traced_fold_loop_is_accepted_and_counts_each_face_once():
    """A seam-explicit loop straddling the fold resolves to its real faces.

    Before physical coincidence was tested ahead of index adjacency, the hand-off pair
    (j=5, i=2) -> (j=5, i=14) -- already a zero-length twin edge, but 12 apart in the index
    lattice -- was sent down the twin-splicing path and raised.
    """
    grid = _pinched_fold_grid()
    i_c, j_c = _mirror_pair_boundary()
    assert _coincident_steps(grid, i_c, j_c) == {3, 7}

    uv = uvindices_from_qindices(grid, i_c, j_c)

    # 8 corner steps, 2 of them hand-offs -> 6 faces: three per cell. The cells' shared
    # seam face is interior to the pair and is correctly not among them.
    assert uv["var"].size == 6
    faces = list(zip(uv["var"].tolist(), uv["i"].tolist(), uv["j"].tolist()))
    assert len(set(faces)) == 6, f"a face was named twice: {faces}"
    assert all(v in ("U", "V") for v in uv["var"])


def test_mask_traced_fold_loop_carries_zero_net_transport():
    """The loop encloses two cells of an exactly non-divergent flow, so the transport
    convergence over it is zero -- the discrete divergence theorem across the fold seam.
    Naming the mirror image of a crossed face, or counting one twice, breaks this."""
    grid = _pinched_fold_grid()
    _, tgrid = _add_streamfunction_transports(grid)
    i_c, j_c = _mirror_pair_boundary()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = convergent_transport(tgrid, i_c, j_c, utr="utest", vtr="vtest")
    total = float(out["conv_mass_transport"].sum().values)

    assert abs(total) < 1.0e-12, f"net convergence {total} should vanish"


def _assert_q_contract(grid, i_c, j_c, f_c=None):
    uv = uvindices_from_qindices(grid, i_c, j_c, f_c=f_c)
    q = uv["q"]
    n = len(i_c)

    assert q.size == uv["var"].size
    assert np.all(np.diff(q) > 0), f"q must be strictly increasing, got {q}"
    assert q.min() >= 0 and q.max() + 1 <= n - 1, "face k must span corners q[k], q[k]+1"

    # The steps q skips are exactly the ones that carried no face.
    skipped = set(range(n - 1)) - set(q.tolist())
    assert skipped == _coincident_steps(grid, i_c, j_c, f_c), (
        f"q skipped {sorted(skipped)}, but the coincident steps are "
        f"{sorted(_coincident_steps(grid, i_c, j_c, f_c))}"
    )
    assert q.size == (n - 1) - len(skipped)
    return uv


def test_q_maps_faces_back_to_corner_steps_across_a_periodic_seam():
    grid = _pinched_fold_grid()
    i_c, j_c = _wrap_section(grid)
    uv = _assert_q_contract(grid, i_c, j_c)
    assert uv["q"].size < len(i_c) - 1, "the seam hand-off must leave a gap in q"


def test_q_maps_faces_back_to_corner_steps_on_a_mask_traced_loop():
    grid = _pinched_fold_grid()
    i_c, j_c = _mirror_pair_boundary()
    uv = _assert_q_contract(grid, i_c, j_c)
    assert uv["q"].tolist() == [0, 1, 2, 4, 5, 6]


def test_q_is_the_identity_when_no_corner_repeats():
    """A section that touches no seam has one face per corner step, and q says so."""
    grid = _pinched_fold_grid()
    geo = get_geo_corners(grid)
    glon, glat = np.asarray(geo["X"].values), np.asarray(geo["Y"].values)
    i_c, j_c, _, _ = grid_section(
        grid,
        [float(glon[0, 3]), float(glon[0, 6])],
        [float(glat[0, 3]), float(glat[0, 6])],
    )
    assert not _coincident_steps(grid, i_c, j_c)
    uv = _assert_q_contract(grid, i_c, j_c)
    np.testing.assert_array_equal(uv["q"], np.arange(len(i_c) - 1))


def test_q_contract_holds_on_a_multitile_grid():
    """Multi-tile corners resolve through the corner-node graph rather than by coordinate
    coincidence, but the reported relation between faces and corner steps is the same."""
    grid = _two_face_transport_grid()
    geo = get_geo_corners(grid)
    glon, glat = np.asarray(geo["X"].values), np.asarray(geo["Y"].values)
    i_c, j_c, f_c, _, _ = grid_section(
        grid,
        [float(glon[0, 1, 1]), float(glon[1, 1, 2])],
        [float(glat[0, 1, 1]), float(glat[1, 1, 2])],
    )
    assert len(set(f_c.tolist())) > 1, "fixture must actually cross the face seam"
    _assert_q_contract(grid, i_c, j_c, f_c=f_c)
