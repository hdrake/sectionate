"""Tests for sections on single-tile grids with a bipolar north-fold boundary.

The fold is expressed purely as an xgcm boundary, e.g.
``padding={"X": "periodic", "Y": {"fold": "corner"}}``; sectionate derives the
seam connectivity from xgcm's padding with no fold-specific code of its own.
"""
import os

import numpy as np
import pytest
import xarray as xr
import xgcm

from sectionate.gridutils import get_geo_corners, build_neighbor_maps
from sectionate.section import grid_section


def _fold_grid(nx=8, ny=5, pivot="corner"):
    """A minimal single-tile C-grid, X-periodic with a bipolar north fold."""
    xq, yq = np.arange(nx), np.arange(ny)
    xh, yh = np.arange(nx) + 0.5, np.arange(ny) + 0.5
    lon_c, lat_c = np.meshgrid(xq * (360.0 / nx), yq)
    lon, lat = np.meshgrid(xh * (360.0 / nx), yh)
    ds = xr.Dataset(coords={
        "xq": xq, "yq": yq, "xh": xh, "yh": yh,
        "geolon_c": (("yq", "xq"), lon_c), "geolat_c": (("yq", "xq"), lat_c),
        "geolon": (("yh", "xh"), lon), "geolat": (("yh", "xh"), lat),
    })
    return xgcm.Grid(
        ds,
        coords={"X": {"center": "xh", "right": "xq"}, "Y": {"center": "yh", "right": "yq"}},
        padding={"X": "periodic", "Y": {"fold": pivot}},
        autoparse_metadata=False,
    )


def _xgcm_supports_fold():
    """Whether the installed xgcm accepts `padding={"Y": {"fold": ...}}`, the bipolar
    north-fold boundary these tests are about. Older releases reject it."""
    try:
        _fold_grid(4, 3)
        return True
    except Exception:
        return False


# Skip the whole module on xgcm releases without the bipolar-fold boundary.
pytestmark = pytest.mark.skipif(
    not _xgcm_supports_fold(),
    reason="installed xgcm does not accept padding={'Y': {'fold': ...}}",
)


def test_fold_neighbor_connectivity():
    """The top row's `up` neighbors cross the seam (fold), unlike a clipped edge."""
    nx, ny = 8, 5
    fold = _fold_grid(nx, ny)
    fold_maps = build_neighbor_maps(fold, get_geo_corners(fold))
    top = ny - 1
    up_f, up_j, up_i = fold_maps["up"]

    # Single-tile: no face dimension.
    assert up_f is None
    # The fold folds the top row down by one row onto mirrored columns: every
    # seam corner's `up` neighbor sits on row top-1 (not a self-wall) at a
    # different column (the zonal mirror).
    assert np.all(up_j[top] == top - 1)
    assert np.any(up_i[top] != np.arange(nx))

    # Contrast with a plain extend edge, where `up` of the top row is a wall
    # (the point itself) -- i.e. the fold connectivity is genuinely being used.
    extend = _fold_grid(nx, ny)
    extend = xgcm.Grid(
        extend._ds,
        coords={"X": {"center": "xh", "right": "xq"}, "Y": {"center": "yh", "right": "yq"}},
        padding={"X": "periodic", "Y": "extend"},
        autoparse_metadata=False,
    )
    ext_maps = build_neighbor_maps(extend, get_geo_corners(extend))
    assert np.all(ext_maps["up"][1][top] == top)        # clipped to self
    assert np.all(ext_maps["up"][2][top] == np.arange(nx))


def _mom6_example_path():
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(
        here, "..", "..", "data", "MOM6_global_example_sigma2_budgets_v0_0_6.nc"
    ))


@pytest.mark.skipif(
    not os.path.exists(_mom6_example_path()),
    reason="MOM6 example dataset not present (download via examples/load_example_model_grid.py)",
)
def test_fold_section_crosses_arctic_on_real_grid():
    """A high-latitude zonal section that clips on an `extend` grid traces the full
    latitude circle once the same grid declares its bipolar fold."""
    ds = xr.open_dataset(_mom6_example_path())
    coords = {"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}}
    lons = np.arange(0.0, 360.0 + 5.0, 5.0)
    lats = [80.0] * len(lons)

    extend = xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": "extend"},
                       autoparse_metadata=False)
    with pytest.raises(RuntimeError):
        grid_section(extend, lons, lats)  # cannot cross the fold without it

    fold = xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": {"fold": "corner"}},
                     autoparse_metadata=False)
    i_c, j_c, lons_c, lats_c = grid_section(fold, lons, lats)
    # The traced path hugs the requested latitude (it did not wander off the seam).
    assert np.all(np.abs(np.asarray(lats_c) - 80.0) < 1.0)
    # It reaches the northern fold row.
    assert j_c.max() == ds.yq.size - 1


@pytest.mark.skipif(
    not os.path.exists(_mom6_example_path()),
    reason="MOM6 example dataset not present (download via examples/load_example_model_grid.py)",
)
def test_fold_grid_sections_contain_no_repeated_corner():
    """On the real MOM6 tripolar grid, sections that cross the periodic seam used to carry
    the seam meridian twice (it is stored as both corner column 0 and corner column nx).
    `grid_section` now drops the redundant index, so no consecutive pair of corners is
    the same physical point and every consecutive pair is a real velocity face."""
    from sectionate.section import distance_on_unit_sphere, COINCIDENT_TOLERANCE_M
    from sectionate.transports import uvindices_from_qindices

    ds = xr.open_dataset(_mom6_example_path())
    coords = {"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}}
    fold = xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": {"fold": "corner"}},
                     autoparse_metadata=False)

    cases = [
        # a closed Arctic polygon that crosses the seam twice
        (dict(lons=[40., 60., 80., 40.], lats=[60., 60., 75., 60.]), {}),
        # a full southern latitude circle, crossing the seam once
        (dict(lons=[0., 120., 240., 360.], lats=[-60.] * 4), dict(curve="latitude circle")),
        # a high-latitude circle that reaches the fold row
        (dict(lons=np.arange(0., 365., 5.).tolist(), lats=[80.] * 73), {}),
    ]
    for section, kwargs in cases:
        i_c, j_c, lons_c, lats_c = grid_section(
            fold, section["lons"], section["lats"], **kwargs
        )
        edge = distance_on_unit_sphere(lons_c[:-1], lats_c[:-1], lons_c[1:], lats_c[1:])
        assert np.all(edge > COINCIDENT_TOLERANCE_M), "section repeats a physical corner"

        uv = uvindices_from_qindices(fold, i_c, j_c)
        assert uv["var"].size == i_c.size - 1
        assert set(np.unique(uv["var"]).tolist()) <= {"U", "V"}
        # cell-center indices stay within the grid
        assert np.all(uv["i"][uv["var"] == "V"] < ds.xh.size)
        assert np.all(uv["j"][uv["var"] == "U"] < ds.yh.size)


@pytest.mark.skipif(
    not os.path.exists(_mom6_example_path()),
    reason="MOM6 example dataset not present (download via examples/load_example_model_grid.py)",
)
def test_meridional_section_through_pole_column_stays_in_range():
    """A meridional section along 120W runs straight into the corner column where the
    bipolar cap converges on one of its poles: 41 corner rows that are all one physical
    point. Collapsing that run turns the step into the pole into a jump of 41 rows along
    "Y" -- which must NOT be read as a periodic wrap, because a bipolar-fold grid's "Y"
    axis does not wrap. Read as one, the U face lands on `j = ny`, one past the last cell
    center, and reading its transport goes out of bounds. Assert the invariant rather than
    the indices: every velocity index sits inside its own centre axis."""
    from sectionate.transports import uvindices_from_qindices, convergent_transport

    ds = xr.open_dataset(_mom6_example_path())
    coords = {"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}}
    fold = xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": {"fold": "corner"}},
                     autoparse_metadata=False)

    # 120W is one of this grid's two bipoles; its neighbourhood must behave the same way.
    for lon in (-121., -120.5, -120., -119.5, -119.):
        i_c, j_c, lons_c, lats_c = grid_section(fold, [lon, lon], [50., 85.])
        uv = uvindices_from_qindices(fold, i_c, j_c)
        # U lives at (X-corner, Y-center) and V at (X-center, Y-corner): each index must
        # be inside the dimension it is staggered onto.
        assert uv["j"][uv["var"] == "U"].max() < ds.sizes["yh"]
        assert uv["i"][uv["var"] == "U"].max() < ds.sizes["xq"]
        assert uv["i"][uv["var"] == "V"].max() < ds.sizes["xh"]
        assert uv["j"][uv["var"] == "V"].max() < ds.sizes["yq"]
        # ...so the transport can actually be read.
        conv = convergent_transport(fold, i_c, j_c, utr="umo", vtr="vmo")
        assert np.isfinite(np.nansum(conv["conv_mass_transport"].values))


@pytest.mark.skipif(
    not os.path.exists(_mom6_example_path()),
    reason="MOM6 example dataset not present (download via examples/load_example_model_grid.py)",
)
def test_section_through_a_pole_column_raises_rather_than_naming_a_wrong_face():
    """A section can reach a bipole along one index of its degenerate column and leave along
    another, far apart in the index lattice -- here in along the 60E meridian at corner row
    180 and out along corner row 140. No single index of that corner is adjacent to both
    neighbours, so no section with distinct consecutive corners carries both flanking faces
    (keeping either one turns the face on the other side from a V into a U). Refuse it and
    name the corner instead of emitting a face the section does not have."""
    ds = xr.open_dataset(_mom6_example_path())
    coords = {"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}}
    # The fold left undeclared: the walk cannot cross the seam, so it is pushed through the
    # pole column instead of around it.
    extend = xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": "extend"},
                       autoparse_metadata=False)
    with pytest.raises(ValueError, match="degenerate"):
        grid_section(extend, [188.437, 32.017], [84.876, 13.851])

    # Declaring the fold routes the same section around the pole, and it traces.
    fold = xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": {"fold": "corner"}},
                     autoparse_metadata=False)
    i_c, j_c, lons_c, lats_c = grid_section(fold, [188.437, 32.017], [84.876, 13.851])
    assert i_c.size > 1
