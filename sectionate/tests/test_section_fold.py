"""Tests for sections on single-tile grids with a bipolar north-fold boundary.

The fold is expressed purely as an xgcm boundary, e.g.
``padding={"X": "periodic", "Y": {"fold": "corner"}}``; sectionate derives the
seam connectivity from xgcm's padding with no fold-specific code of its own.
"""
import os
import warnings

import numpy as np
import pytest
import xarray as xr
import xgcm

from sectionate.gridutils import get_geo_corners, build_neighbor_maps
from sectionate.section import grid_section, distance_on_unit_sphere
from sectionate.transports import convergent_transport, uvindices_from_qindices


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
    """The bipolar-fold boundary is only available in newer xgcm (xgcm/xgcm#711)."""
    try:
        _fold_grid(4, 3)
        return True
    except Exception:
        return False


# Skip the whole module on xgcm releases without the bipolar-fold boundary.
pytestmark = pytest.mark.skipif(
    not _xgcm_supports_fold(),
    reason="installed xgcm lacks the bipolar north-fold boundary (xgcm/xgcm#711)",
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


# ---------------------------------------------------------------------------
# Transports across the fold seam.
#
# The seam row identifies each corner with its mirrored column, so the same
# physical corner carries two indices. Crossing the seam, the walk steps from one
# of them to a corner adjacent to the *other*, which is a physical edge but not an
# index-adjacent one. The tests below pin down that such a crossing names the
# velocity face the section actually crossed, with the right sign, exactly once.
#
# They use a streamfunction fixture: psi is a single-valued function of *physical*
# position, so seam twins get identical psi, and transports are built from
# index-space differences of psi. That makes the flow exactly non-divergent and the
# answer for any section known analytically -- transport = psi(end) - psi(start) --
# while automatically reproducing the duplicated, sign-flipped seam row of an
# ORCA-style fold (NEMO 4.2 manual, Appendix E) without hard-coding the flip.
# ---------------------------------------------------------------------------


def _pinched_fold_grid(nxh=16, ny=6):
    """
    A single-tile C-grid that genuinely carries the corner-pivot fold identity.

    Picture a cylinder (X-periodic) whose cross-section is pinched shut as `j` rises:
    at row `j` the corners sit on an ellipse with semi-axes ``(a_j, b_j)``, and
    ``b_{ny-1} = 0``, so the top row collapses onto a segment traversed out and back.
    Under ``i <-> nxh - i`` that row is therefore *exactly* mirror-symmetric, which is
    the corner-pivot fold identity, with its two fold poles at ``i = 0`` and
    ``i = nxh/2``. ``a_j`` grows with `j` so the ellipses nest and every corner below
    the seam stays distinct.

    Unlike `_fold_grid`, whose plain lat/lon coordinates do not carry the fold its
    metadata declares, this one is geometrically consistent with the fold, so
    transports across the seam are well defined.
    """
    i_q, j_q = np.arange(nxh + 1), np.arange(ny)
    i_h, j_h = np.arange(nxh) + 0.5, np.arange(ny - 1) + 0.5

    def lonlat(jj, ii):
        theta = 2.0 * np.pi * np.asarray(ii) / nxh
        t = np.asarray(jj) / (ny - 1)
        x = (1.0 + 0.5 * t) * np.cos(theta)
        y = (1.0 - t) * np.sin(theta)
        return 25.0 * x, 45.0 + 18.0 * y

    lon_c, lat_c = lonlat(*np.meshgrid(j_q, i_q, indexing="ij"))
    lon_h, lat_h = lonlat(*np.meshgrid(j_h, i_h, indexing="ij"))

    ds = xr.Dataset(coords={
        "xq": i_q, "yq": j_q, "xh": i_h, "yh": j_h,
        "geolon_c": (("yq", "xq"), lon_c), "geolat_c": (("yq", "xq"), lat_c),
        "geolon": (("yh", "xh"), lon_h), "geolat": (("yh", "xh"), lat_h),
    })
    return xgcm.Grid(
        ds,
        coords={"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}},
        padding={"X": "periodic", "Y": {"fold": "corner"}},
        autoparse_metadata=False,
    )


def _streamfunction(lon, lat):
    """A degree-1 spherical harmonic: smooth everywhere (poles included) and
    single-valued in physical space, so seam twins receive identical values."""
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    return (0.3 * np.cos(la) * np.cos(lo)
            - 0.8 * np.cos(la) * np.sin(lo)
            + 0.5 * np.sin(la))


def _add_streamfunction_transports(grid, utr="utest", vtr="vtest"):
    """Attach exactly non-divergent transports derived from `_streamfunction`.

    Returns the corner-array streamfunction and a grid carrying the transports.
    `umo[j,i] = psi[j,i] - psi[j+1,i]` and `vmo[j,i] = psi[j,i+1] - psi[j,i]` make
    the transport across any section telescope to `psi(end) - psi(start)`.
    """
    geo = get_geo_corners(grid)
    psi = _streamfunction(np.asarray(geo["X"].values), np.asarray(geo["Y"].values))
    ydim, xdim = geo["X"].dims[-2], geo["X"].dims[-1]
    ycenter = grid.axes["Y"].coords["center"]
    xcenter = grid.axes["X"].coords["center"]

    ds = grid._ds.copy()
    ds[utr] = xr.DataArray(psi[:-1, :] - psi[1:, :], dims=(ycenter, xdim))
    ds[vtr] = xr.DataArray(psi[:, 1:] - psi[:, :-1], dims=(ydim, xcenter))
    regridded = xgcm.Grid(
        ds,
        coords={ax: dict(grid.axes[ax].coords) for ax in ("X", "Y")},
        padding={ax: grid.axes[ax].padding for ax in ("X", "Y")},
        autoparse_metadata=False,
    )
    return psi, regridded


def _section_transport(grid, i_c, j_c, utr="utest", vtr="vtest"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # open sections warn about `positive_in`
        out = convergent_transport(grid, i_c, j_c, utr=utr, vtr=vtr)
    return float(out["conv_mass_transport"].sum().values)


def test_pinched_fold_grid_carries_the_fold_symmetry():
    """The fixture really does duplicate its seam row, and the streamfunction
    transports really do reproduce the ORCA duplicate-with-sign-flip structure."""
    nxh, ny = 16, 6
    grid = _pinched_fold_grid(nxh, ny)
    geo = get_geo_corners(grid)
    lon, lat = np.asarray(geo["X"].values), np.asarray(geo["Y"].values)
    top = ny - 1

    # Seam row: corner i and its mirror nxh-i are the same physical point ...
    mirrored = distance_on_unit_sphere(lon[top], lat[top], lon[top, ::-1], lat[top, ::-1])
    assert np.nanmax(mirrored) < 1.0e-6
    # ... while the row just below it holds two genuinely distinct halves.
    below = distance_on_unit_sphere(lon[top-1], lat[top-1], lon[top-1, ::-1], lat[top-1, ::-1])
    assert below[1:nxh // 2].min() > 1.0e4

    psi, tgrid = _add_streamfunction_transports(grid)
    v = tgrid._ds["vtest"].values
    # The seam row's V transports are duplicates carrying a sign flip, and the
    # transports are exactly non-divergent -- neither was put in by hand.
    assert np.abs(v[top] + v[top, ::-1]).max() < 1.0e-12 * max(np.abs(v[top]).max(), 1.0)
    u = tgrid._ds["utest"].values
    conv = (u[:, :-1] - u[:, 1:]) + (v[:-1, :] - v[1:, :])
    assert np.abs(conv).max() < 1.0e-12


def test_transport_crossing_the_fold_seam_is_exact():
    """A path that goes up to the seam and back down the mirrored side must transport
    exactly `psi(end) - psi(start)`, using each velocity face once.

    Before the seam-twin fix this crossing was attributed to the mirror column's face
    with the wrong sign, so the total was wrong by ~2x that face's transport.
    """
    nxh, ny = 16, 6
    top = ny - 1
    psi, grid = _add_streamfunction_transports(_pinched_fold_grid(nxh, ny))

    for i0 in (3, 5, 11):
        mirror = nxh - i0
        # up to the seam at column i0, across the fold, back down at the mirror column
        j_c = np.array([top - 2, top - 1, top, top - 1, top - 2])
        i_c = np.array([i0, i0, i0, mirror, mirror])

        got = _section_transport(grid, i_c, j_c)
        want = psi[j_c[-1], i_c[-1]] - psi[j_c[0], i_c[0]]
        assert got == pytest.approx(want, abs=1.0e-12), f"column {i0}"

        # no velocity face is used twice, and the zero-length twin edge emits none
        uv = uvindices_from_qindices(grid, i_c, j_c)
        faces = list(zip(uv["var"].tolist(), uv["i"].tolist(), uv["j"].tolist()))
        assert len(faces) == len(set(faces)) == 4


def test_transport_along_the_fold_seam_is_exact():
    """Running *along* the duplicated, sign-flipped seam row needs no fold-specific
    handling: the traversal direction and the stored velocity share an index frame,
    so the flip cancels. This already held before the fix and must keep holding."""
    nxh, ny = 16, 6
    top = ny - 1
    psi, grid = _add_streamfunction_transports(_pinched_fold_grid(nxh, ny))

    for i0, i1 in ((2, 6), (6, 12), (4, 12)):   # last one straddles the fold pivot
        i_c = np.concatenate([np.arange(i0, i1 + 1), [i1]])
        j_c = np.concatenate([np.full(i1 - i0 + 1, top), [top - 1]])
        got = _section_transport(grid, i_c, j_c)
        want = psi[j_c[-1], i_c[-1]] - psi[j_c[0], i_c[0]]
        assert got == pytest.approx(want, abs=1.0e-12), f"columns {i0}..{i1}"


def test_fold_crossing_raises_when_coordinates_lack_the_fold_symmetry():
    """`_fold_grid` declares a fold its plain lat/lon coordinates do not carry, so the
    seam has no twins and the crossing edge is genuinely undefined. Say so, rather
    than silently attributing it to some other face."""
    nxh, ny = 8, 5
    grid = _fold_grid(nxh, ny)
    j_c = np.array([ny - 2, ny - 1, ny - 2])
    i_c = np.array([3, 3, nxh - 3])
    with pytest.raises(ValueError, match="fold"):
        uvindices_from_qindices(grid, i_c, j_c)


@pytest.mark.skipif(
    not os.path.exists(_mom6_example_path()),
    reason="MOM6 example dataset not present (download via examples/load_example_model_grid.py)",
)
def test_fold_crossing_transport_is_exact_on_real_tripolar_grid():
    """The real-grid counterpart: traced sections that cross the tripolar seam --
    a closed Arctic latitude circle and an open path over the pole -- transport
    exactly `psi(end) - psi(start)`, and use no velocity face twice."""
    ds = xr.open_dataset(_mom6_example_path())
    coords = {"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}}
    grid = xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": {"fold": "corner"}},
                     autoparse_metadata=False)
    psi, tgrid = _add_streamfunction_transports(grid)

    sections = {
        "latitude circle 80N": (np.arange(0.0, 360.0 + 5.0, 5.0), [80.0] * 73, "latitude circle"),
        "meridional over pole": ([100.0, 100.0, 280.0, 280.0], [60.0, 89.0, 89.0, 60.0], "great circle"),
    }
    for name, (lons, lats, curve) in sections.items():
        i_c, j_c, _, _ = grid_section(grid, lons, lats, curve=curve)
        assert j_c.max() == ds.yq.size - 1, f"{name} never reached the seam"

        got = _section_transport(tgrid, i_c, j_c)
        want = psi[j_c[-1], i_c[-1]] - psi[j_c[0], i_c[0]]
        assert got == pytest.approx(want, abs=1.0e-12), name

        uv = uvindices_from_qindices(grid, i_c, j_c)
        faces = list(zip(uv["var"].tolist(), uv["i"].tolist(), uv["j"].tolist()))
        assert len(faces) == len(set(faces)), f"{name} counted a velocity face twice"
