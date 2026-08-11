import numpy as np
import xarray as xr
import xgcm
import pytest

from sectionate.gridutils import (
    get_geo_corners,
    build_neighbor_maps,
)
from sectionate.section import drop_repeated_corners


# define simple lat-lon grid
lon, lat = np.meshgrid(np.arange(360), np.arange(-80, 81))
ds = xr.Dataset()
ds["lon"] = xr.DataArray(lon, dims=("y", "x"))
ds["lat"] = xr.DataArray(lat, dims=("y", "x"))


def _latlon_grid(vertical_axis=False):
    """The lat-lon grid above (X-periodic, Y clipped), optionally also registering a
    vertical axis -- as every real model grid does, and as sections never traverse."""
    ny, nx = lat.shape
    ds = xr.Dataset(coords={
        "xq": np.arange(nx), "yq": np.arange(ny),
        "xh": np.arange(nx) + 0.5, "yh": np.arange(ny) + 0.5,
        "geolon_c": (("yq", "xq"), lon.astype(float)),
        "geolat_c": (("yq", "xq"), lat.astype(float)),
        "geolon": (("yh", "xh"), lon.astype(float)),
        "geolat": (("yh", "xh"), lat.astype(float)),
    })
    coords = {"X": {"center": "xh", "right": "xq"}, "Y": {"center": "yh", "right": "yq"}}
    if vertical_axis:
        ds = ds.assign_coords({"z_l": ("z_l", np.array([5., 15.])),
                               "z_i": ("z_i", np.array([0., 10., 20.]))})
        coords["Z"] = {"center": "z_l", "outer": "z_i"}
    return xgcm.Grid(ds, coords=coords, padding={"X": "periodic", "Y": "extend"},
                     autoparse_metadata=False)


def _latlon_neighbor_maps(vertical_axis=False):
    """Neighbor maps for the lat-lon grid above, built the same way `grid_section`
    does -- from an xgcm.Grid. The low-level pathfinder always requires these; a grid
    is the only source of topology-aware connectivity."""
    g = _latlon_grid(vertical_axis=vertical_axis)
    return build_neighbor_maps(g, get_geo_corners(g))


def test_vertical_axis_does_not_affect_neighbor_maps():
    """Horizontal connectivity must not depend on whether the grid also registers a
    vertical axis. `build_neighbor_maps` padded its index arrays over *every* axis of
    the grid, and xgcm's `pad` raises on an axis the array has no dimension for, so a
    Z axis made an otherwise ordinary grid untraceable."""
    plain = _latlon_neighbor_maps()
    with_z = _latlon_neighbor_maps(vertical_axis=True)
    assert set(plain) == set(with_z)
    for d in plain:
        for a, b in zip(plain[d], with_z[d]):
            if a is None or b is None:
                assert a is None and b is None
            else:
                np.testing.assert_array_equal(a, b)


def test_grid_section_on_grid_with_vertical_axis():
    """End-to-end: a section traced on a grid registering X, Y and Z is identical to
    the same section traced on the horizontal-only view of that grid."""
    from sectionate.section import grid_section
    lons, lats = [10., 40.], [-10., 20.]
    plain = grid_section(_latlon_grid(), lons, lats)
    with_z = grid_section(_latlon_grid(vertical_axis=True), lons, lats)
    for a, b in zip(plain, with_z):
        np.testing.assert_array_equal(a, b)


def test_distance_on_unit_sphere():
    from sectionate.section import distance_on_unit_sphere

    # test of few points with unit radius
    d = distance_on_unit_sphere(0, 0, 1.e-20, 0, R=1.)
    assert np.isclose(d, 0., atol=1.e-14)
    d = distance_on_unit_sphere(0, 0, 360, 0, R=1.)
    assert np.isclose(d, 0., atol=1.e-14)
    d = distance_on_unit_sphere(0, 45, 0, -45, R=1.)
    assert np.isclose(d, np.pi/2, atol=1.e-14)
    d = distance_on_unit_sphere(0, 0, 180, 0, R=1.)
    assert np.isclose(d, np.pi, atol=1.e-14)
    d = distance_on_unit_sphere(180, 0, 90, 0, R=1.)
    assert np.isclose(d, np.pi/2, atol=1.e-14)
    d = distance_on_unit_sphere(180, 45, 180, 0, R=1.)
    assert np.isclose(d, np.pi/4, atol=1.e-14)


def test_find_closest_grid_point():
    from sectionate.section import find_closest_grid_point

    # check it works with numpy arrays
    i, j = find_closest_grid_point(0, 0, lon, lat)
    assert np.equal(i, 0)
    assert np.equal(j, 80)

    # and xarray
    i, j = find_closest_grid_point(0, 0, ds["lon"], ds["lat"])
    assert np.equal(i, 0)
    assert np.equal(j, 80)

    i, j = find_closest_grid_point(180, 80, ds["lon"], ds["lat"])
    assert np.equal(i, 180)
    assert np.equal(j, 160)


def test_grid_path():
    from sectionate.section import infer_grid_path
    maps = _latlon_neighbor_maps()

    # test zonal line
    isec, jsec, lonsec, latsec = infer_grid_path(0, 80, 179, 80, lon, lat, maps)
    assert len(isec) == 180
    assert lonsec[0] == 0.0
    assert lonsec[-1] == 179.0
    assert latsec[0] == 0.0
    assert latsec[-1] == 0.0

    # test merid line
    isec, jsec, lonsec, latsec = infer_grid_path(0, 0, 0, 160, lon, lat, maps)
    assert len(isec) == 161
    assert lonsec[0] == 0.
    assert lonsec[-1] == 0.
    assert latsec[0] == -80.0
    assert latsec[-1] == 80.0

    # test diagonal
    isec, jsec, lonsec, latsec = infer_grid_path(0, 0, 100, 100, lon, lat, maps)
    assert len(isec) == 201  # expect ni+nj+1 values
    isec, jsec, lonsec, latsec = infer_grid_path(0, 0, 50, 100, lon, lat, maps)
    assert len(isec) == 151  # expect ni+nj+1 values
    isec, jsec, lonsec, latsec = infer_grid_path(10, 10, 100, 50, lon, lat, maps)
    assert len(isec) == 131  # expect ni+nj+1 values


def test_infer_grid_path_from_geo():
    from sectionate.section import infer_grid_path_from_geo
    maps = _latlon_neighbor_maps()

    # test zonal line
    isec, jsec, lonsec, latsec = infer_grid_path_from_geo(0, 0, 179, 0, lon, lat, maps)
    assert len(isec) == 180
    assert lonsec[0] == 0.0
    assert lonsec[-1] == 179.0
    assert latsec[0] == 0.0
    assert latsec[-1] == 0.0

    # test merid line
    isec, jsec, lonsec, latsec = infer_grid_path_from_geo(180, -80, 180, 80, lon, lat, maps)
    assert len(isec) == 161
    assert lonsec[0]  == 180.0
    assert lonsec[-1] == 180.0
    assert latsec[0]  == -80.0
    assert latsec[-1] == 80.0


def test_infer_grid_path_requires_neighbor_maps():
    """The low-level pathfinder needs grid-derived connectivity; there is no fallback."""
    from sectionate.section import infer_grid_path, infer_grid_path_from_geo
    with pytest.raises(ValueError):
        infer_grid_path(0, 0, 1, 1, lon, lat, None)
    with pytest.raises(ValueError):
        infer_grid_path_from_geo(0, 0, 1, 1, lon, lat, None)


def _symmetric_periodic_grid(N=8):
    """A symmetric ('outer') X-periodic grid: the seam meridian is stored twice, once as
    corner column 0 (lon 0) and once as corner column N (lon 360)."""
    xq = np.linspace(0., 360., N + 1); xh = 0.5 * (xq[:-1] + xq[1:])
    yq = np.array([-30., 0., 30.]); yh = np.array([-15., 15.])
    lon_c, lat_c = np.meshgrid(xq, yq); lon2, lat2 = np.meshgrid(xh, yh)
    return xgcm.Grid(
        xr.Dataset(coords={
            "xq": np.arange(N + 1), "yq": np.arange(3), "xh": np.arange(N), "yh": np.arange(2),
            "geolon_c": (("yq", "xq"), lon_c), "geolat_c": (("yq", "xq"), lat_c),
            "geolon": (("yh", "xh"), lon2), "geolat": (("yh", "xh"), lat2),
        }),
        coords={"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}},
        padding={"X": "periodic", "Y": "extend"}, autoparse_metadata=False,
    )


def test_zero_length_seam_corner_dropped_by_section_finding():
    """On a symmetric periodic grid the seam vertex (360 == 0) carries two indices, and the
    walk steps through both. That pair spans no grid cell, so it is not a velocity face:
    `grid_section` drops the redundant index, so the section it returns never contains a
    consecutive pair of corners at the same physical point, and every consecutive pair
    becomes a face. The faces are the same ones the walk implied, in both directions."""
    from sectionate.section import grid_section, distance_on_unit_sphere
    from sectionate.transports import uvindices_from_qindices
    g = _symmetric_periodic_grid()
    # short way from 315 deg to 45 deg runs east across the periodic seam; the reverse
    # section runs west across it.
    for lons, expected_i, expected_faces in [
        ([315., 45.], [7, 0, 1], [7, 0]),
        ([45., 315.], [1, 8, 7], [0, 7]),
    ]:
        i_c, j_c, lons_c, lats_c = grid_section(g, lons, [0., 0.])

        # The seam vertex survives exactly once: no consecutive pair is the same point.
        edge_len = distance_on_unit_sphere(lons_c[:-1], lats_c[:-1], lons_c[1:], lats_c[1:])
        assert np.all(edge_len > 1.e-3)
        assert i_c.tolist() == expected_i

        uv = uvindices_from_qindices(g, i_c, j_c)
        # every consecutive pair of corners is now a face, and they are the two real
        # faces flanking the seam vertex (index arithmetic wraps the seam crossing).
        assert uv["var"].size == i_c.size - 1
        assert np.all(uv["var"] == "V")
        assert uv["i"].tolist() == expected_faces

        # normalising an already-normalised path is a no-op
        again = drop_repeated_corners(g, i_c, j_c)
        assert again[0].tolist() == i_c.tolist()
        assert again[1].tolist() == j_c.tolist()


def test_repeated_corner_raises_rather_than_inventing_a_face():
    """`uvindices_from_qindices` requires the section-finding invariant instead of
    re-establishing it, so it must CHECK rather than assume: a hand-built path that steps
    through both of the seam vertex's indices has to raise. Silently deriving a face from
    that pair would invent a duplicate of a neighbouring face and double-count its flux."""
    from sectionate.section import grid_section, drop_repeated_corners
    from sectionate.transports import uvindices_from_qindices

    g = _symmetric_periodic_grid()
    i_c, j_c, _, _ = grid_section(g, [315., 45.], [0., 0.])
    assert i_c.tolist() == [7, 0, 1]

    # the raw walk, stepping through both indices of the seam vertex (360 == 0)
    bad_i, bad_j = [7, 8, 0, 1], [1, 1, 1, 1]
    with pytest.raises(ValueError, match="same physical point"):
        uvindices_from_qindices(g, bad_i, bad_j)

    # `drop_repeated_corners` is the documented remedy, and recovers the traced section
    fixed_i, fixed_j, fixed_f, _, _ = drop_repeated_corners(g, bad_i, bad_j)
    assert fixed_f is None
    assert fixed_i.tolist() == i_c.tolist()
    assert fixed_j.tolist() == j_c.tolist()


def test_create_section_composite_normalises_when_given_a_grid():
    """`create_section_composite` is public and documented as the lower-level entry point,
    so what it returns must be usable by the rest of the public API. Given the grid it
    normalises exactly as `grid_section` does; without one the caller still gets the raw
    walk, which `uvindices_from_qindices` refuses."""
    from sectionate.section import create_section_composite, grid_section
    from sectionate.gridutils import get_geo_corners, build_neighbor_maps
    from sectionate.transports import uvindices_from_qindices

    g = _symmetric_periodic_grid()
    geocorners = get_geo_corners(g)
    maps = build_neighbor_maps(g, geocorners)
    args = (geocorners["X"], geocorners["Y"], [315., 45.], [0., 0.])

    raw_i, raw_j, _, _ = create_section_composite(*args, neighbor_maps=maps)
    assert raw_i.tolist() == [7, 8, 0, 1]            # the seam vertex under both indices
    with pytest.raises(ValueError, match="same physical point"):
        uvindices_from_qindices(g, raw_i, raw_j)

    i_c, j_c, _, _ = create_section_composite(*args, neighbor_maps=maps, grid=g)
    assert i_c.tolist() == grid_section(g, [315., 45.], [0., 0.])[0].tolist()
    assert uvindices_from_qindices(g, i_c, j_c)["var"].size == i_c.size - 1


def _idealized_grid(xpad, ypad, N=8):
    """A symmetric ('outer') grid whose two axes can each be declared periodic or walled,
    for exercising the seam-crossing index arithmetic on either topology."""
    xq = np.linspace(0., 360., N + 1); xh = 0.5 * (xq[:-1] + xq[1:])
    yq = np.linspace(-40., 40., N + 1); yh = 0.5 * (yq[:-1] + yq[1:])
    lon_c, lat_c = np.meshgrid(xq, yq); lon2, lat2 = np.meshgrid(xh, yh)
    return xgcm.Grid(
        xr.Dataset(coords={
            "xq": np.arange(N + 1), "yq": np.arange(N + 1),
            "xh": np.arange(N), "yh": np.arange(N),
            "geolon_c": (("yq", "xq"), lon_c), "geolat_c": (("yq", "xq"), lat_c),
            "geolon": (("yh", "xh"), lon2), "geolat": (("yh", "xh"), lat2),
        }),
        coords={"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}},
        padding={"X": xpad, "Y": ypad}, autoparse_metadata=False,
    )


def test_wrap_rule_applies_only_to_periodic_axes():
    """A step of more than one index along an axis means the section crossed that axis'
    seam and went the short way round -- so its direction is the opposite of what the index
    difference says, and the face is the one on the near side of the seam. That reading is
    only valid on an axis that WRAPS. On a walled axis the same step is a jump between two
    indices of one physical corner (the degenerate column of a bipolar cap), and reading it
    as a wrap puts the face on the wrong side and its cell index out of range. Each rule is
    therefore gated on its own axis' periodicity."""
    from sectionate.transports import uvindices_from_qindices

    periodic = _idealized_grid("periodic", "periodic")
    walled = _idealized_grid("extend", "extend")

    # A zonal step of -7: eastward across the "X" seam if it wraps, else a jump west.
    uv = uvindices_from_qindices(periodic, [7, 0], [1, 1])
    assert (uv["var"][0], uv["i"][0], bool(uv["Xinc"][0])) == ("V", 7, True)
    uv = uvindices_from_qindices(walled, [7, 0], [1, 1])
    assert (uv["var"][0], uv["i"][0], bool(uv["Xinc"][0])) == ("V", 0, False)

    # ...and the same, one axis over: a meridional step of -7 across the "Y" seam.
    uv = uvindices_from_qindices(periodic, [1, 1], [7, 0])
    assert (uv["var"][0], uv["j"][0], bool(uv["Yinc"][0])) == ("U", 7, True)
    uv = uvindices_from_qindices(walled, [1, 1], [7, 0])
    assert (uv["var"][0], uv["j"][0], bool(uv["Yinc"][0])) == ("U", 0, False)


def _pole_column_grid(N=6, M=4, jp=2):
    """A miniature bipolar cap: an X-periodic symmetric grid whose corner column `i=0`
    (and its seam twin `i=N`) collapses to a single physical point from row `jp` up, the
    way a bipolar cap's grid column converges on its pole. Rows `jp..M` of that column are
    one corner stored 2*(M-jp+1) times."""
    xq = np.linspace(0., 360., N + 1); xh = 0.5 * (xq[:-1] + xq[1:])
    yq = np.linspace(-30., 30., M + 1); yh = 0.5 * (yq[:-1] + yq[1:])
    lon_c, lat_c = np.meshgrid(xq, yq); lon2, lat2 = np.meshgrid(xh, yh)
    for i in (0, N):
        lon_c[jp:, i] = lon_c[jp, 0]
        lat_c[jp:, i] = lat_c[jp, 0]
    return xgcm.Grid(
        xr.Dataset(coords={
            "xq": np.arange(N + 1), "yq": np.arange(M + 1),
            "xh": np.arange(N), "yh": np.arange(M),
            "geolon_c": (("yq", "xq"), lon_c), "geolat_c": (("yq", "xq"), lat_c),
            "geolon": (("yh", "xh"), lon2), "geolat": (("yh", "xh"), lat2),
        }),
        coords={"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}},
        padding={"X": "periodic", "Y": "extend"}, autoparse_metadata=False,
    )


def test_degenerate_column_collapses_to_the_index_that_names_both_faces():
    """A path that walks up a degenerate column and turns off it enters and leaves the pole
    on the SAME index, so one index stands in for the whole run: the collapsed path must
    yield exactly the two faces the raw path did. Which index survives is not free -- it is
    the one that reproduces them."""
    from sectionate.section import drop_repeated_corners
    from sectionate.transports import uvindices_from_qindices

    g = _pole_column_grid()
    raw_i, raw_j = [6, 6, 6, 6, 5], [1, 2, 3, 4, 4]
    raw_faces = [("U", 6, 1), ("V", 5, 4)]      # the only two real faces of that path

    i_c, j_c, f_c, lons_c, lats_c = drop_repeated_corners(g, raw_i, raw_j)
    assert f_c is None
    assert i_c.tolist() == [6, 6, 5] and j_c.tolist() == [1, 4, 4]

    uv = uvindices_from_qindices(g, i_c, j_c)
    assert uv["var"].size == i_c.size - 1
    assert list(zip(uv["var"], uv["i"].tolist(), uv["j"].tolist())) == raw_faces


def test_degenerate_column_entered_and_left_on_different_indices_raises():
    """The pole of a bipolar cap is one physical corner stored under a whole column of
    indices. A section can reach it along one of them and leave along another, far apart in
    the index lattice; then no single index is adjacent to both neighbours, so no path with
    distinct consecutive corners carries both flanking faces. That must raise and name the
    corner, not silently emit a face the section does not have (collapsing to the last
    index here would turn the leading V face into a U face)."""
    from sectionate.section import drop_repeated_corners

    g = _pole_column_grid()
    # In along row 4 (a V face), out along row 2 (another V face), via the pole.
    raw_i, raw_j = [1, 0, 0, 0, 6, 5], [4, 4, 3, 2, 2, 2]
    with pytest.raises(ValueError, match="degenerate"):
        drop_repeated_corners(g, raw_i, raw_j)
    with pytest.raises(ValueError, match=r"enters it from \(1, 4\)"):
        drop_repeated_corners(g, raw_i, raw_j)


def test_gridded_section_roundtrips_through_save_load(tmp_path):
    """A section written by `save_gridded_section` already satisfies the invariant, so
    loading it neither needs to nor does change it, and it yields identical faces."""
    import json
    from sectionate.section import Section, GriddedSection
    from sectionate.utils import save_gridded_section, load_gridded_section
    from sectionate.transports import uvindices_from_qindices

    g = _symmetric_periodic_grid()
    gs = GriddedSection(Section("seam", ([315., 45.], [0., 0.])), g)

    path = str(tmp_path / "gs.json")
    save_gridded_section(path, gs)
    with open(path) as fh:
        assert json.load(fh)["i_c"] == [7, 0, 1]      # the file holds the normalised path

    gs2 = load_gridded_section(path, g)
    assert np.asarray(gs2.i_c).tolist() == np.asarray(gs.i_c).tolist()
    assert np.asarray(gs2.j_c).tolist() == np.asarray(gs.j_c).tolist()
    assert np.allclose(gs2.lons_c, gs.lons_c)

    uv1 = uvindices_from_qindices(g, gs.i_c, gs.j_c)
    uv2 = uvindices_from_qindices(g, gs2.i_c, gs2.j_c)
    for key in uv1:
        assert np.array_equal(uv1[key], uv2[key])


def test_coincident_twin_termination():
    """When the walker reaches a corner that is a different index but the same physical
    point as the target (a fold-seam twin), it terminates there via COINCIDENT_TOLERANCE_M
    rather than oscillating to the step-count cap. Pre-fix this looped to a RuntimeError."""
    from sectionate.section import (
        infer_grid_path, COINCIDENT_TOLERANCE_M, distance_on_unit_sphere,
    )
    # col 3 is a near-exact twin of col 0 (the target): a sub-tolerance offset that is
    # nonetheless above the old exact-coincidence threshold.
    glon = np.array([[0.0, 2.0, 1.0, 1.e-12]])
    glat = np.zeros((1, 4))
    d_twin = distance_on_unit_sphere(glon[0, 0], glat[0, 0], glon[0, 3], glat[0, 3])
    assert 1.e-12 < d_twin < COINCIDENT_TOLERANCE_M

    def m(imap):
        return (None, np.zeros((1, 4), dtype=int), np.array([imap], dtype=int))
    # a line  col1 -- col2 -- col3 (~col0);  col0 hangs only off col3
    maps = {
        "right": m([3, 1, 1, 2]),
        "left":  m([0, 2, 3, 0]),
        "up":    m([0, 1, 2, 3]),
        "down":  m([0, 1, 2, 3]),
    }
    i_c, j_c, lons_c, lats_c = infer_grid_path(1, 0, 0, 0, glon, glat, maps)
    # stopped on the twin (index 3), never needing to reach the target's index (0)
    assert i_c[-1] == 3
    assert distance_on_unit_sphere(lons_c[-1], lats_c[-1], glon[0, 0], glat[0, 0]) < COINCIDENT_TOLERANCE_M
