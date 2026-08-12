import numpy as np
import xarray as xr
import xgcm
import pytest

from sectionate.gridutils import get_geo_corners
from sectionate.topology import corner_topology


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


def test_vertical_axis_does_not_affect_horizontal_topology():
    """Horizontal connectivity must not depend on whether the grid also registers a
    vertical axis. Resolving the topology pads tracer cells, and xgcm's `pad` raises
    on an axis the array has no dimension for, so padding over *every* axis of the
    grid made an otherwise ordinary grid untraceable."""
    plain = corner_topology(_latlon_grid())
    with_z = corner_topology(_latlon_grid(vertical_axis=True))
    np.testing.assert_array_equal(plain.node_id, with_z.node_id)
    np.testing.assert_array_equal(plain.node_native, with_z.node_native)



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







def test_axis_aligned_sections_take_one_step_per_cell():
    """
    On a regular one-degree grid an axis-aligned section has a step count nothing has
    to be looked up: a meridional leg from 80S to 80N crosses 160 rows and so visits
    161 corners, and a zonal leg of 179 degrees visits 180.
    """
    from sectionate.section import grid_section

    grid = _latlon_grid()
    i, j, lons, lats = grid_section(grid, [0., 179.], [0., 0.])
    assert len(i) == 180
    assert np.array_equal(lons, np.arange(0., 180.))
    assert np.all(lats == 0.)

    i, j, lons, lats = grid_section(grid, [0., 0.], [-80., 80.])
    assert len(j) == 161
    assert np.array_equal(lats, np.arange(-80., 81.))
    assert np.all(lons == 0.)


def test_crossing_the_periodic_seam_emits_one_face_per_cell_crossed():
    """
    A seam corner is one point that a grid may spell more than once, and where the
    section is written in both, the step between them spans no cell. How many
    spellings there are is a property of the staggering -- an 'outer' grid stores the
    seam column twice, a 'right' grid once -- so the contract is stated in terms of
    the repeats actually present: every step that is not a repeat is one real face,
    and `q` skips exactly the repeats.
    """
    from sectionate.section import grid_section
    from sectionate.transports import uvindices_from_qindices
    from sectionate.topology import corner_topology

    grid = _latlon_grid()
    i, j, lons, lats = grid_section(grid, [355., 5.], [0., 0.])
    ct = corner_topology(grid)
    nodes = ct.node_id[0, j + ct.t, i + ct.t]
    repeats = int((nodes[1:] == nodes[:-1]).sum())
    # the path really does cross the seam, one cell at a time
    assert lons[0] == 355. and lons[-1] == 5.
    for a, b in zip(nodes[:-1], nodes[1:]):
        assert a == b or int(b) in ct.neighbors(int(a))

    uv = uvindices_from_qindices(grid, i, j)
    assert uv["var"].size == len(i) - 1 - repeats
    assert set(uv["var"].tolist()) <= {"U", "V"}
    # `q` says which corner step each face came from, and skips exactly the hand-off
    skipped = sorted(set(range(len(i) - 1)) - set(uv["q"].tolist()))
    assert skipped == [k for k in range(len(i) - 1) if nodes[k] == nodes[k + 1]]
