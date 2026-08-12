import numpy as np
import xarray as xr
import xgcm

# define simple xgcm grid
xq = np.array([0., 60, 120, 180, 240, 300., 360.])
yq = np.array([-80., -40, 0, 40, 80.])

lon_c, lat_c = np.meshgrid(xq, yq)
ds = xr.Dataset({}, coords={
    "xq":xr.DataArray(xq, dims=("xq",)),
    "yq":xr.DataArray(yq, dims=("yq",)),
    "lon_c":xr.DataArray(lon_c, dims=("yq", "xq",)),
    "lat_c":xr.DataArray(lat_c, dims=("yq", "xq",))
})
coords = {
    'X': {'outer': 'xq'},
    'Y': {'outer': 'yq'}
}
boundary = {
    'X': 'periodic',
    'Y': 'extend'
}
grid = xgcm.Grid(ds, coords=coords, padding=boundary, autoparse_metadata=False)

def modequal(a,b):
    return np.equal(np.mod(a, 360.), np.mod(b, 360.))

def test_open_gridded_section():
    from sectionate.section import Section, GriddedSection
    lonseg = np.array([0., 120, 120, 0])
    latseg = np.array([-80., -80, 0, 0])
    sec = Section("testsec", (lonseg, latseg))
    sec_gridded = GriddedSection(sec, grid)

    assert np.all([
        modequal(sec_gridded.i_c, np.array([0, 1, 2, 2, 2, 1, 0])),
        modequal(sec_gridded.j_c, np.array([0, 0, 0, 1, 2, 2, 2])),
        modequal(sec_gridded.lons_c, np.array([0.,  60., 120., 120., 120.,  60., 0.])),
        modequal(sec_gridded.lats_c, np.array([-80., -80., -80., -40.,   0.,   0.,   0.]))
    ])

def test_gridded_section_copy():
    from sectionate.section import Section, GriddedSection
    lonseg = np.array([0., 120, 120, 0])
    latseg = np.array([-80., -80, 0, 0])
    sec_gridded = GriddedSection(Section("testsec", (lonseg, latseg)), grid)

    dup = sec_gridded.copy()
    # a real GriddedSection is returned (the old implementation returned None)
    assert isinstance(dup, GriddedSection)
    assert dup is not sec_gridded
    # same contents ...
    assert np.array_equal(dup.i_c, sec_gridded.i_c)
    assert np.array_equal(dup.j_c, sec_gridded.j_c)
    assert np.array_equal(dup.lons_c, sec_gridded.lons_c)
    assert np.array_equal(dup.lats_c, sec_gridded.lats_c)
    assert dup.name == sec_gridded.name
    # ... grid shared, index arrays are independent copies
    assert dup.grid is sec_gridded.grid
    assert dup.i_c is not sec_gridded.i_c
    dup.i_c[0] = 999
    assert sec_gridded.i_c[0] != 999


def test_closed_gridded_parent_section():
    from sectionate.section import Section, join_sections, GriddedSection
    lonseg = np.array([  0., 120, 120,  0,   0])
    latseg = np.array([-80., -80,   0,  0, -80.])
    # Test join_sections and children/parent relationships
    sec1 = Section("sec1", (lonseg[0:3], latseg[0:3]))
    sec2 = Section("sec2", (lonseg[2: ], latseg[2: ]))
    sec = join_sections("sec", sec1, sec2)
    assert isinstance(sec.children["sec1"], Section)
    # Test results from join_section
    sec_gridded = GriddedSection(sec, grid)
    assert np.all([
        modequal(sec_gridded.i_c, np.array([0, 1, 2, 2, 2, 1, 0, 0, 0])),
        modequal(sec_gridded.j_c, np.array([0, 0, 0, 1, 2, 2, 2, 1, 0])),
        modequal(sec_gridded.lons_c, np.array([0.,  60., 120., 120., 120.,  60., 0., 0., 0.])),
        modequal(sec_gridded.lats_c, np.array([-80., -80., -80., -40.,   0.,   0.,   0., -40., -80.]))
    ])

def test_node_ids_are_derived_and_survive_a_save_roundtrip(tmp_path):
    """
    A section carries which physical corner each of its points is, and that is
    always derived from the grid rather than stored: a node id is an artefact of how
    the topology was built, so a saved one could disagree with the grid it is loaded
    against. Deriving it is what makes a reloaded section, or one built by hand from
    indices, behave exactly like one this package traced.
    """
    import numpy as np
    from sectionate.section import Section, GriddedSection
    from sectionate.utils import save_gridded_section, load_gridded_section

    sec = GriddedSection(Section("box", ([0., 120.], [-80., -80.])), grid)
    assert sec.n_c.shape == sec.i_c.shape
    assert (sec.n_c >= 0).all()
    # a single-tile grid is one face, so the face index is present and zero
    assert np.array_equal(sec.f_c, np.zeros_like(sec.i_c))

    path = tmp_path / "box.json"
    save_gridded_section(str(path), sec)
    back = load_gridded_section(str(path), grid)
    for name in ("i_c", "j_c", "f_c", "n_c"):
        np.testing.assert_array_equal(getattr(sec, name), getattr(back, name))

    # built by hand from the same indices: same nodes, no repair pass needed
    hand = GriddedSection(Section("hand", (sec.lons_c, sec.lats_c)), grid,
                          i_c=sec.i_c, j_c=sec.j_c, f_c=sec.f_c)
    np.testing.assert_array_equal(hand.n_c, sec.n_c)
