import warnings
import numpy as np
import xarray as xr
import dask

from .gridutils import (
    corner_offset, coord_dict, get_geo_corners, get_facedim,
)
from .section import distance_on_unit_sphere










def uvindices_from_qindices(grid, i_c, j_c, f_c=None):
    """
    Find the `grid` indices of the N-1 velocity points defined by the consecutive indices of
    N vorticity points, automatically reading the vorticity corner position from `grid`
    metadata ('outer', 'right', or 'left'; see `gridutils.corner_offset`).

    PARAMETERS:
    -----------
    grid: xgcm.Grid
        Grid object describing ocean model grid and containing data variables
    i_c: int
        vorticity point indices along "X" dimension
    j_c: int
        vorticity point indices along "Y" dimension
    f_c: array of int or None
        Face indices of the vorticity points for multi-tile grids (`face_connections`);
        None for single-tile grids.

    RETURNS:
    --------
    uvindices : dict
        Dictionary containing:
          - "var" : "U" if corresponding to "X"-direction velocity (usually nominally zonal), "V" otherwise
          - "i" : "X"-dimension index of appropriate "U" or "V" velocity
          - "j" : "Y"-dimension index of appropriate "U" or "V" velocity
          - "Yinc" : True if point was passed through while going in positive "j"-index direction
          - "Xinc" : True if point was passed through while going in positive "i"-index direction
        For multi-tile grids (`f_c` given) the dict additionally contains:
          - "face" : face index of the velocity point
          - "Lsign" : +1 if the velocity's positive direction points left of the section's
            direction of travel, else -1 (a geometric sign that stays consistent across
            rotated face seams; used in place of "Xinc"/"Yinc").
    """
    from .topology import corner_topology

    i_c = np.asarray(i_c, dtype=np.int64)
    j_c = np.asarray(j_c, dtype=np.int64)
    topology = corner_topology(grid)
    t = topology.t
    faces = (np.zeros_like(i_c) if f_c is None else np.asarray(f_c, dtype=np.int64))

    # Which physical corner each given index denotes. Everything below is decided
    # from these, so it does not matter which of a corner's spellings the caller
    # used, nor whether a seam crossing was written with both of them or one.
    if (
        (faces < 0).any() or (faces >= topology.nf).any()
        or (j_c + t < 0).any() or (j_c + t >= topology.nqy).any()
        or (i_c + t < 0).any() or (i_c + t >= topology.nqx).any()
    ):
        raise ValueError("Section contains indices that are not grid corners.")
    nodes = topology.node_id[faces, j_c + t, i_c + t]

    handed = topology.face_handedness
    var, vi, vj, vface, lsign, qq, xinc, yinc = [], [], [], [], [], [], [], []

    for k in range(nodes.size - 1):
        a, b = int(nodes[k]), int(nodes[k + 1])
        if a == b:
            # Two spellings of one corner. The step spans no cell, so it is not a
            # velocity face -- this is how a seam crossing is written, and dropping
            # it here is what stops the crossing being counted twice.
            continue
        stored = topology.edge_velocities(a, b)
        if not stored:
            raise ValueError(
                f"The section steps from corner (face={faces[k]}, j={j_c[k]}, "
                f"i={i_c[k]}) to (face={faces[k+1]}, j={j_c[k+1]}, i={i_c[k+1]}), "
                "which is not one step across a velocity face. Either the two "
                "corners are not neighbours on this grid, or the face between them "
                "is stored on none of its arrays. A section must step from each "
                "corner to an adjacent one."
            )
        # Read the face in the spelling the section was actually written in. Where a
        # fold stores one edge twice, the two copies sit in mirrored frames and carry
        # opposite signs, so a section that runs out and back along such a row only
        # cancels if each traversal is read in its own frame. Falling back to the
        # section's face, and then to any storage, covers a crossing written in the
        # frame it is leaving.
        want = (
            (int(faces[k]), int(j_c[k]) + t, int(i_c[k]) + t),
            (int(faces[k + 1]), int(j_c[k + 1]) + t, int(i_c[k + 1]) + t),
        )
        pick = next(
            (s for s in stored if s[7] == want),
            next((s for s in stored if s[1] == faces[k]), stored[0]),
        )
        v, f, jv, iv, _to_cell, _from_cell, step, _slots = pick

        if v == "U":
            # `+x` velocity: it points left of travel when travel runs in `-y`.
            s = -1 if step[0] > 0 else 1
        else:
            # `+y` velocity: it points left of travel when travel runs in `+x`.
            s = 1 if step[1] > 0 else -1

        var.append(v)
        vi.append(int(iv))
        vj.append(int(jv))
        vface.append(int(f))
        lsign.append(int(s) * int(handed[f]))
        xinc.append(bool(step[1] > 0))
        yinc.append(bool(step[0] > 0))
        qq.append(k)

    uvindices = {
        "var": np.array(var, dtype="<U2"),
        "i": np.array(vi, dtype=np.int64),
        "j": np.array(vj, dtype=np.int64),
        "Yinc": np.array(yinc, dtype=bool),
        "Xinc": np.array(xinc, dtype=bool),
        "Lsign": np.array(lsign, dtype=np.int64),
        "q": np.array(qq, dtype=np.int64),
    }
    if f_c is not None:
        uvindices["face"] = np.array(vface, dtype=np.int64)
    return uvindices




def uvcoords_from_uvindices(grid, uvindices):
    """
    Find the (lons,lats) coordinates of the N-1 velocity points defined by `uvindices` (returned by `uvindices_from_qindices`).
    Assumes the names of longitude and latitude coordinates in `grid` contain the sub-strings "lon" and "lat", respectively,
    but otherwise finds the names using `grid` metadata.

    PARAMETERS:
    -----------
    grid: xgcm.Grid
        Grid object describing ocean model grid and containing data variables
    uvindices : dict
        Dictionary returned by `sectionate.transports.uvindices_from_qindices`, containing:
          - "var" : "U" if corresponding to "X"-direction velocity (usually nominally zonal), "V" otherwise
          - "i" : "X"-dimension index of appropriate "U" or "V" velocity
          - "j" : "Y"-dimension index of appropriate "U" or "V" velocity
          - "Yinc" : True if point was passed through while going in positive "j"-index direction
          - "Xinc" : True if point was passed through while going in positive "i"-index direction

    RETURNS:
    --------
    lons : np.ndarray(float)
    lats : np.ndarray(float)
    """
    lons, lats = np.zeros(len(uvindices["var"])), np.zeros(len(uvindices["var"]))

    ds = grid._ds
    coords = coord_dict(grid)
    geo_coords = [c for c in list(ds.coords) if ("lon" in c) or ("lat" in c)]
    center_names = {f"geo{d}_center":c for d,c in
              {d:c for d in ["lon", "lat"] for c in geo_coords
               if ((coords["X"]["center"] in ds[c].coords) and
                   (coords["Y"]["center"] in ds[c].coords))
               if d in c}.items()}
    u_names = {f"geo{d}_u":c for d,c in
              {d:c for d in ["lon", "lat"] for c in geo_coords
               if ((coords["X"]["corner"] in ds[c].coords) and 
                   (coords["Y"]["center"] in ds[c].coords))
               if d in c}.items()}
    v_names = {f"geo{d}_v":c for d,c in
              {d:c for d in ["lon", "lat"] for c in geo_coords
               if ((coords["X"]["center"] in ds[c].coords) and
                   (coords["Y"]["corner"] in ds[c].coords))
               if d in c}.items()}
    corner_names = {f"geo{d}_corner":c for d,c in
              {d:c for d in ["lon", "lat"] for c in geo_coords
               if ((coords["X"]["corner"] in ds[c].coords) and
                   (coords["Y"]["corner"] in ds[c].coords))
               if d in c}.items()}

    facedim = get_facedim(grid)
    faces = uvindices.get("face")

    for p in range(len(uvindices["var"])):
        var, i, j = uvindices["var"][p], uvindices["i"][p], uvindices["j"][p]
        if var not in ("U", "V"):
            # Degenerate edge (e.g. a seam crossing through a shared corner): no point.
            lons[p], lats[p] = np.nan, np.nan
            continue
        # On multi-tile grids, also select the velocity point's face.
        fsel = {facedim: int(faces[p])} if (facedim is not None and faces is not None) else {}
        if var == "U":
            if (f"geolon_u" in u_names) and (f"geolat_u" in u_names):
                lon = ds[u_names[f"geolon_u"]].isel({
                    coords["X"]["corner"]:i,
                    coords["Y"]["center"]:j,
                    **fsel
                }).values
                lat = ds[u_names[f"geolat_u"]].isel({
                    coords["X"]["corner"]:i,
                    coords["Y"]["center"]:j,
                    **fsel
                }).values
            elif (f"geolon_corner" in corner_names) and (f"geolat_center" in center_names):
                lon = ds[corner_names[f"geolon_corner"]].isel({
                    coords["X"]["corner"]:i,
                    coords["Y"]["corner"]:j,
                    **fsel
                }).values
                lat = ds[center_names[f"geolat_center"]].isel({
                    coords["X"]["center"]:wrap_idx(i, grid, "X"),
                    coords["Y"]["center"]:wrap_idx(j, grid, "Y"),
                    **fsel
                }).values
            else:
                raise ValueError("Cannot locate grid coordinates necessary to\
                identify U-velociy faces.")
        elif var == "V":
            if (f"geolon_v" in v_names) and (f"geolat_v" in v_names):
                lon = ds[v_names[f"geolon_v"]].isel({
                    coords["X"]["center"]:wrap_idx(i, grid, "X"),
                    coords["Y"]["corner"]:j,
                    **fsel
                }).values
                lat = ds[v_names[f"geolat_v"]].isel({
                    coords["X"]["center"]:wrap_idx(i, grid, "X"),
                    coords["Y"]["corner"]:j,
                    **fsel
                }).values
            elif (f"geolon_center" in center_names) and (f"geolat_corner" in corner_names):
                lon = ds[center_names[f"geolon_center"]].isel({
                    coords["X"]["center"]:wrap_idx(i, grid, "X"),
                    coords["Y"]["center"]:wrap_idx(j, grid, "Y"),
                    **fsel
                }).values
                lat = ds[corner_names[f"geolat_corner"]].isel({
                    coords["X"]["corner"]:i,
                    coords["Y"]["corner"]:j,
                    **fsel
                }).values
            else:
                raise ValueError("Cannot locate grid coordinates necessary to\
                identify V-velociy faces.")
        lons[p] = lon
        lats[p] = lat
    return lons, lats
    
def uvcoords_from_qindices(grid, i_c, j_c, f_c=None):
    """
    Directly finds coordinates of velocity points from vorticity point indices, wrapping other functions.

    PARAMETERS:
    -----------
    grid: xgcm.Grid
        Grid object describing ocean model grid and containing data variables
    i_c: int
        vorticity point indices along "X" dimension
    j_c: int
        vorticity point indices along "Y" dimension
    f_c: int or None
        Face indices of the vorticity points for multi-tile grids; None for single-tile grids.

    RETURNS:
    --------
    lons : np.ndarray(float)
    lats : np.ndarray(float)
    """
    return uvcoords_from_uvindices(
        grid,
        uvindices_from_qindices(grid, i_c, j_c, f_c=f_c),
    )

# xgcm positions at which a vertical *interface* (cell-edge) coordinate can sit,
# as opposed to the "center" position where the paired layer coordinate sits.
_INTERFACE_POSITIONS = ("outer", "inner", "left", "right")


def _interface_coords(grid, da):
    """Interface (cell-edge) coordinates matching the vertical dimensions `da` carries.

    A transport's layer (cell-center) coordinate needs no help: it is one of the
    transport's own dimensions, so it rides along from `utr`/`vtr` into the section.
    Its interface coordinate cannot -- it is one point longer, and so is a dimension of
    nothing on the section. The grid knows it, though: an axis that registers one of
    `da`'s dimensions at its "center" position names the matching interface coordinate
    at one of its interface positions.

    Keyed off the data's own dimensions rather than off an axis called "Z", so a
    vertical axis under any name is found, and a grid that declares only its horizontal
    axes (as most grids handed to sectionate do, since sections are only ever traced
    horizontally) simply contributes nothing.

    Only names present in `grid._ds` are returned: xgcm requires an axis' positions to
    name *dimensions*, but a dimension need not carry coordinate values.
    """
    coords = {}
    for axis in grid.axes.values():
        positions = axis.coords
        if positions.get("center") not in da.dims:
            continue
        name = next(
            (positions[pos] for pos in _INTERFACE_POSITIONS
             if positions.get(pos) in grid._ds),
            None
        )
        if name is not None:
            coords[name] = grid._ds[name]
    return coords


def convergent_transport(
    grid,
    i_c,
    j_c,
    f_c=None,
    utr="umo",
    vtr="vmo",
    outname="conv_mass_transport",
    sect_coord="sect",
    geometry="spherical",
    positive_in=True,
    cell_widths={"U":"dyCu", "V":"dxCv"},
    ):
    """
    Lazily calculates extensive transports normal to a section, with the sign convention of positive into the spherical polygon
    defined by the section, unless overridden by changing the "positive_in=True" keyword argument. Supports curvlinear geometries
    and complicated grid topologies, as long as the grid is *locally* orthogonal.
    Lazily broadcasts the calculation in all dimensions except ("X", "Y").

    PARAMETERS:
    -----------
    grid: xgcm.Grid
        Grid object describing ocean model grid and containing data variables. Must include variables "utr" and "vtr" (see kwargs).
    i_c: int
        Vorticity point indices along "X" dimension.
    j_c: int
        Vorticity point indices along "Y" dimension.
    f_c: array-like or None
        Vorticity point face (tile) indices, required on multi-tile grids (those with
        `face_connections`) and returned by `grid_section` for them; leave as None for
        single-tile grids.
    utr: str
        Name of "X"-direction tracer transport
    vtr: str
        Name of "Y"-direction tracer transport
    outname : str
        Name of output xr.DataArray variable. Default: "conv_mass_transport".
    sect_coord: str
        Name of the dimension describing along-section data in the output. Default: "sect".
    geometry : str
        Geometry to use to check orientation of the section. Supported geometries are ["cartesian", "spherical"].
        Default: "spherical".
    positive_in : bool or xr.DataArray of type bool
        If True, convergence is defined as "inwards" with respect to the corresponding "geometry".
        If False, convergence is defined as "outwards" (equivalently, negative the inward convergence).
        If a boolean xr.DataArray, get value of positive_in by selecting the value of the mask on the inside
        of an arbitrary velocity face.
        For *open* sections (whose first and last waypoints are far apart) there is no enclosing
        polygon, so "inwards"/"outwards" is undefined. In that case the sign instead follows the
        left-of-transect convention: `positive_in=True` makes positive transport point to the LEFT
        of the section as it is traversed from the first to the last waypoint, and `positive_in=False`
        flips it to the right. A UserWarning is raised so the orientation can be verified.
    cell_widths : dict
        Values of "U" and "V" items in `cell_widths` correspond to the names of the coordinates describing the width of
        velocity cells. If they are both present in `grid._ds`, accumulate distance along the section and add it to the
        returned xr.DataArray.

    RETURNS:
    --------
    dsout : xr.DataArray
        Contains the calculated normal transport and the coordinates of the contributing velocity points (`lon`, `lat`),
        as well as some useful metadata, such as whether each point corresponds to a "U" or "V" velocity and whether
        the sign of the transport had to be flipped to make it point inwards.

        Vertical coordinates are not named anywhere in the call: the layer (cell-center)
        coordinate is inherited from `utr`/`vtr`, being one of their dimensions, and the
        matching interface (cell-edge) coordinate is read off whichever axis of `grid`
        registers that layer dimension at its "center" position. Declare the vertical
        axis on the `xgcm.Grid` (e.g. `coords={..., "Z": {"center": "z_l", "outer":
        "z_i"}}`) if the interface coordinate is wanted downstream.
    """

    # On a multi-tile grid the contributing velocity face varies along the section
    # (`uvindices["face"]`); it is selected pointwise in every `.isel` below.
    facedim = get_facedim(grid) if f_c is not None else None

    uvindices = uvindices_from_qindices(grid, i_c, j_c, f_c=f_c)
    uvcoords = uvcoords_from_qindices(grid, i_c, j_c, f_c=f_c)

    sect = xr.Dataset()
    sect = sect.assign_coords({
        sect_coord: xr.DataArray(
            np.arange(uvindices["i"].size),
            dims=(sect_coord,)
        )
    })
    sect["i"] = xr.DataArray(uvindices["i"], dims=sect_coord)
    sect["j"] = xr.DataArray(uvindices["j"], dims=sect_coord)
    if facedim is not None:
        sect["face"] = xr.DataArray(uvindices["face"], dims=sect_coord)
    fsel = {facedim: sect["face"]} if facedim is not None else {}
    sect["var"] = xr.DataArray(uvindices["var"], dims=sect_coord)
    sect["Umask"] = xr.DataArray(uvindices["var"]=="U", dims=sect_coord)
    sect["Vmask"] = xr.DataArray(uvindices["var"]=="V", dims=sect_coord)

    # Per-edge sign: +1 if the velocity's positive direction points left of the
    # section's direction of travel. Both are read in the frame the velocity is
    # stored in, so a rotated or reversed seam needs no special case, and an edge
    # whose two corners happen to coincide still gets a definite answer. This is the
    # only sign there is now: the face-frame `Usign`/`Vsign` said the same thing on a
    # single-tile grid and the wrong thing on any other, so they are gone.
    sect["Lsign"] = xr.DataArray(uvindices["Lsign"], dims=sect_coord)

    mask_types = (np.ndarray, dask.array.Array, xr.DataArray)
    if isinstance(positive_in, mask_types):
        positive_in = is_mask_inside(positive_in, grid, sect)
        
    else:
        if (geometry == "cartesian") and (grid.axes["X"].padding == "periodic"):
            raise ValueError("Periodic cartesian domains are not yet supported!")
        coords = coord_dict(grid)
        geo_corners = get_geo_corners(grid)
        # corner-grid selection is over all N corners ("pt"); the face varies per corner.
        corner_fsel = (
            {facedim: xr.DataArray(np.asarray(f_c), dims=("pt",))}
            if facedim is not None else {}
        )
        idx = {
            coords["X"]["corner"]:xr.DataArray(i_c, dims=("pt",)),
            coords["Y"]["corner"]:xr.DataArray(j_c, dims=("pt",)),
            **corner_fsel
        }
        lons_q = geo_corners["X"].isel(idx).values
        lats_q = geo_corners["Y"].isel(idx).values
        section_is_open = (
            distance_on_unit_sphere(lons_q[0], lats_q[0], lons_q[-1], lats_q[-1]) > 10.
        )
        if section_is_open:
            # An open section encloses no polygon, so the inward/outward meaning of
            # `positive_in` is undefined. Fall back to the left-of-transect convention:
            # `positive_in=True` keeps the per-edge `Lsign` (positive transport to the LEFT
            # of travel, from the first waypoint to the last) and `positive_in=False` flips
            # it to the right. No polygon-orientation correction is applied.
            warnings.warn(
                "This section is open (its first and last waypoints are far apart), so it does "
                "not enclose a polygon and the inward/outward sense of `positive_in` is undefined. "
                "Sectionate applies the left-of-transect convention instead: with `positive_in=True` "
                "positive transport points to the LEFT of the section as it is traversed from the "
                "first to the last waypoint (set `positive_in=False` to flip to right-of-transect). "
                "Verify the sign matches your expectations, e.g. by plotting the section's normal vectors."
            )
        else:
            # Closed section encloses a polygon: orient so `positive_in` means "into" it.
            counterclockwise = is_section_counterclockwise(
                lons_q, lats_q, geometry=geometry
            )
            positive_in = positive_in ^ (not(counterclockwise))
    orient_fact = 1 if positive_in else -1

    coords = coord_dict(grid)
    usel = {
        coords["X"]["corner"]: sect["i"],
        coords["Y"]["center"]: wrap_idx(sect["j"], grid, "Y"),
        **fsel
    }
    vsel = {
        coords["X"]["center"]: wrap_idx(sect["i"], grid, "X"),
        coords["Y"]["corner"]: sect["j"],
        **fsel
    }
    
    u = grid._ds[utr]
    v = grid._ds[vtr]
    
    conv_umo_masked = u.isel(usel).fillna(0.)*sect["Umask"]
    conv_vmo_masked = v.isel(vsel).fillna(0.)*sect["Vmask"]
    conv_transport = xr.DataArray(
        (conv_umo_masked + conv_vmo_masked)*sect["Lsign"]*orient_fact,
    )
    dsout = xr.Dataset({outname: conv_transport})
    
    if ((cell_widths["U"] in grid._ds.coords) and
        (cell_widths["V"] in grid._ds.coords)):
        
        dsout = dsout.assign_coords({
            "dl": xr.DataArray(
                (
                    (grid._ds[cell_widths["U"]].isel(usel).fillna(0.)
                     *sect["Umask"])+
                    (grid._ds[cell_widths["V"]].isel(vsel).fillna(0.)
                     *sect["Vmask"])
                ),
                dims=(sect_coord,),
                attrs={"units":"m"}
            )
        })

    dsout = dsout.assign_coords({
        "sign": orient_fact*sect["Lsign"],
        "dir": xr.DataArray(
            np.array(["U" if u else "V" for u in sect["Umask"]]),
            coords=(dsout[sect_coord],),
            dims=(sect_coord,)
        ),
        "lon": xr.DataArray(
            uvcoords[0],
            coords=(dsout[sect_coord],),
            dims=(sect_coord,)
        ),
        "lat": xr.DataArray(
            uvcoords[1],
            coords=(dsout[sect_coord],),
            dims=(sect_coord,)
        ),
    })
    dsout[outname].attrs = {**dsout[outname].attrs, **{
        "orient_fact":orient_fact,
        "positive_in":positive_in,
    }}

    # The layer (cell-center) coordinate arrived with `utr`/`vtr`; its interface
    # coordinate is a point longer and has to come from the grid (`_interface_coords`).
    dsout = dsout.assign_coords(_interface_coords(grid, dsout[outname]))

    return dsout

def is_section_counterclockwise(lons_c, lats_c, geometry="spherical"):
    """
    Check if the polygon defined by the consecutive (lons_c, lats_c) is `counterclockwise` (with respect to a given
    `geometry`). Under the hood, it does this by checking whether the signed area (or determinant) of the polygon
    is negative (counterclockwise) or positive (clockwise). This is only a meaningful calculation if the section
    is closed, i.e. (lons_c[-1], lats_c[-1]) == (lons_c[0], lats_c[0]), and therefore defines a polygon.

    If the section is *open* (its first and last waypoints are far apart), it is closed here with a
    great-circle segment from the last waypoint back to the first so that an orientation can still be
    computed, and a UserWarning is raised. Because that closure is arbitrary, the inward/outward sense is
    undefined for open sections; callers should instead rely on the left-of-transect convention (positive
    transport to the left of travel from the first to the last waypoint).

    For the case `geometry="spherical"`, the periodic nature of the longitude coordinate complicates things;
    instead of working in spherical coordinates, we use a South-Pole stereographic projection of the surface of the sphere
    and evaluate the orientation of the projected polygon with respect to the stereographic plane.

    PARAMETERS:
    -----------
    lons_c : np.ndarray(float), longitude of corner points, in degrees
    lats_c : np.ndarray(float), latitude of corner points, in degrees
    geometry : str
        Geometry to use to check orientation of the section. Supported geometries are ["cartesian", "spherical"].
        Default: "spherical".

    RETURNS:
    --------
    counterclockwise : bool
    """
    if distance_on_unit_sphere(lons_c[0], lats_c[0], lons_c[-1], lats_c[-1]) > 10.:
        warnings.warn(
            "`is_section_counterclockwise` was called on an open section; it is closed here with "
            "a great-circle segment from the last waypoint back to the first so that an orientation "
            "can still be returned, but that orientation is arbitrary for an open section. For "
            "transports, do not rely on this: `convergent_transport` instead uses the well-defined "
            "left-of-transect convention (positive transport to the LEFT of travel, from the first "
            "waypoint to the last)."
        )
        lons_c = np.append(lons_c, lons_c[0])
        lats_c = np.append(lats_c, lats_c[0])
    
    if geometry == "spherical":
        X, Y = stereographic_projection(*_turned_away_from_the_pole(lons_c, lats_c))
    elif geometry == "cartesian":
        X, Y = lons_c, lats_c
    else:
        raise ValueError("""Only "spherical" and "cartesian" geometries are currently supported.""")
    
    signed_area = 0.
    for i in range(X.size-1):
        signed_area += (X[i+1]-X[i])*(Y[i+1]+Y[i])
    return signed_area < 0.

def _turned_away_from_the_pole(lons, lats):
    """
    Rotate a closed section so that none of its corners sits near the point the
    stereographic projection is taken from.

    The projection below sends the *north* pole to infinity (its radius is
    `sin(phi)/(1 - cos(phi))`, which diverges as the colatitude goes to zero), so a
    corner anywhere near it projects to a huge radius that swamps the signed area and
    can flip the answer. A corner exactly at a geographic pole is not an exotic case
    here but an ordinary one: it is where a tripolar cap's singular meridian ends,
    and it is a vertex of every cell of an Arctic cap.

    Turning the whole configuration first costs nothing and removes the problem: a
    rotation does not change which way round a polygon runs, so the orientation this
    returns is the same one, measured somewhere the projection behaves.
    """
    lo, la = np.deg2rad(np.asarray(lons, dtype=float)), np.deg2rad(np.asarray(lats, dtype=float))
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

    # Point the corners' mean direction at the south pole, which this projection
    # sends to the origin -- as far from the singularity as they can collectively get.
    m = v.mean(axis=0)
    n = np.linalg.norm(m)
    if n < 1.e-12:            # corners spread symmetrically: any turn is as good
        return lons, lats
    m = m / n
    axis = np.cross(m, np.array([0.0, 0.0, -1.0]))
    s = np.linalg.norm(axis)
    if s < 1.e-12:            # already pointing at a pole
        if m[2] < 0.0:
            return lons, lats
        v = v * np.array([1.0, -1.0, -1.0])          # a turn, not a reflection
    else:
        axis = axis / s
        angle = np.arctan2(s, -m[2])
        c, sn = np.cos(angle), np.sin(angle)
        v = (v * c
             + np.cross(axis, v) * sn
             + axis * (v @ axis)[:, None] * (1.0 - c))     # Rodrigues

    return (np.rad2deg(np.arctan2(v[:, 1], v[:, 0])),
            np.rad2deg(np.arcsin(np.clip(v[:, 2], -1.0, 1.0))))


def stereographic_projection(lons, lats):
    """
    Projects longitudes and latitudes onto the South-Polar stereographic plane.

    PARAMETERS:
    -----------
    lons : np.ndarray(float), in degrees
    lats : np.ndarray(float), in degrees

    RETURNS:
    --------
    X : np.ndarray(float)
    Y : np.ndarray(float)
    """
    lats = np.clip(lats, -90. +1.e-3, 90. -1.e-3)
    varphi = np.deg2rad(-lats+90.)
    theta = np.deg2rad(lons)
    
    R = np.sin(varphi)/(1. - np.clip(np.cos(varphi), -1. +1.e-14, 1. -1.e-14))
    Theta = -theta
    
    X, Y = R*np.cos(Theta), R*np.sin(Theta)
    
    return X, Y

def is_mask_inside(mask, grid, sect, idx=0):
    """
    Find the `(i,j)` indices of the `grid` tracer-cell point "inside" of the velocity face at index `idx`,
    and evaluate the value of the `mask` there.

    PARAMETERS:
    -----------
    mask : xr.DataArray
    grid : xgcm.Grid
    sect : dict
        Dictionary of uvindices (see `sectionate.transports.uvindices_from_qindices`).

    RETURNS:
    --------
    positive_in : bool
    """
    offset = corner_offset(grid)
    coords = coord_dict(grid)
    facedim = get_facedim(grid)
    fsel = {facedim: int(sect["face"][idx])} if (facedim is not None and "face" in sect) else {}
    # Pick the neighbouring tracer cell on the side the stored velocity's positive direction
    # points to, using the geometric `Lsign` (+1 if that direction is left of travel) rather
    # than the face-frame `Usign`/`Vsign`. On single-tile 'outer'/'right' grids `Lsign` equals
    # `Usign`/`Vsign` for the relevant face, so this is unchanged there; on 'left' (MITgcm/ECCO)
    # and rotated multi-tile seams it carries the correct staggering/topology, where the
    # face-frame signs would select the wrong side and invert `positive_in`.
    if sect["var"][idx]=="U":
        i = (
            sect["i"][idx]
            - (1 if sect["Lsign"][idx].values==-1. else 0)
            + offset
        )
        j = sect["j"][idx]
        if 0<=i<=grid._ds[coords["X"]["center"]].size-1:
            positive_in = mask.isel({
                coords["X"]["center"]: i,
                coords["Y"]["center"]: j,
                **fsel
            }).values
        elif i==-1:
            positive_in = not(mask.isel({
                coords["X"]["center"]: i+1,
                coords["Y"]["center"]: j,
                **fsel
            })).values
        elif i==grid._ds[coords["X"]["center"]].size:
            positive_in = not(mask.isel({
                coords["X"]["center"]: i-1,
                coords["Y"]["center"]: j,
                **fsel
            })).values
    elif sect["var"][idx]=="V":
        i = sect["i"][idx]
        j = (
            sect["j"][idx]
            - (1 if sect["Lsign"][idx].values==-1. else 0)
            + offset
        )
        if 0<=j<=grid._ds[coords["Y"]["center"]].size-1:
            positive_in = mask.isel({
                coords["X"]["center"]: i,
                coords["Y"]["center"]: j,
                **fsel
            }).values
        elif j==-1:
            positive_in = not(mask.isel({
                coords["X"]["center"]: i,
                coords["Y"]["center"]: j+1,
                **fsel
            })).values
        elif j==grid._ds[coords["Y"]["center"]].size:
            positive_in = not(mask.isel({
                coords["X"]["center"]: i,
                coords["Y"]["center"]: j-1,
                **fsel
            })).values
    return positive_in


def wrap_idx(idx, grid, axis):
    coords = coord_dict(grid)
    if grid.axes[axis].padding == "periodic":
        idx = np.mod(idx, grid._ds[coords[axis]["center"]].size)
    else:
        idx = np.minimum(idx, grid._ds[coords[axis]["center"]].size-1)
    return idx