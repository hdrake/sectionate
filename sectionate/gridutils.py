import itertools
import numpy as np
import xarray as xr
from xgcm.padding import pad as _module_pad


def get_facedim(grid):
    """
    Return the name of the `grid`'s face/tile dimension if it has `face_connections`
    metadata (multi-tile grids such as the lat-lon-cap or cubed-sphere), else None.

    Parameters
    ----------
    grid: xgcm.Grid

    Returns
    -------
    str or None
    """
    return getattr(grid, "_facedim", None)


def _pad_axes(grid, dims):
    """
    Names of the `grid` axes that `dims` spans.

    `xgcm.padding.pad` iterates the whole `padding_width` mapping it is given and
    looks each axis' position up in the array being padded, so an axis the array
    has no dimension for is an error rather than a no-op. The arrays padded in
    this module are the horizontal corner/center index arrays built a few lines
    above each call, so passing `grid.axes` wholesale makes them fail on any grid
    that registers a vertical axis -- which every real model grid does. Deriving
    the list from the array's own dims keeps padding independent of whatever
    *other* axes the grid happens to carry.

    Parameters
    ----------
    grid: xgcm.Grid
    dims: iterable of str
        Dimension names of the array about to be padded.

    Returns
    -------
    list of str
    """
    dims = set(dims)
    return [
        name for name, axis in grid.axes.items()
        if dims & set(axis.coords.values())
    ]


def corner_position(grid):
    """
    Return the C-grid vorticity ("corner") position shared by the X and Y axes:
    "outer", "right", or "left".

    Parameters
    ----------
    grid: xgcm.Grid

    Returns
    -------
    str
        One of "outer", "right", "left".
    """
    for pos in ("outer", "right", "left"):
        if (pos in grid.axes["X"].coords) and (pos in grid.axes["Y"].coords):
            return pos
    raise ValueError(
        "Only C-grids with vorticity coordinates at a shared 'outer', 'right', or "
        "'left' position on both the X and Y axes are supported."
    )


def corner_offset(grid):
    """
    Integer index shift from a vorticity ("corner") point to its staggered velocity
    point, by corner position (see `corner_position`). 'outer' grids are the baseline
    (0); 'right' grids shift by +1; 'left' grids index like 'outer' (0), differing
    only in array length and in which boundary row/column is absent.

    Parameters
    ----------
    grid: xgcm.Grid

    Returns
    -------
    int
        0 for 'outer'/'left', 1 for 'right'.
    """
    return {"outer": 0, "right": 1, "left": 0}[corner_position(grid)]


def get_geo_corners(grid):
    """
    Find longitude and latitude coordinates from grid dataset, assuming the coordinate
    names contain the sub-strings "lon" and "lat", respectively.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.
        
    Returns
    -------
    dict
        Dictionary containing names of longitude and latitude coordinates.
    """
    pos = corner_position(grid)
    dims = {axis: grid.axes[axis].coords[pos] for axis in ["X", "Y"]}

    coords = grid._ds.coords

    geo_coord_dict = {
        axis: [
            coords[c] for c in coords
            if (
                (geoc in c.lower()) and
                (dims["X"] in coords[c].dims) and
                (dims["Y"] in coords[c].dims)
            )
        ]
        for axis, geoc in zip(["X", "Y"], ["lon", "lat"])
    }
    if any([len(v) == 0 for (k,v) in geo_coord_dict.items()]):
        raise ValueError("""grid._ds must contain two-dimensional ("X", "Y") coordinates including the strings "lon" and "lat", consistent with grid.coords.""")
    return {k:v[0] for (k,v) in geo_coord_dict.items()}


def coord_dict(grid):
    """
    Find names of "X" and "Y" dimension variables from grid dataset.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.
        
    Returns
    -------
    dict
        Dictionary containing names of "X" and "Y" dimension variables, at both cell 'center'
        position and the corner position ('outer', 'right', or 'left'; see `corner_position`).
    """
    corner_pos = corner_position(grid)

    return {
        "X": {
            "center": grid.axes["X"].coords["center"],
            "corner": grid.axes["X"].coords[corner_pos]},
        "Y": {
            "center": grid.axes["Y"].coords["center"],
            "corner": grid.axes["Y"].coords[corner_pos]},
    }
    
def check_outer(grid):
    """
    Check whether the grid's shared vorticity ("corner") position is 'outer'.
    'outer' C-grids have tracers on (M,N) 'center' positions and vorticity on
    (M+1, N+1) 'outer' positions. 'right' and 'left' grids have vorticity on (M,N)
    positions; both return False here. See `corner_position` for the general case.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.

    Returns
    -------
    bool
        True if the corner position is 'outer'; False otherwise ('right' or 'left').
    """
    return corner_position(grid) == "outer"


# ---------------------------------------------------------------------------
# Topology-aware neighbor maps
#
# The section pathfinder walks corner-to-corner across the grid. Rather than
# hard-coding how to step across periodic boundaries, multi-tile face seams, or
# the bipolar north fold, we precompute -- for every corner point -- the
# ([face,] j, i) index of each of its four neighbors ("right", "left", "up",
# "down"). Where there is no neighbor -- a closed domain edge, i.e. a "fill" or
# "extend" boundary -- the point is recorded as its own neighbor. A step in that
# direction therefore stays put, so it never gets the walk closer to the target
# and is never taken: the section stops at the boundary instead of running off
# the array.
#
# For single-tile grids and multi-tile grids with shared ('outer') corners, the
# topology logic lives upstream in `xgcm`: we pad index-valued arrays with the
# grid's own boundary/`face_connections` metadata and read the halos (see
#
# Multi-tile grids with *staggered* ('left'/'right') corners -- MITgcm/ECCO
# lat-lon-cap, cubed-sphere -- are different: a single face's corner lattice is
# incomplete along two of its edges, a physical vorticity point may be stored
# once, not at all, or more than once, and xgcm's halo padding of corner-position
# arrays can land one corner off across rotated/reversed seams. For these grids
# the neighbor maps are instead built from the grid's *outer* (shared-corner)
# lattice: each face is extended by its missing corner row+column, and every
# extended slot is resolved to the native corner that stores that physical point
# by matching the surrounding tracer cells, which pad reliably (with a small
# coordinate fallback only where too few of those cells survive to identify a
# corner at all). The resulting global corner graph -- on which face seams, tile
# junctions and grid cuts are all ordinary nodes -- is projected back onto native
# ([face,] j, i) indices.
# ---------------------------------------------------------------------------
