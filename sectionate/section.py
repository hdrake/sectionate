import numpy as np
import xarray as xr

from .gridutils import (
    get_geo_corners,
    get_facedim,
    build_neighbor_maps,
    outer_topology,
)

# Two corner indices map to the same physical point on seams that fold or wrap (e.g.
# the bipolar north fold). Geodesic distances below this many metres are treated as
# the same point: far below any real grid spacing, far above float round-trip error.
COINCIDENT_TOLERANCE_M = 1.e-3

# Walk determinism tolerance. Curve-deviation differences within this absolute tolerance
# (radians, for both curve types) are treated as tied and broken deterministically by
# index, so the path is independent of platform and of travel direction.
WALK_DEVIATION_ATOL = 1.e-9

# The three curves a section can be asked to follow. "latitude and great circle" is not a
# curve in its own right: it resolves, segment by segment, to one of the other two (see
# `_segment_curve`). `grid_section` documents what each one means.
CURVES = ("great circle", "latitude circle", "latitude and great circle")

# Angular tolerance, in degrees, for deciding whether a segment's two endpoints share a
# latitude -- the classification that says whether a segment follows a parallel. About
# 11 cm on Earth: five orders of magnitude finer than any grid cell, yet coarse enough to
# absorb round-off in waypoints that have been through single precision (a float32
# latitude is off by ~4.e-7 degrees, and model corner coordinates are commonly stored as
# float32).
CONSTANT_LATITUDE_ATOL_DEG = 1.e-6

# Angular tolerance, in degrees, on the separation at which the two ways round a segment
# are equally long. This is a different quantity from the classification tolerance above,
# and it is deliberately tight: it catches endpoints *written* exactly half a circle
# apart (lon 0 -> 180) and nothing else. A segment even slightly short of half a circle is
# not ambiguous to the code -- which way round it runs is settled by the sign of the
# round-off, so 0 -> 179.9999999 goes east and 0 -> 180.0000001 goes west -- so subdivide
# such segments with an intermediate waypoint rather than relying on this check.
HALF_CIRCLE_ATOL_DEG = 1.e-9

class Section():
    """A named hydrographic section"""
    def __init__(self, name, coords, children = {}, parent = None):
        """Initiate named hydrographic section

        Arguments
        ---------
        name [str] -- name of the section
        coords [list or tuple] -- coordinates that define the section
        
            If type is list, elements of the list must all be 2-tuples
            of the form (lon, lat).
            
            If type is tuple, it must be of the form (lons, lats), where
            lons and lats are lists of np.ndarray instances of the same
            length with elements of type float.

        Keyword Arguments
        -----------------
        children [mapping from str to Section (default: {})] -- dictionary
            mapping the names of child sections to their Section instances.
            This attribute will generally be populated automatically from
            the function `join_sections`.

        parent [Section (default: None)] -- TO DO

        Returns
        -------
        instance of Section

        Examples
        --------
        >>> sec.Section("Bering Strait", [(-170.3, 66.1), (-167.6,65.7)])
        Section(Bering Strait, [(-170.3, 66.1), (-167.6, 65.7)])
        """
        self.name = name
        if type(coords) is tuple:
            if len(coords) == 2:
                self.update_coords(coords_from_lonlat(coords[0], coords[1]))
            else:
                raise ValueError("If coords is a tuple, must be (lons, lats)")
        elif type(coords) is list:
            if all([(type(c) is tuple) and (len(c)==2) for c in coords]):
                self.coords = coords.copy()
                self.lons_c, self.lats_c = lonlat_from_coords(self.coords)
            else:
                raise ValueError("If coords is a list, its elements must be (lon,lat) 2-tuples")
        else:
            raise ValueError("coords must be a 2-tuple of lists/arrays or a list of 2-tuples")
            
        self.children = children.copy() # need this to be a copy or get a recursion error in __repr__...
        self.parent = parent
        self.save = {}

    def reverse(self):
        """Reverse the section's direction"""
        self.update_coords(self.coords[::-1])
        return self

    def update_coords(self, coords):
        """Update coordinates (including longitude and latitude arrrays)"""
        self.coords = coords
        self.lons_c, self.lats_c = lonlat_from_coords(self.coords)

    def copy(self):
        """Create a deep copy of the Section instance"""
        section = Section(
            self.name,
            self.coords,
            children=self.children,
            parent=self.parent
        )
        section.save = self.save.copy()
        return section
    
    def __repr__(self, indent=0, show_attributes=True):
        indent_str = "  " * indent
        summary = f"{indent_str}Section({self.name}, {self.coords})"
        
        # Automatically extract and add attributes
        if show_attributes:
            summary += f"\n{indent_str}attributes:"
            for attr, value in vars(self).items():
                if attr not in ['children', 'save'] and not attr.startswith('_'):
                    summary += f"\n{indent_str}  {attr}"
    
        if len(self.children) > 0:
            summary += f"\n{indent_str}  children:"
            for child in self.children.values():
                child_repr = child.__repr__(indent + 3, show_attributes=False)
                summary += f"\n{indent_str}    - {child_repr.lstrip()}"
        return summary
        
class GriddedSection(Section):
    """Initiate named hydrographic section specific to an ocean model grid

    Arguments
    ---------
    section [sectionate.Section] -- named Sectionate section
    grid [xgcm.Grid] -- ocean model grid object

    Keyword Arguments
    -----------------
    i_c [list or np.ndarray] -- x corner indices
    j_c [list or np.ndarray] -- y corner indices

    Returns
    -------
    instance of GriddedSection
    """
    def __init__(self, section, grid, i_c=None, j_c=None, f_c=None):
        super().__init__(
            section.name,
            section.coords,
            children = section.children,
            parent = section.parent
        )
        self.grid = grid
        self.f_c = f_c
        if isinstance(i_c, (list, np.ndarray)) & isinstance(j_c, (list, np.ndarray)):
            self.i_c = i_c
            self.j_c = j_c
        else:
            self.grid_section()

    def grid_section(self, **kwargs):
        """Pass this Section's coordinates to sectionate.grid_section

        Arguments
        ---------
        grid

        Keyword Arguments
        -----------------
        **kwargs passed directly to sectionate.grid_section
        """
        out = grid_section(
            self.grid,
            self.lons_c,
            self.lats_c,
            **kwargs
        )
        if len(out) == 5:
            self.i_c, self.j_c, self.f_c, self.lons_c, self.lats_c = out
        else:
            self.i_c, self.j_c, self.lons_c, self.lats_c = out
            self.f_c = None

        return out
    
    def copy(self):
        """Create a copy of this GriddedSection, with deep copies of all attributes
        except the `grid`, which is shared (not copied)."""
        new = GriddedSection(
            self,                       # supplies name, coords, children, parent
            self.grid,                  # shared, not copied
            i_c=self.i_c.copy(),
            j_c=self.j_c.copy(),
            f_c=None if self.f_c is None else self.f_c.copy(),
        )
        # Carry over the gridded path coordinates (grid_section overwrites lons_c/lats_c
        # in place; `__init__` would otherwise reset them to the raw waypoint coords).
        new.lons_c = self.lons_c.copy()
        new.lats_c = self.lats_c.copy()
        new.save = self.save.copy()
        return new


def join_sections(name, *sections, **kwargs):
    """
    Joins child Sections together to create a parent Section.

    Arguments
    ---------
    name [str] -- name of the parent section
    *sections [Section] -- the sequence of child sections to be joined

    Keyword Arguments
    -----------------
    align [bool (Default : True)] -- reverse sections as needed to minimize
        the distance between end/start points of consecutive sections

    Returns
    -------
    instance of Section

    Example
    -------
    >>> section1 = sec.Section("section1", ([0., 100.], [0., 0.]))
    >>> section2 = sec.Section("section2", ([100., 200.], [0., 0.]))
    >>> section = sec.join_sections("section", section1, section2)
    >>> section
    Section(section, [(0.0, 0.0), (100.0, 0.0), (100.0, 0.0), (200.0, 0.0)])
 Children:
  - Section(section1, [(0.0, 0.0), (100.0, 0.0)])
  - Section(section2, [(100.0, 0.0), (200.0, 0.0)])
    """
    if type(name) is not str:
        raise ValueError("first argument (name) must be a str.")
    elif any([not(isinstance(s, Section)) for s in sections]):
        raise ValueError("all positional arguments after the first must be instances of Section")
    align = kwargs["align"] if "align" in kwargs else True
    extend = kwargs["extend"] if "extend" in kwargs else False
    
    section = Section(name, sections[0].coords)
    if len(sections) > 1:
        for i, s in enumerate(sections[1:], start=1):
            if not(align):
                coords1, coords2 = section.coords, s.coords
            else:
                coords1, coords2 = align_coords(
                    section.coords,
                    s.coords,
                    extend=extend
                )

            s.update_coords(coords2)
            section.update_coords(coords1 + coords2)
            
            if i == 1:
                sections[0].update_coords(coords1)
                section.children[sections[0].name] = sections[0]
            section.children[s.name] = s
            
    return section
        
def grid_section(grid, lons, lats, curve="great circle"):
    """
    Compute composite section along model `grid` velocity faces that approximates paths
    between consecutive points defined by (lons, lats).

    The grid topology is inferred entirely from the `grid` metadata: each axis' `padding`
    condition ("periodic" wraps, otherwise clip) for single-tile grids, and `face_connections`
    for multi-tile grids (e.g. the lat-lon-cap or cubed-sphere).

    Parameters
    ----------
    grid: xgcm.Grid
        Object describing the geometry of the ocean model grid, including metadata about variable names for
        the staggered C-grid dimensions and coordinates.
    lons: list or np.ndarray
        Longitudes, in degrees, of consecutive vertices defining a piece-wise geodesic section.
    lats: list or np.ndarray
        Latitudes, in degrees (in range [-90, 90]), of consecutive vertices defining a piece-wise section.
    curve: str
        Curve followed between consecutive vertices. One of:

        - "great circle" (default): every segment follows the geodesic.
        - "latitude circle": every segment follows a circle of constant latitude, marching
          in longitude. A segment whose endpoints do not share a latitude (to within
          1.e-6 degrees) lies on no such circle, so it raises a ValueError.
        - "latitude and great circle": decided per segment. A segment whose endpoints
          share a latitude follows the parallel; every other segment follows the geodesic.
          This is the option for a section that is zonal in places and joined up by arbitrarily
          oriented legs elsewhere.

        Under every option each segment takes the **shortest** path between its two
        vertices. Raw longitudes are never read as a request to go the long way round, so
        a segment written 0 -> 270 along the equator runs 90 degrees *west*. Encircle the
        globe by giving intermediate vertices (e.g. 0 -> 120 -> 240 -> 360), which is also
        what says which way round it goes.

    Returns
    -------
    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c) correspond to indices of vorticity points that define velocity faces. For
        multi-tile grids, the face index f_c of each point is returned as well.
        (lons_c, lats_c) are the corresponding longitude and latitudes.
    """
    geocorners = get_geo_corners(grid)
    facedim = get_facedim(grid)

    if facedim is not None:
        _check_supported_topology(grid)

    # All topologies -- periodic wrap, fill/extend walls, multi-tile face
    # connections, and the bipolar north fold -- are derived uniformly from the
    # grid's xgcm metadata by padding index arrays (see `build_neighbor_maps`).
    # Where one physical corner is stored under two indices (a periodic seam, a shared
    # multi-tile boundary corner, or the fold seam), the walk simply steps through
    # both; that hand-off step spans no grid cell, so the redundant index is dropped
    # right here, by `drop_repeated_corners`, before the section is returned.
    neighbor_maps = build_neighbor_maps(grid, geocorners)

    return create_section_composite(
        geocorners["X"],
        geocorners["Y"],
        lons,
        lats,
        neighbor_maps=neighbor_maps,
        curve=curve,
        grid=grid,
    )


def drop_repeated_corners(grid, i_c, j_c, f_c=None):
    """
    Normalise a traced grid path so that consecutive corners are always distinct
    physical points.

    A single physical vorticity point can be stored under more than one index: the
    duplicated seam column (or row) of a symmetric ('outer') periodic grid, a corner
    shared by several tiles of a multi-tile grid, the two coincident lips of a grid cut,
    or the whole degenerate column where a bipolar cap converges on one of its poles.
    `infer_grid_path_from_geo` deliberately admits the "seam twin" of the corner it is
    standing on, so that a seam crossing is decided by the grid's topology rather than by
    floating-point rounding. The resulting hand-off step spans no grid cell, so that pair
    of corners does not define a velocity face.

    `grid_section` calls this immediately after the path is traced, so that every section
    it returns satisfies:

    * no two consecutive corners are the same physical point -- every consecutive pair
      is a real velocity face; and
    * on a multi-tile grid, every corner is the canonical native index of the physical
      corner point it denotes (`_OuterTopology.node_native`), which is what lets the
      velocity-face attribution look a corner up in either the source or the
      destination face's frame.

    Of a repeated corner's several indices, the one kept is the one that names the same
    two velocity faces as the run it replaces: the faces flanking the run keep exactly
    the indices they would have had if the walk had never changed frame. On a multi-tile
    grid that is automatic, since every index of a corner resolves to the same canonical
    native one. On a single tile the LAST index of the run -- the frame the path continues
    in -- is preferred and is what a seam hand-off needs; the earlier ones are tried in
    turn if it does not serve, and a run that no single index can stand in for raises
    (see `_collapse_single_tile_runs`).

    `uvindices_from_qindices` requires this invariant and checks it, so hand-built index
    arrays that were not produced by `grid_section` can be passed through here first.

    PARAMETERS:
    -----------
    grid: xgcm.Grid
        Grid the path was traced on.
    i_c, j_c: array-like of int
        Vorticity-point indices along the "X" and "Y" dimensions.
    f_c: array-like of int or None
        Face indices for multi-tile grids (`face_connections`); None for single-tile.

    RETURNS:
    --------
    i_c, j_c, f_c, lons_c, lats_c: np.ndarray
        The normalised path and the coordinates of its corners (`f_c` is None for
        single-tile grids).
    """
    i_c = np.asarray(i_c, dtype=np.int64)
    j_c = np.asarray(j_c, dtype=np.int64)
    f_c = None if f_c is None else np.asarray(f_c, dtype=np.int64)

    geocorners = get_geo_corners(grid)
    glon = np.asarray(geocorners["X"].values)
    glat = np.asarray(geocorners["Y"].values)

    if f_c is not None:
        # Multi-tile: which physical point an index denotes is exact and purely
        # topological -- two indices denote the same point iff they resolve to the same
        # node of the corner graph.
        ot = outer_topology(grid)
        nodes = ot.node_id[f_c, j_c + ot.t, i_c + ot.t]
        if (nodes < 0).any():
            raise ValueError(
                "Section contains indices that are not grid corners."
            )
        native = ot.node_native[nodes]
        if (native[:, 0] < 0).any():
            raise ValueError(
                "Section passes through a corner point that is stored on no face of "
                "the grid (e.g. a cube vertex or an unstored cap/cut corner), so it "
                "has no native index."
            )
        f_c, j_c, i_c = native[:, 0].copy(), native[:, 1].copy(), native[:, 2].copy()
        # Every index of a corner has just been rewritten to the node's one canonical
        # native index, so the members of a run are now identical: which one survives
        # cannot matter, and the neighbour maps link canonical indices only.
        repeated = nodes[1:] == nodes[:-1]
        keep = np.ones(i_c.size, dtype=bool)
        keep[:-1] = ~repeated
    else:
        # Single tile: the seam twins of a periodic/fold boundary are separate indices
        # with nothing to compare them by, so coincidence is measured physically. The
        # tolerance sits far below any real grid spacing (see COINCIDENT_TOLERANCE_M).
        lo, la = glon[j_c, i_c], glat[j_c, i_c]
        repeated = distance_on_unit_sphere(
            lo[:-1], la[:-1], lo[1:], la[1:]
        ) < COINCIDENT_TOLERANCE_M
        keep = _collapse_single_tile_runs(grid, i_c, j_c, repeated)

    i_c, j_c = i_c[keep], j_c[keep]
    f_c = None if f_c is None else f_c[keep]

    if f_c is not None:
        lons_c, lats_c = glon[f_c, j_c, i_c], glat[f_c, j_c, i_c]
    else:
        lons_c, lats_c = glon[j_c, i_c], glat[j_c, i_c]
    return i_c, j_c, f_c, np.asarray(lons_c), np.asarray(lats_c)


def _collapse_single_tile_runs(grid, i_c, j_c, repeated):
    """
    Boolean mask keeping exactly one index out of each run of repeated corners in a
    single-tile path (`repeated[k]` is True where corners k and k+1 are the same
    physical point).

    Collapsing a run replaces the index the path arrived at with the index it is left
    under, so it is only safe when the surviving index still names the two velocity
    faces that flank the run. That is checked, not assumed: the flanking faces are
    derived from the raw path and from each candidate index, with
    `transports._single_tile_face` -- the same arithmetic
    `transports.uvindices_from_qindices` will apply -- and only an index reproducing
    both is kept. Compared are the velocity (var, i, j) and the direction flag that
    carries that face's transport sign ("Xinc" for a V face, "Yinc" for a U face; see
    `convergent_transport`), so collapsing changes neither a face of the section nor
    the sign of its transport.

    Candidates are tried last-index-first, because for a seam hand-off (the common case,
    where a run is just a corner's two seam-twin indices) the last one is the frame the
    path continues in.

    Where a whole grid column degenerates to one point -- the pole of a bipolar cap,
    stored as every corner of a column -- a section can enter the point along one of its
    indices and leave along another, far apart in the index lattice. Then no single index
    names both flanking faces, and this raises rather than emit a face the section does
    not have.
    """
    # Imported here, not at module scope: `transports` imports this module for the
    # coincidence test, and the face arithmetic must have exactly one implementation.
    from .transports import _single_tile_face, _single_tile_face_context

    keep = np.ones(i_c.size, dtype=bool)
    if not np.any(repeated):
        return keep

    context = _single_tile_face_context(grid)

    def face(a, b):
        """The velocity face between path positions a and b, with its transport sign."""
        var, vi, vj, Xinc, Yinc = _single_tile_face(
            int(i_c[a]), int(j_c[a]), int(i_c[b]), int(j_c[b]), *context
        )
        return var, vi, vj, (Xinc if var == "V" else Yinc)

    n = i_c.size
    start = 0
    while start < n:
        end = start
        while end < n - 1 and repeated[end]:
            end += 1
        if end > start:
            before = start - 1 if start > 0 else None
            after = end + 1 if end + 1 < n else None
            face_in = None if before is None else face(before, start)
            face_out = None if after is None else face(end, after)
            chosen = None
            for r in range(end, start - 1, -1):
                if before is not None and face(before, r) != face_in:
                    continue
                if after is not None and face(r, after) != face_out:
                    continue
                chosen = r
                break
            if chosen is None:
                raise ValueError(
                    f"Section corners {start}..{end} -- (i,j) "
                    f"{(int(i_c[start]), int(j_c[start]))} through "
                    f"{(int(i_c[end]), int(j_c[end]))} -- are all the same physical grid "
                    "corner, a column that degenerates to a single point (a bipolar "
                    "cap's pole). The section enters it from "
                    f"{(int(i_c[before]), int(j_c[before]))} and leaves it towards "
                    f"{(int(i_c[after]), int(j_c[after]))}, and no single index of that "
                    "corner is adjacent to both, so no section with distinct consecutive "
                    "corners can carry both of the velocity faces on either side. If this "
                    "grid has a bipolar/tripolar fold, declare it -- "
                    "padding={'Y': {'fold': 'corner'}} -- so the walk crosses the cap "
                    "through the fold seam rather than through the pole column; that is "
                    "what this arises from. Otherwise route the section around the pole."
                )
            keep[start:end + 1] = False
            keep[chosen] = True
        start = end + 1
    return keep


def _check_supported_topology(grid):
    """
    Raise if the multi-tile `grid` requires topology features sectionate does not yet support.

    A tripolar/bipolar north fold expressed as a `face_connections` self-connection (a face that
    connects to itself along the "Y" axis) is not supported. Use the single-tile bipolar-fold
    padding instead -- ``padding={"X": "periodic", "Y": {"fold": "corner"}}`` (xgcm >= the
    bipolar-fold release) -- which sectionate handles natively via xgcm's fold padding.

    A multi-tile grid must also register a cell-center dimension on both horizontal axes.
    That is what the corner-node topology is reconstructed from (`gridutils.outer_topology`
    fingerprints each corner by the tracer cells around it, because cells -- unlike corner
    arrays -- pad reliably across any seam), and it is also where half of each velocity's
    dimensions live: U sits at (X-corner, Y-center) and V at (X-center, Y-corner). Only the
    *dimensions* are needed, not tracer longitudes/latitudes. Without them a multi-tile grid
    can carry no velocities and each of its seam corners cannot be resolved to one index, so
    a section across a seam could not be traced correctly; refuse it at the front door rather
    than return a path whose seam steps are wrong.
    """
    facedim = grid._facedim
    connections = (getattr(grid, "_face_connections", None) or {}).get(facedim, {})
    for face, axis_sides in connections.items():
        for sides in axis_sides.values():
            for side in sides:
                if side is None:
                    continue
                neighbor_face = side[0]
                if neighbor_face == face:
                    raise NotImplementedError(
                        "Grids that represent the tripolar/bipolar north fold as a face that "
                        "connects to itself (a `face_connections` self-connection) are not "
                        "supported. Build the grid as a single tile with the fold expressed as "
                        "padding instead: padding={'X':'periodic','Y':{'fold':'corner'}}."
                    )

    missing = [ax for ax in ("X", "Y") if "center" not in grid.axes[ax].coords]
    if missing:
        raise ValueError(
            "Multi-tile grids (`face_connections`) must register a cell-center dimension on "
            f"both horizontal axes; {' and '.join(missing)} declare none. Sectionate rebuilds "
            "the shared-corner topology from the tracer cells around each corner, and the "
            "velocities themselves are staggered onto the center dimensions (U at (X-corner, "
            "Y-center), V at (X-center, Y-corner)), so a grid without them carries no "
            "velocities and its seam corners cannot be identified. Declare them, e.g. "
            "coords={'X': {'center': 'xh', 'outer': 'xq'}, "
            "'Y': {'center': 'yh', 'outer': 'yq'}} -- the center entries need only name "
            "dimensions of the dataset, they need not carry coordinate values."
        )

def create_section_composite(
    gridlon,
    gridlat,
    lons,
    lats,
    neighbor_maps,
    curve="great circle",
    grid=None,
    ):
    """
    Compute composite section along velocity faces, as defined by coordinates of vorticity points (gridlon, gridlat),
    that most closely approximates geodesic paths between consecutive points defined by (lons, lats).

    PARAMETERS:
    -----------

    gridlon: np.ndarray
        Array of longitude, in degrees. 2d (Y, X) for single-tile grids; 3d (face, Y, X) for
        multi-tile grids (`face_connections`).
    gridlat: np.ndarray
        Array of latitude, in degrees, with the same shape as `gridlon`.
    lons: list of float
        longitude of section starting, intermediate and end points, in degrees
    lats: list of float
        latitude of section starting, intermediate and end points, in degrees
    neighbor_maps: dict
        Topology-aware neighbor maps from `sectionate.gridutils.build_neighbor_maps`
        (single- or multi-tile). Sections are always built from an `xgcm.Grid`, so these
        are always supplied; the usual entry point is `sectionate.grid_section`.
    curve: str
        Curve followed between consecutive vertices: "great circle" (default),
        "latitude circle", or "latitude and great circle". Each segment is resolved and
        checked independently; `sectionate.grid_section` documents what the three options
        mean and which segments they reject.
    grid: xgcm.Grid or None
        The grid `neighbor_maps` and (gridlon, gridlat) came from. Pass it to get a section
        that satisfies the invariant the rest of the API expects -- no two consecutive
        corners at the same physical point -- by normalizing the raw walk with
        `drop_repeated_corners`. Without it the raw walk is returned unchanged, which
        `transports.uvindices_from_qindices` refuses; `grid_section` always passes it.

    RETURNS:
    -------

    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c[, f_c]) correspond to indices of vorticity points that define velocity faces;
        the face index f_c is only returned for multi-tile grids.
        (lons_c, lats_c) are the corresponding longitude and latitudes.

        Without `grid` this is the raw walk: where a section crosses a seam whose corner is
        stored under two indices, both are still present, and the step between them is not
        a velocity face.
    """

    # A face dimension (3-D corner arrays) marks a multi-tile grid, whose sections
    # carry a per-point face index. This is independent of `neighbor_maps`, which is
    # now always provided for grid-derived sections (single- and multi-tile alike).
    multitile = np.ndim(gridlon) == 3

    i_c = np.array([], dtype=np.int64)
    j_c = np.array([], dtype=np.int64)
    f_c = np.array([], dtype=np.int64)
    lons_c = np.array([], dtype=np.float64)
    lats_c = np.array([], dtype=np.float64)

    if len(lons) != len(lats):
        raise ValueError("lons and lats should have the same length")

    for k in range(len(lons) - 1):
        seg = create_section(
            gridlon,
            gridlat,
            lons[k],
            lats[k],
            lons[k + 1],
            lats[k + 1],
            neighbor_maps=neighbor_maps,
            curve=curve,
        )
        if multitile:
            i_c_seg, j_c_seg, f_c_seg, lons_c_seg, lats_c_seg = seg
        else:
            i_c_seg, j_c_seg, lons_c_seg, lats_c_seg = seg

        i_c = np.concatenate([i_c, i_c_seg[:-1]], axis=0)
        j_c = np.concatenate([j_c, j_c_seg[:-1]], axis=0)
        lons_c = np.concatenate([lons_c, lons_c_seg[:-1]], axis=0)
        lats_c = np.concatenate([lats_c, lats_c_seg[:-1]], axis=0)
        if multitile:
            f_c = np.concatenate([f_c, f_c_seg[:-1]], axis=0)

    i_c = np.concatenate([i_c, [i_c_seg[-1]]], axis=0)
    j_c = np.concatenate([j_c, [j_c_seg[-1]]], axis=0)
    lons_c = np.concatenate([lons_c, [lons_c_seg[-1]]], axis=0)
    lats_c = np.concatenate([lats_c, [lats_c_seg[-1]]], axis=0)
    if multitile:
        f_c = np.concatenate([f_c, [f_c_seg[-1]]], axis=0)

    i_c, j_c = i_c.astype(np.int64), j_c.astype(np.int64)
    f_c = f_c.astype(np.int64) if multitile else None
    if grid is not None:
        i_c, j_c, f_c, lons_c, lats_c = drop_repeated_corners(grid, i_c, j_c, f_c)
    if multitile:
        return i_c, j_c, f_c, lons_c, lats_c

    return i_c, j_c, lons_c, lats_c

def create_section(gridlon, gridlat, lonstart, latstart, lonend, latend, neighbor_maps, curve="great circle"):
    """
    Compute a section segment along velocity faces, as defined by coordinates of vorticity points (gridlon, gridlat),
    that most closely approximates the geodesic path between points (lonstart, latstart) and (lonend, latend).

    PARAMETERS:
    -----------

    gridlon: np.ndarray
        2d array of longitude (with dimensions ("Y", "X")), in degrees
    gridlat: np.ndarray
        2d array of latitude (with dimensions ("Y", "X")), in degrees
    lonstart: float
        longitude of starting point, in degrees
    lonend: float
        longitude of end point, in degrees
    latstart: float
        latitude of starting point, in degrees
    latend: float
        latitude of end point, in degrees
    neighbor_maps: dict
        Topology-aware neighbor maps from `sectionate.gridutils.build_neighbor_maps`
        (single- or multi-tile).

    RETURNS:
    -------

    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c[, f_c]) correspond to indices of vorticity points that define velocity faces;
        the face index f_c is only returned for multi-tile grids.
        (lons_c, lats_c) are the corresponding longitude and latitudes.
    """

    return infer_grid_path_from_geo(
        lonstart,
        latstart,
        lonend,
        latend,
        gridlon,
        gridlat,
        neighbor_maps=neighbor_maps,
        curve=curve,
    )

def _wrapped_dlon(lon1, lon2):
    """Signed longitude change from `lon1` to `lon2`, wrapped into [-180, 180).

    This is the change along the *shortest* way round: a raw change of +270 degrees comes
    back as -90. Endpoints that coincide modulo 360 degrees (e.g. a 360 -> 0 loop
    closure) give 0.
    """
    return (lon2 - lon1 + 180.) % 360. - 180.


def _is_constant_latitude(lat1, lat2):
    """Whether a segment's two endpoint latitudes agree, and so lie on one parallel.

    The single classification used both to accept or reject a "latitude circle" segment
    and to pick the metrics a segment is walked with, so those two can never disagree.
    """
    return abs(lat2 - lat1) <= CONSTANT_LATITUDE_ATOL_DEG


def _segment_curve(lat1, lat2, curve):
    """The curve one segment actually follows: always "great circle" or "latitude circle".

    Resolves the section-wide `curve` request (one of `CURVES`) for a single segment, and
    raises ValueError if it is not a recognized request. Passing an already-resolved value
    returns it unchanged, so this is safe to apply more than once along a call chain.
    """
    if curve in ("great circle", "latitude circle"):
        return curve
    if curve == "latitude and great circle":
        return "latitude circle" if _is_constant_latitude(lat1, lat2) else "great circle"
    raise ValueError(
        f"curve must be one of {', '.join(repr(c) for c in CURVES)}; got {curve!r}."
    )


def _check_segment_span(lon1, lat1, lon2, lat2, curve):
    """Validate one section segment and return the curve it follows.

    `curve` is the section-wide request; the return value is what it resolves to for this
    segment, either "great circle" or "latitude circle". Raises ValueError if the segment
    is ill posed under `curve` -- see `sectionate.grid_section` for the rules and for what
    to write instead.
    """
    segment_curve = _segment_curve(lat1, lat2, curve)

    if curve == "latitude circle" and not _is_constant_latitude(lat1, lat2):
        raise ValueError(
            f"Segment from (lon={lon1}, lat={lat1}) to (lon={lon2}, lat={lat2}) does not "
            f"follow a circle of constant latitude: its endpoints differ in latitude by "
            f"{abs(lat2 - lat1)} degrees. Use curve='latitude and great circle' to follow "
            "the parallel where the endpoints do share a latitude and the geodesic "
            "everywhere else, or curve='great circle' throughout."
        )

    if segment_curve == "latitude circle":
        # A segment along a parallel travels only in longitude, so the separation that
        # decides ambiguity is the shortest-way-round longitude change -- degrees of
        # longitude, which near the poles is a far larger number than the arc it spans.
        sep = abs(_wrapped_dlon(lon1, lon2))
        measure = "degrees of longitude along the parallel"
    else:
        sep = np.rad2deg(distance_on_unit_sphere(lon1, lat1, lon2, lat2, R=1.))
        measure = "degrees of arc"

    if sep >= 180. - HALF_CIRCLE_ATOL_DEG:
        raise ValueError(
            f"Segment from (lon={lon1}, lat={lat1}) to (lon={lon2}, lat={lat2}) has "
            f"endpoints half a circle apart ({sep:.4f} {measure}), so neither way round "
            "is the shorter and there is no shortest path to take. Add an intermediate "
            "waypoint to say which way the section goes."
        )

    return segment_curve


def infer_grid_path_from_geo(lonstart, latstart, lonend, latend, gridlon, gridlat, neighbor_maps, curve="great circle"):
    """
    Find the grid indices (and coordinates) of vorticity points that most closely approximates
    the geodesic path between points (lonstart, latstart) and (lonend, latend).

    PARAMETERS:
    -----------

    lonstart: float
        longitude of section starting point, in degrees
    latstart: float
        latitude of section starting point, in degrees
    lonend: float
        longitude of section end point, in degrees
    latend: float
        latitude of section end point, in degrees
    gridlon: np.ndarray
        Array of longitude, in degrees. 2d (Y, X) for single-tile grids; 3d (face, Y, X) for
        multi-tile grids (`face_connections`).
    gridlat: np.ndarray
        Array of latitude, in degrees, with the same shape as `gridlon`.
    neighbor_maps: dict
        Topology-aware neighbor maps from `sectionate.gridutils.build_neighbor_maps`
        (single- or multi-tile).

    RETURNS:
    -------

    i_c, j_c[, f_c], lons_c, lats_c: `np.ndarray`
        (i_c, j_c[, f_c]) correspond to indices of vorticity points that define velocity faces;
        the face index f_c is only returned for multi-tile grids.
        (lons_c, lats_c) are the corresponding longitude and latitudes.
    """

    # This function is called once per segment, so it is where the section-wide `curve`
    # request becomes the one curve this segment follows. Resolve it here, from the
    # *requested* waypoints, rather than leaving it to `infer_grid_path`, which sees only
    # the grid corners the waypoints snap to -- and whose latitudes can differ from the
    # requested ones by up to half a cell.
    segment_curve = _check_segment_span(lonstart, latstart, lonend, latend, curve)

    multitile = np.ndim(gridlon) == 3
    if multitile:
        istart, jstart, fstart = find_closest_grid_point(lonstart, latstart, gridlon, gridlat)
        iend, jend, fend = find_closest_grid_point(lonend, latend, gridlon, gridlat)
    else:
        istart, jstart = find_closest_grid_point(lonstart, latstart, gridlon, gridlat)
        iend, jend = find_closest_grid_point(lonend, latend, gridlon, gridlat)
        fstart, fend = None, None

    return infer_grid_path(
        istart,
        jstart,
        iend,
        jend,
        gridlon,
        gridlat,
        neighbor_maps=neighbor_maps,
        curve=segment_curve,
        f1=fstart,
        f2=fend,
    )


def infer_grid_path(i1, j1, i2, j2, gridlon, gridlat, neighbor_maps, f1=None, f2=None, curve="great circle"):
    """
    Find the grid indices (and coordinates) of vorticity points that most closely approximate
    the geodesic path between the starting and ending corner points.

    PARAMETERS:
    -----------

    i1: integer
        i-coord of point1
    j1: integer
        j-coord of point1
    i2: integer
        i-coord of point2
    j2: integer
        j-coord of point2
    gridlon: np.ndarray
        Array of longitude, in degrees. 2d (Y, X) for single-tile grids; 3d (face, Y, X) for
        multi-tile grids (`face_connections`).
    gridlat: np.ndarray
        Array of latitude, in degrees, with the same shape as `gridlon`.
    neighbor_maps: dict
        Topology-aware neighbor maps from `sectionate.gridutils.build_neighbor_maps`. The
        grid's full topology (periodic wrap, fill/extend walls, multi-tile face seams, and
        the bipolar north fold) is encoded here, so a grid is always required to produce
        them -- typically via `sectionate.grid_section`.
    f1, f2: integer or None
        Face indices of the starting and ending points (multi-tile grids only); None otherwise.
    curve: str
        Curve this segment follows; see `sectionate.grid_section`. "latitude and great
        circle" resolves here from the two endpoint corners' latitudes, since this entry
        point is given indices rather than requested waypoints.

    RETURNS:
    -------

    For single-tile grids:
        i_c_seg, j_c_seg, lons_c_seg, lats_c_seg
    For multi-tile grids, additionally the face index of each point:
        i_c_seg, j_c_seg, f_c_seg, lons_c_seg, lats_c_seg

    (i_c_seg, j_c_seg[, f_c_seg]) are the vorticity-point indices bounded by the start and end
    points; (lons_c_seg, lats_c_seg) are the corresponding longitude and latitude.
    """
    if isinstance(gridlon, xr.core.dataarray.DataArray):
        gridlon = gridlon.values
    if isinstance(gridlat, xr.core.dataarray.DataArray):
        gridlat = gridlat.values

    if neighbor_maps is None:
        raise ValueError(
            "infer_grid_path requires topology-aware `neighbor_maps`. Build them from an "
            "xgcm.Grid with sectionate.gridutils.build_neighbor_maps(grid, get_geo_corners(grid)), "
            "or use the high-level sectionate.grid_section(grid, lons, lats)."
        )

    multitile = gridlon.ndim == 3
    if multitile:
        nfaces, ny, nx = gridlon.shape
    else:
        ny, nx = gridlon.shape
        nfaces = 1

    def coord(arr, f, j, i):
        return arr[j, i] if f is None else arr[f, j, i]

    def neighbor(direction, f, j, i):
        fmap, jmap, imap = neighbor_maps[direction]
        if fmap is None:
            return (None, int(jmap[j, i]), int(imap[j, i]))
        return (int(fmap[f, j, i]), int(jmap[f, j, i]), int(imap[f, j, i]))

    # target coordinates
    lon1, lat1 = coord(gridlon, f1, j1, i1), coord(gridlat, f1, j1, i1)
    lon2, lat2 = coord(gridlon, f2, j2, i2), coord(gridlat, f2, j2, i2)

    segment_curve = _segment_curve(lat1, lat2, curve)

    # Per-curve metrics used by the deterministic neighbor selection below.
    #  - progress(lon, lat): remaining distance to the segment endpoint (smaller = nearer);
    #    admits only neighbors that do not move away from the endpoint.
    #  - deviation(lon, lat): how far a candidate lies from the desired curve between the
    #    endpoints (smaller = better); chooses among admitted neighbors. It is symmetric in
    #    the two endpoints so the path does not depend on travel direction, and is in radians
    #    for both curve types so WALK_DEVIATION_ATOL is meaningful for both.
    # Physical coincidence with the endpoint (the seam-twin stop) always uses true geodesic
    # distance, independent of `curve`.
    if segment_curve == "great circle":
        def progress(lon, lat):
            return distance_on_unit_sphere(lon, lat, lon2, lat2)
        def deviation(lon, lat):
            return (spherical_angle(lon2, lat2, lon1, lat1, lon, lat)
                    + spherical_angle(lon1, lat1, lon2, lat2, lon, lat))
    else:   # "latitude circle" -- the only other value `_segment_curve` returns
        # Progress purely in longitude, deviation purely in latitude: the metrics of a
        # march along a parallel, and meaningful only for a segment whose endpoints share
        # a latitude. Segments that do not are never resolved to this curve.
        def progress(lon, lat):
            # sin^2(delta-lon/2) is the haversine of the longitude gap: periodic in 360
            # degrees and monotonic in |delta-lon| up to 180, so it measures the *shortest*
            # way round regardless of how the endpoint longitudes were written (0 -> 270
            # descends westward just as 0 -> -90 does). Direction-symmetric.
            return np.sin(np.deg2rad((lon - lon2) / 2.)) ** 2
        def deviation(lon, lat):
            # angular distance off the constant-latitude curve through the endpoints
            # (radians). Flat between the two endpoint latitudes, which absorbs the
            # sub-cell mismatch left when each endpoint snaps to its nearest grid corner.
            return np.deg2rad(abs(lat - lat1)) + np.deg2rad(abs(lat - lat2))

    def order_key(pt):
        _f, _j, _i = pt
        return (-1 if _f is None else _f, _j, _i)

    # init loop position to starting point
    f, j, i = f1, j1, i1

    i_c_seg = [i]  # add first point to list of points
    j_c_seg = [j]  # add first point to list of points
    f_c_seg = [f]  # add first point to list of points

    # iterate through the grid path steps until we reach end of section
    ct = 0 # grid path step counter
    # safety bound: enough steps to cross the whole grid (all faces) once
    nstep_max = (nx + ny + 1) * nfaces

    # Grid-agnostic algorithm (see the per-step selection below):
    # First, find all four neighbors (using grid topology via `neighbor_maps`).
    # Second, keep those strictly closer to the endpoint (plus any seam twin of the current
    #   point), excluding the previous point.
    # Third, step to the admitted neighbor closest to the desired `curve`, with a
    #   deterministic, direction-independent tie-break.
    f_prev, j_prev, i_prev = f, j, i
    while (f, j, i) != (f2, j2, i2):

        # safety precaution: exit after taking enough steps to have crossed the entire model grid
        if ct > nstep_max:
            raise RuntimeError(f"Should have reached the endpoint by now.")

        d_current = distance_on_unit_sphere(
                coord(gridlon, f, j, i),
                coord(gridlat, f, j, i),
                lon2,
                lat2
            )

        # We are done once we reach the endpoint. On grids where two distinct
        # corner indices map to the same physical point -- the bipolar fold seam,
        # where corner i is identified with its mirror image -- the walker can land
        # on the endpoint's twin index rather than the endpoint index itself, so we
        # also stop on physical coincidence. The tolerance (in metres) sits far below
        # any real grid spacing yet well above the fold's float round-trip error.
        if d_current < COINCIDENT_TOLERANCE_M:
            break

        here = (f, j, i)
        neighbors = [
            neighbor("right", f, j, i),
            neighbor("left",  f, j, i),
            neighbor("down",  f, j, i),
            neighbor("up",    f, j, i),
        ]
        prev = (f_prev, j_prev, i_prev)
        # A neighbor equal to the current index is an unconnected wall (`build_neighbor_maps`
        # represents a wall as the point itself); never step onto it. Likewise never step
        # back to the point we came from.
        skip = {here, prev}

        # Step straight onto the endpoint (or its physical twin across a seam) as soon as a
        # neighbor coincides with it -- this also avoids the degenerate `deviation` evaluated
        # exactly at the endpoint.
        next_pt = None
        for nb in neighbors:
            if nb in skip:
                continue
            if distance_on_unit_sphere(
                coord(gridlon, *nb), coord(gridlat, *nb), lon2, lat2
            ) < COINCIDENT_TOLERANCE_M:
                next_pt = nb
                break

        if next_pt is None:
            # Admit a neighbor (other than the one we came from) if it gets us strictly
            # closer to the endpoint, OR if it is a seam twin of the current point -- the
            # same physical location reached by a different index across a periodic or fold
            # seam. The twin is a valid zero-length step that does *not* reduce the distance,
            # so a strict "closer than current" test admits it only depending on floating-point
            # rounding (the bug that made paths flip across numpy/libm versions); admitting it
            # explicitly by physical coincidence makes that deterministic, without admitting
            # genuinely-distinct near-equidistant neighbors (which would wander off a normal
            # path). Among the admitted neighbors take the one closest to the desired curve,
            # breaking near-ties (within WALK_DEVIATION_ATOL) deterministically by index.
            p_current = progress(coord(gridlon, f, j, i), coord(gridlat, f, j, i))
            cur_lon, cur_lat = coord(gridlon, f, j, i), coord(gridlat, f, j, i)
            cand = []
            for nb in neighbors:
                if nb in skip:
                    continue
                nb_lon, nb_lat = coord(gridlon, *nb), coord(gridlat, *nb)
                seam_twin = distance_on_unit_sphere(nb_lon, nb_lat, cur_lon, cur_lat) < COINCIDENT_TOLERANCE_M
                if (progress(nb_lon, nb_lat) < p_current) or seam_twin:
                    dev = deviation(nb_lon, nb_lat)
                    if np.isfinite(dev):
                        cand.append((dev, order_key(nb), nb))
            if cand:
                best_dev = min(c[0] for c in cand)
                next_pt = min(
                    (c for c in cand if c[0] <= best_dev + WALK_DEVIATION_ATOL),
                    key=lambda c: c[1],
                )[2]
            else:
                # No admissible forward move (e.g. closing a fold): take the closest
                # non-wall, non-previous neighbor, with a deterministic index tie-break.
                alt = [nb for nb in neighbors if nb not in skip] or [nb for nb in neighbors if nb != prev]
                next_pt = min(
                    alt,
                    key=lambda nb: (
                        distance_on_unit_sphere(coord(gridlon, *nb), coord(gridlat, *nb), lon2, lat2),
                        order_key(nb),
                    ),
                )

        f_prev, j_prev, i_prev = f, j, i

        f, j, i = next_pt

        i_c_seg.append(i)
        j_c_seg.append(j)
        f_c_seg.append(f)

        ct+=1

    # create lat/lon vectors from (f,j,i) triples
    lons_c_seg = []
    lats_c_seg = []
    for ff, jj, ji in zip(f_c_seg, j_c_seg, i_c_seg):
        lons_c_seg.append(coord(gridlon, ff, jj, ji))
        lats_c_seg.append(coord(gridlat, ff, jj, ji))

    i_c, j_c = np.array(i_c_seg), np.array(j_c_seg)
    lons_c, lats_c = np.array(lons_c_seg), np.array(lats_c_seg)
    if multitile:
        return i_c, j_c, np.array(f_c_seg), lons_c, lats_c
    return i_c, j_c, lons_c, lats_c


def find_closest_grid_point(lon, lat, gridlon, gridlat):
    """
    Find integer indices of closest grid point in grid of coordinates
    (gridlon, gridlat), for a given point (lon, at).

    PARAMETERS:
    -----------
        lon (float): longitude of point to find, in degrees
        lat (float): latitude of point to find, in degrees
        gridlon (numpy.ndarray): grid longitudes, in degrees
        gridlat (numpy.ndarray): grid latitudes, in degrees

    RETURNS:
    --------

    For 2d (single-tile) grids:
        iclose, jclose: integer grid indices for the geographical point of interest
    For 3d (multi-tile) grids, additionally the face index:
        iclose, jclose, fclose
    """

    if isinstance(gridlon, xr.core.dataarray.DataArray):
        gridlon = gridlon.values
    if isinstance(gridlat, xr.core.dataarray.DataArray):
        gridlat = gridlat.values
    dist = distance_on_unit_sphere(lon, lat, gridlon, gridlat)
    idx = np.unravel_index(np.nanargmin(dist), gridlon.shape)
    if gridlon.ndim == 3:
        fclose, jclose, iclose = idx
        return iclose, jclose, fclose
    jclose, iclose = idx
    return iclose, jclose

def distance_on_unit_sphere(lon1, lat1, lon2, lat2, R=6.371e6, method="vincenty"):
    """
    Calculate geodesic arc distance between points (lon1, lat1) and (lon2, lat2).

    PARAMETERS:
    -----------
        lon1 : float
            Start longitude(s), in degrees
        lat1 : float
            Start latitude(s), in degrees
        lon2 : float
            End longitude(s), in degrees
        lat2 : float
            End latitude(s), in degrees
        R : float
            Radius of sphere. Default: 6.371e6 (realistic Earth value). Set to 1 for
            arc distance in radius.
        method : str
            Name of method. Supported methods: ["vincenty", "haversine", "law of cosines"].
            Default: "vincenty", which is the most robust. Note, however, that it still can result in
            vanishingly small (but crucially non-zero) errors; such as that the distance between (0., 0.)
            and (360., 0.) is 1.e-16 meters when it should be identically zero.

    RETURNS:
    --------

    dist : float
        Geodesic distance between points (lon1, lat1) and (lon2, lat2).
    """
    
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.abs(phi2-phi1)
    
    lam1 = np.deg2rad(lon1)
    lam2 = np.deg2rad(lon2)
    dlam = np.abs(lam2-lam1)
    
    if method=="vincenty":
        numerator = np.sqrt(
            (np.cos(phi2)*np.sin(dlam))**2 +
            (np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam))**2
        )
        denominator = np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(dlam)
        arc = np.arctan2(numerator, denominator)
        
    elif method=="haversine":
        arc = 2*np.arcsin(np.sqrt(
            np.sin(dphi/2.)**2 + (1. - np.sin(dphi/2.)**2 - np.sin((phi1+phi2)/2.)**2)*np.sin(dlam/2.)**2
        ))
    
        
    elif method=="law of cosines":
        arc = np.arccos(
            np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(dlam)
        )

    return R * arc

def spherical_angle(lonA, latA, lonB, latB, lonC, latC):
    """
    Calculate the spherical triangle angle alpha between geodesic arcs AB and AC defined by
    [(lonA, latA), (lonB, latB)] and [(lonA, latA), (lonC, latC)], respectively.

    PARAMETERS:
    -----------
        lonA : float
            Longitude of point A, in degrees
        latA : float
            Latitude of point A, in degrees
        lonB : float
            Longitude of point B, in degrees
        latB : float
            Latitude of point B, in degrees
        lonC : float
            Longitude of point C, in degrees
        latC : float
            Latitude of point C, in degrees

    RETURNS:
    --------

    angle : float
        Spherical absolute value of triangle angle alpha, in radians. Returns 0 when B or
        C coincides with the vertex A (a degenerate, zero-length arc).
    """
    a = distance_on_unit_sphere(lonB, latB, lonC, latC, R=1.)
    b = distance_on_unit_sphere(lonC, latC, lonA, latA, R=1.)
    c = distance_on_unit_sphere(lonA, latA, lonB, latB, R=1.)

    # The spherical law of cosines divides by sin(b)*sin(c), where b is the arc A->C and
    # c is the arc A->B. When b == 0 (C coincides with the vertex A) or c == 0 (B coincides
    # with A), one arc is degenerate and the ratio is 0/0 -> NaN. The limiting angle is 0:
    # a point sitting on the vertex A lies on the arc, so its angular deviation is zero.
    # This arises in the walk's `deviation` metric when a candidate corner lands exactly on
    # a section endpoint (e.g. approaching the bipolar-fold seam); returning 0 avoids a
    # spurious divide-by-zero warning without changing any selection (such a candidate is
    # already handled by the endpoint-coincidence step / progress test).
    if b == 0. or c == 0.:
        return 0.

    return np.arccos(np.clip((np.cos(a) - np.cos(b)*np.cos(c))/(np.sin(b)*np.sin(c)), -1., 1.))

def align_coords(coords1, coords2, extend=False):
    """Align coords1 and coords2 by minimizing distance between coords[-1] and coords[0]

    Arguments
    ---------
    coords1 [list of (lon,lat) tuples]
    coords2 [list of (lon,lat) tuples]

    Keyword Arguments
    -----------------
    extend [bool (Default : False)] -- extends coords1 so that its starting point is 
        equal to the end point of coords2 and its end point is the starting point of
        coords2.

    Returns
    -------
    (coords1, coords2)

    Examples
    --------
    >>> coords1 = [(-100, 0), (-50, 0)]
    >>> coords2 = [(   0, 0), (-40, 0)]
    >>> sec.align_coords(coords1, coords2)
    
    """
    coords_options = [
        [coords1      , coords2      ],
        [coords1[::-1], coords2      ],
        [coords1      , coords2[::-1]],
        [coords1[::-1], coords2[::-1]]
    ]
    dists = np.array([
        coord_distance(c1[-1], c2[0])
        for (c1,c2) in coords_options
    ])
    coords1, coords2 = coords_options[np.argmin(dists)]
    if extend:
        coords1 = [coords2[-1]] + coords1 + [coords2[0]]
    return coords1, coords2

def coord_distance(coord1, coord2):
    """Spherical distance between coord1 and coord2"""
    return distance_on_unit_sphere(
        coord1[0],
        coord1[1],
        coord2[0],
        coord2[1]
    )

def lonlat_from_coords(coords):
    """Turns list of coordinate pairs into arrays of longitudes and latitudes"""
    return (
            np.array([lon for (lon, lat) in coords]),
            np.array([lat for (lon, lat) in coords])
        )

def coords_from_lonlat(lons, lats):
    """Turns iterable longitudes and latitudes into a list of coordinate pairs"""
    return [(lon, lat) for (lon, lat) in zip(lons, lats)]
