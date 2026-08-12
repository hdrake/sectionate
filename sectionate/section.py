import numpy as np
import xarray as xr

from .gridutils import get_geo_corners, get_facedim

# Two corner indices map to the same physical point on seams that fold Tor wrap (e.g.
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
    from .topology import corner_topology
    from .walk import find_closest_corner, infer_grid_path, native_path

    facedim = get_facedim(grid)
    if facedim is not None:
        _check_supported_topology(grid)

    # Every topology a structured grid can declare -- a periodic wrap, a wall, a
    # bipolar fold, a multi-tile seam however it is rotated -- becomes an ordinary
    # edge of the grid's physical corner graph, so the walk below needs no case for
    # any of them, and never has to ask whether two indices are the same corner.
    topology = corner_topology(grid)

    if len(lons) != len(lats):
        raise ValueError("lons and lats should have the same length")

    nodes = []
    for k in range(len(lons) - 1):
        segment_curve = _check_segment_span(
            lons[k], lats[k], lons[k + 1], lats[k + 1], curve
        )
        n1 = find_closest_corner(lons[k], lats[k], topology)
        n2 = find_closest_corner(lons[k + 1], lats[k + 1], topology)
        seg = infer_grid_path(n1, n2, topology, curve=segment_curve)
        nodes.extend(seg[:-1] if k < len(lons) - 2 else seg)

    corners, _ = native_path(topology, nodes)
    f_c, j_c, i_c = corners[:, 0], corners[:, 1], corners[:, 2]

    # Report each corner's coordinate as its *emitted* representation stores it, not
    # as the node's canonical one. The two are the same physical point, but a
    # periodic seam's two spellings differ by a turn of longitude, and it is the
    # emitted one that keeps the reported path continuous -- 300, 360 rather than
    # 300, 0 -- which is what anything measuring a step along it depends on.
    geo = get_geo_corners(grid)
    glon = geo["X"].values
    glat = geo["Y"].values
    if facedim is None:
        lons_c, lats_c = glon[j_c, i_c], glat[j_c, i_c]
    else:
        lons_c, lats_c = glon[f_c, j_c, i_c], glat[f_c, j_c, i_c]

    if facedim is not None:
        return i_c, j_c, f_c, lons_c, lats_c
    return i_c, j_c, lons_c, lats_c


def _check_supported_topology(grid):
    """
    Raise if the multi-tile `grid` describes a topology sectionate cannot trace a section on.

    A section is traced by walking a graph of corner points, and a step across a tile seam is
    recognised by its two ends having different face indices (see `transports._uv_for_edge`).
    
    A face glued to *itself* therefore cannot be traced: both sides of such a seam carry the
    same face index, so a crossing is indistinguishable from an ordinary step within the face,
    and the seam's velocity is read from the wrong side -- silently, since nothing about the
    indices looks out of place. Topologies of that shape belong on the axis rather than in
    `face_connections`: a zonally periodic axis as ``padding="periodic"``, a bipolar/tripolar
    north fold as ``padding={"Y": {"fold": ...}}`` on a single-tile grid. sectionate handles
    both of those natively.

    A grid can still prove unsupported once its corner graph is actually built -- a corner that
    ends up with more than four neighbours, or a staggered grid whose seam corners are stored on
    no face, for instance. Those are rejected there, each with its own reason; see
    `gridutils.build_neighbor_maps` and `gridutils._OuterTopology`.
    """
    facedim = grid._facedim
    connections = (getattr(grid, "_face_connections", None) or {}).get(facedim, {})
    for face, axis_sides in connections.items():
        for axis, sides in axis_sides.items():
            for side in sides:
                if side is None:
                    continue
                if side[0] == face:
                    raise NotImplementedError(
                        f"Face {face} of this grid's `face_connections` is glued to itself "
                        f"(along its own '{axis}' axis). sectionate tells a seam crossing from "
                        "an ordinary step by the face index changing, so it cannot trace a face "
                        "joined to itself. Express that topology on the axis instead, on a "
                        "single-tile grid: padding='periodic' for a periodic axis, or "
                        "padding={'Y': {'fold': 'corner'}} for a bipolar/tripolar north fold."
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
