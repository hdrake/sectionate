import numpy as np
import xarray as xr

from .gridutils import get_geo_corners, check_symmetric

class Section():
    """A named hydrographic section"""
    def __init__(self, name, coords, children = {}, parents = {}):
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

        children [mapping from str to Section (default: {})] -- dictionary
            mapping the names of child sections to their Section instances.
            This attribute will generally be populated automatically from
            the function `join_sections`.

        parents [mapping from str to Section (default: {})] -- To do

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
                self.lons, self.lats = lonlat_from_coords(self.coords)
            else:
                raise ValueError("If coords is a list, its elements must be (lon,lat) 2-tuples")
        else:
            raise ValueError("coords must be a 2-tuple of lists/arrays or a list of 2-tuples")
            
        self.children = children.copy()
        self.parents = parents.copy()
        self.save = {}

    def reverse(self):
        """Reverse the section's direction"""
        self.update_coords(self.coords[::-1])
        return self

    def update_coords(self, coords):
        """Update coordinates (including longitude and latitude arrrays)"""
        self.coords = coords
        self.lons, self.lats = lonlat_from_coords(self.coords)

    def copy(self):
        """Create a deep copy of the Section instance"""
        section = Section(
            self.name,
            self.coords,
            children=self.children,
            parents=self.parents
        )
        section.save = self.save.copy()
        return section
    
    def __repr__(self):
        summary = f"Section({self.name}, {self.coords})"
        if len(self.children) > 0:
            child_list = "\n".join([f"  - {s}" for s in self.children.values()])
            summary += f"\n Children:\n{child_list}"
        return summary

def join_sections(name, *sections, **kwargs):
    if type(name) is not str:
        raise ValueError("first arugment (name) must be a str.")
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
        
def grid_section(grid, lons, lats, topology="latlon"):
    """
    Compute composite section along model `grid` velocity faces that approximates geodesic paths
    between consecutive points defined by (lons, lats).

    Parameters
    ----------
    grid: xgcm.Grid
        Object describing the geometry of the ocean model grid, including metadata about variable names for
        the staggered C-grid dimensions and c oordinates.
    lons: list or np.ndarray
        Longitudes, in degrees, of consecutive vertices defining a piece-wise geodesic section.
    lats: list or np.ndarray
        Latitudes, in degrees (in range [-90, 90]), of consecutive vertices defining a piece-wise geodesic section.
    topology: str
        Default: "latlon". Currently only supports the following options: ["latlon", "cartesian", "MOM-tripolar"].
        
    Returns
    -------
    isect, jsect, lonsect, latsect: `np.ndarray` of types (int, int, float, float) 
        (isect, jsect) correspond to indices of vorticity points that define velocity faces.
        (lonsect, latsect) are the corresponding longitude and latitudes.
    """
    geocorners = get_geo_corners(grid)
    return create_section_composite(
        geocorners["X"],
        geocorners["Y"],
        lons,
        lats,
        check_symmetric(grid),
        boundary={ax:grid.axes[ax]._boundary for ax in grid.axes},
        topology=topology
    )

def create_section_composite(
    gridlon,
    gridlat,
    lons,
    lats,
    symmetric,
    boundary={"X":"periodic", "Y":"extend"},
    topology="latlon"
    ):
    """
    Compute composite section along velocity faces, as defined by coordinates of vorticity points (gridlon, gridlat),
    that most closely approximates geodesic paths between consecutive points defined by (lons, lats).

    PARAMETERS:
    -----------

    gridlon: np.ndarray
        2d array of longitude (with dimensions ("Y", "X")), in degrees
    gridlat: np.ndarray
        2d array of latitude (with dimensions ("Y", "X")), in degrees
    lons: list of float
        longitude of section starting, intermediate and end points, in degrees
    lats: list of float
        latitude of section starting, intermediate and end points, in degrees
    symmetric: bool
        True if symmetric (vorticity on "outer" positions); False if non-symmetric (assuming "right" positions).
    boundary: dictionary mapping grid axis to boundary condition
        Default: {"X":"periodic", "Y":"extend"}. Set to {"X":"extend", "Y":"extend"} if using a non-periodic regional domain.
    topology: str
        Default: "latlon". Currently only supports the following options: ["latlon", "cartesian", "MOM-tripolar"].

    RETURNS:
    -------

    isect, jsect, lonsect, latsect: `np.ndarray` of types (int, int, float, float) 
        (isect, jsect) correspond to indices of vorticity points that define velocity faces.
        (lonsect, latsect) are the corresponding longitude and latitudes.
    """

    isect = np.array([], dtype=np.int64)
    jsect = np.array([], dtype=np.int64)
    lonsect = np.array([], dtype=np.float64)
    latsect = np.array([], dtype=np.float64)

    if len(lons) != len(lats):
        raise ValueError("lons and lats should have the same length")

    for k in range(len(lons) - 1):
        iseg, jseg, lonseg, latseg = create_section(
            gridlon,
            gridlat,
            lons[k],
            lats[k],
            lons[k + 1],
            lats[k + 1],
            symmetric,
            boundary=boundary,
            topology=topology
        )

        isect = np.concatenate([isect, iseg[:-1]], axis=0)
        jsect = np.concatenate([jsect, jseg[:-1]], axis=0)
        lonsect = np.concatenate([lonsect, lonseg[:-1]], axis=0)
        latsect = np.concatenate([latsect, latseg[:-1]], axis=0)
        
    isect = np.concatenate([isect, [iseg[-1]]], axis=0)
    jsect = np.concatenate([jsect, [jseg[-1]]], axis=0)
    lonsect = np.concatenate([lonsect, [lonseg[-1]]], axis=0)
    latsect = np.concatenate([latsect, [latseg[-1]]], axis=0)

    return isect.astype(np.int64), jsect.astype(np.int64), lonsect, latsect

def create_section(gridlon, gridlat, lonstart, latstart, lonend, latend, symmetric, boundary={"X":"periodic", "Y":"extend"}, topology="latlon"):
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
    symmetric: bool
        True if symmetric (vorticity on "outer" positions); False if non-symmetric (assuming "right" positions).
    boundary: dictionary mapping grid axis to boundary condition
        Default: {"X":"periodic", "Y":"extend"}. Set to {"X":"extend", "Y":"extend"} if using a non-periodic regional domain.
    topology: str
        Default: "latlon". Currently only supports the following options: ["latlon", "cartesian", "MOM-tripolar"].

    RETURNS:
    -------

    isect, jsect, lonsect, latsect: `np.ndarray` of types (int, int, float, float) 
        (isect, jsect) correspond to indices of vorticity points that define velocity faces.
        (lonsect, latsect) are the corresponding longitude and latitudes.
    """

    if symmetric and boundary["X"] == "periodic":
        gridlon=gridlon[:,:-1]
        gridlat=gridlat[:,:-1]

    iseg, jseg, lonseg, latseg = infer_grid_path_from_geo(
        lonstart,
        latstart,
        lonend,
        latend,
        gridlon,
        gridlat,
        boundary=boundary,
        topology=topology
    )
    return (
        iseg,
        jseg,
        lonseg,
        latseg
    )

def infer_grid_path_from_geo(lonstart, latstart, lonend, latend, gridlon, gridlat, boundary={"X":"periodic", "Y":"extend"}, topology="latlon"):
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
        2d array of longitude, in degrees
    gridlat: np.ndarray
        2d array of latitude, in degrees
    boundary: dictionary mapping grid axis to boundary condition
        Default: {"X":"periodic", "Y":"extend"}. Set to {"X":"extend", "Y":"extend"} if using a non-periodic regional domain.
    topology: str
        Default: "latlon". Currently only supports the following options: ["latlon", "cartesian", "MOM-tripolar"].

    RETURNS:
    -------

    isect, jsect, lonsect, latsect: `np.ndarray` of types (int, int, float, float) 
        (isect, jsect) correspond to indices of vorticity points that define velocity faces.
        (lonsect, latsect) are the corresponding longitude and latitudes.
    """

    istart, jstart = find_closest_grid_point(
        lonstart,
        latstart,
        gridlon,
        gridlat
    )
    iend, jend = find_closest_grid_point(
        lonend,
        latend,
        gridlon,
        gridlat
    )
    iseg, jseg, lonseg, latseg = infer_grid_path(
        istart,
        jstart,
        iend,
        jend,
        gridlon,
        gridlat,
        boundary=boundary,
        topology=topology
    )

    return iseg, jseg, lonseg, latseg


def infer_grid_path(i1, j1, i2, j2, gridlon, gridlat, boundary={"X":"periodic", "Y":"extend"}, topology="latlon"):
    """
    Find the grid indices (and coordinates) of vorticity points that most closely approximate
    the geodesic path between points (gridlon[j1,i1], gridlat[j1,i1]) and
    (gridlon[j2,i2], gridlat[j2,i2]).

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
        2d array of longitude, in degrees
    gridlat: np.ndarray
        2d array of latitude, in degrees
    boundary: dictionary mapping grid axis to boundary condition
        Default: {"X":"periodic", "Y":"extend"}. Set to {"X":"extend", "Y":"extend"} if using a non-periodic regional domain.
    topology: str
        Default: "latlon". Currently only supports the following options: ["latlon", "cartesian", "MOM-tripolar"].

    RETURNS:
    -------

    iseg, jseg: list of int
        list of (i,j) pairs bounded by (i1, j1) and (i2, j2)
    lonseg, latseg: list of float
        corresponding longitude and latitude for iseg, jseg
    """
    ny, nx = gridlon.shape
    
    if isinstance(gridlon, xr.core.dataarray.DataArray):
        gridlon = gridlon.values
    if isinstance(gridlat, xr.core.dataarray.DataArray):
        gridlat = gridlat.values

    # target coordinates
    lon1, lat1 = gridlon[j1, i1], gridlat[j1, i1]
    lon2, lat2 = gridlon[j2, i2], gridlat[j2, i2]
    
    # init loop index to starting position
    i = i1
    j = j1

    iseg = [i]  # add first point to list of points
    jseg = [j]  # add first point to list of points

    # iterate through the grid path steps until we reach end of section
    ct = 0 # grid path step counter

    # Grid-agnostic algorithm:
    # First, find all four neighbors (subject to grid topology)
    # Second, throw away any that are further from the destination than the current point
    # Third, go to the valid neighbor that has the smallest angle from the arc path between the
    # start and end points (the shortest geodesic path)
    j_prev, i_prev = j,i
    while (i%nx != i2) or (j != j2):
                
        # safety precaution: exit after taking enough steps to have crossed the entire model grid
        if ct > (nx+ny+1):
            raise RuntimeError(f"Should have reached the endpoint by now.")

        d_current = distance_on_unit_sphere(
                gridlon[j,i],
                gridlat[j,i],
                lon2,
                lat2
            )
        
        if d_current < 1.e-12:
            break
        
        if boundary["X"] == "periodic":
            right = (j, (i+1)%nx)
            left = (j, (i-1)%nx)
        else:
            right = (j, np.clip(i+1, 0, nx-1))
            left = (j, np.clip(i-1, 0, nx-1))
        down = (np.clip(j-1, 0, ny-1), i)
        
        if topology=="MOM-tripolar":
            if j!=ny-1:
                up = (j+1, i%nx)
            else:
                up = (j-1, (nx-1) - (i%nx))
                
        elif topology=="cartesian" or topology=="latlon":
                up = (np.clip(j+1, 0, ny-1), i)
        else:
            raise ValueError("Only 'cartesian', 'latlon', and 'MOM-tripolar' grid topologies are currently supported.")
        
        neighbors = [right, left, down, up]

        j_next, i_next = None, None
        smallest_angle = np.inf
        d_list = []
        for (_j, _i) in neighbors:
            d = distance_on_unit_sphere(
                gridlon[_j,_i],
                gridlat[_j,_i],
                lon2,
                lat2
            )
            d_list.append(d/d_current)
            if d < d_current:
                if d==0.: # We're done!
                    j_next, i_next = _j, _i
                    smallest_angle = 0.
                    break
                # Instead of simply moving to the point that gets us closest to the target,
                # a more robust approach is to pick, among the points that do get us closer,
                # the one that most closely follows the great circle between the start and
                # end points of the section. We average the angles relative to both end
                # points so that the shortest path is unique and insensitive to which direction
                # the section is traveled.
                else:
                    angle1 = spherical_angle(
                        lon2,
                        lat2,
                        lon1,
                        lat1,
                        gridlon[_j,_i],
                        gridlat[_j,_i],
                    )
                    angle2 = spherical_angle(
                        lon1,
                        lat1,
                        lon2,
                        lat2,
                        gridlon[_j,_i],
                        gridlat[_j,_i],
                    )
                    angle = (angle1+angle2)/2.
                    if angle < smallest_angle:
                        j_next, i_next = _j, _i
                        smallest_angle = angle
        
        # There can be some strange edge cases in which none of the neighboring points
        # actually get us closer to the target (e.g. when closing folds in the grid).
        # In these cases, simply pick the adjacent point that gets us closest, as long as
        # it was not our previous point (to avoid endless loops). This algorithm should be
        # guaranteed to always get us to the target point.
        if (smallest_angle == np.inf) or (j_next, i_next) == (j_prev, i_prev):
            if (j_prev, i_prev) in neighbors:
                idx = neighbors.index((j_prev, i_prev))
                del neighbors[idx]
                del d_list[idx]
            
            (j_next, i_next) = neighbors[np.argmin(d_list)]

        j_prev, i_prev = j,i
        
        j = j_next
        i = i_next

        iseg.append(i)
        jseg.append(j)
        
        ct+=1

    # create lat/lon vectors from i,j pairs
    lonseg = []
    latseg = []
    for jj, ji in zip(jseg, iseg):
        lonseg.append(gridlon[jj, ji])
        latseg.append(gridlat[jj, ji])
    return np.array(iseg), np.array(jseg), np.array(lonseg), np.array(latseg)


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

    iclose, jclose: integer
        grid indices for geographical point of interest
    """

    if isinstance(gridlon, xr.core.dataarray.DataArray):
        gridlon = gridlon.values
    if isinstance(gridlat, xr.core.dataarray.DataArray):
        gridlat = gridlat.values
    dist = distance_on_unit_sphere(lon, lat, gridlon, gridlat)
    jclose, iclose = np.unravel_index(np.nanargmin(dist), gridlon.shape)
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
        Spherical absolute value of triangle angle alpha, in radians.
    """
    a = distance_on_unit_sphere(lonB, latB, lonC, latC, R=1.)
    b = distance_on_unit_sphere(lonC, latC, lonA, latA, R=1.)
    c = distance_on_unit_sphere(lonA, latA, lonB, latB, R=1.)
        
    return np.arccos(np.clip((np.cos(a) - np.cos(b)*np.cos(c))/(np.sin(b)*np.sin(c)), -1., 1.))

def align_coords(coords1, coords2, extend=False):
    """Align coords1 and coords2 by minimizing distance between coords[-1] and coords[0]

    Arguments
    ---------
    coords1 [list of (lon,lat) tuples] -- 
    coords2 [list of (lon,lat) tuples] --

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
