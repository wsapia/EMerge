# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.
from __future__ import annotations
import numpy as np
from ..cs import CoordinateSystem, GCS, Axis, _parse_axis
from ..geometry import GeoVolume, GeoPolygon, GeoEdge, GeoSurface
from .shapes import Alignment
import gmsh
from typing import Generator, Callable
from ..selection import FaceSelection
from typing import Literal
from functools import reduce
from loguru import logger


def _discretize_curve(xfunc: Callable, yfunc: Callable, 
                      t0: float, t1: float, xmin: float, tol: float=1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Computes a discreteized curve in X/Y coordinates based on the input parametric coordinates.

    Args:
        xfunc (Callable): The X-coordinate function fx(t)
        yfunc (Callable): The Y-coordinate function fy(t)
        t0 (float): The minimum value for the t-prameter
        t1 (float): The maximum value for the t-parameter
        xmin (float): The minimum distance for subsequent points
        tol (float, optional): The curve matching tolerance. Defaults to 1e-4.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    from ..mth.optimized import _subsample_coordinates
    
    td = np.linspace(t0, t1, 10001)
    xs = xfunc(td)
    ys = yfunc(td)
    XS, YS = _subsample_coordinates(xs, ys, tol, xmin)
    return XS, YS

def rotate_point(point: tuple[float, float, float],
                 axis: tuple[float, float, float],
                 ang: float,
                 origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 degrees: bool = False) -> tuple[float, float, float]:
    """
    Rotate a 3‑D point around an arbitrary axis that passes through `origin`.

    Parameters
    ----------
    point   : (x, y, z) coordinate of the point to rotate.
    axis    : (ux, uy, uz) direction vector of the rotation axis (need not be unit length).
    ang     : rotation angle.  Positive values follow the right‑hand rule.
    origin  : (ox, oy, oz) point through which the axis passes.  Defaults to global origin.
    degrees : If True, `ang` is interpreted in degrees; otherwise in radians.

    Returns
    -------
    (x,y,z) : tuple with the rotated coordinates.
    """
    # Convert inputs to numpy arrays
    p = np.asarray(point, dtype=float)
    o = np.asarray(origin, dtype=float)
    u = np.asarray(axis, dtype=float)

    # Shift so the axis passes through the global origin
    p_shifted = p - o

    # Normalise the axis direction
    norm = np.linalg.norm(u)
    if norm == 0:
        raise ValueError("Axis direction vector must be non‑zero.")
    u = u / norm

    # Convert angle to radians if necessary
    if degrees:
        ang = np.radians(ang)

    # Rodrigues’ rotation formula components
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    cross = np.cross(u, p_shifted)
    dot = np.dot(u, p_shifted)

    rotated = (p_shifted * cos_a
               + cross * sin_a
               + u * dot * (1 - cos_a))

    # Shift back to original reference frame
    rotated += o
    return tuple(rotated)

def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**(0.5)

def _pair_points(x1: np.ndarray,
                 y1: np.ndarray,
                 x2: np.ndarray,
                 y2: np.ndarray) -> list[tuple[int,int]]:
    if x1.shape[0]==x2.shape[0]:
        return [(i,i) for i in range(x1.shape[0])]

    #create point tuples
    p1s = [(x,y) for x,y in zip(x1, y1)]
    p2s = [(x,y) for x,y in zip(x2, y2)]
    
    pairs = []
    for i, p1 in enumerate(p1s):
        d1 = _dist(p1, p2s[i-1])
        d2 = _dist(p1, p2s[i])
        d3 = _dist(p1, p2s[i+1])
        mind = min([d1, d2, d3])
        if mind==d1:
            pairs.append((i,i-1))
            continue
        elif mind==d2:
            pairs.append((i,i))
        else:
            pairs.append((i,i+1))
    return pairs
            
def orthonormalize(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a set of orthonormal vectors given an input vector X

    Args:
        axis (np.ndarray): The X-axis

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The X, Y and Z axis (orthonormal)
    """
    Xaxis = axis/np.linalg.norm(axis)
    V = np.array([0,1,0])
    if 1-np.abs(np.dot(Xaxis, V)) < 1e-12:
        V = np.array([0,0,1])
    Yaxis = np.cross(Xaxis, V)
    Yaxis = np.abs(Yaxis/np.linalg.norm(Yaxis))
    Zaxis = np.cross(Xaxis, Yaxis)
    Zaxis = np.abs(Zaxis/np.linalg.norm(Zaxis))
    return Xaxis, Yaxis, Zaxis


class GeoPrism(GeoVolume):
    """The GepPrism class generalizes the GeoVolume for extruded convex polygons.
    Besides having a volumetric definitions, the class offers a .front_face 
    and .back_face property that selects the respective faces.

    Args:
        GeoVolume (_type_): _description_
    """
    def __init__(self,
                 volume_tag: int,
                 front_tag: int | None = None,
                 side_tags: list[int] | None = None,
                 _axis: Axis | None = None,
                 name: str | None = None):
        super().__init__(volume_tag, name=name)
        
        
        
        if front_tag is not None and side_tags is not None:
            self.front_tag: int = front_tag
            self.back_tag: int = None

            gmsh.model.occ.synchronize()
            self._add_face_pointer('back', tag=self.front_tag)

            tags = gmsh.model.get_boundary(self.dimtags, oriented=False)
            
            for dim, tag in tags:
                if (dim,tag) in side_tags:
                    continue
                self._add_face_pointer('front',tag=tag)
                self.back_tag = tag
                break

            self.side_tags: list[int] = [dt[1] for dt in tags if dt[1]!=self.front_tag and dt[1]!=self.back_tag]

            for tag in self.side_tags:
    
                self._add_face_pointer(f'side{tag}', tag=tag)
                self.back_tag = tag
                
        elif _axis is not None:
            _axis = _parse_axis(_axis)
            gmsh.model.occ.synchronize()
            tags = gmsh.model.get_boundary(self.dimtags, oriented=False)
            faces = []
            for dim, tag in tags:
                o1 = np.array(gmsh.model.occ.get_center_of_mass(2, tag))
                n1 = np.array(gmsh.model.get_normal(tag, (0,0)))
                if abs(np.sum(n1*_axis.np)) > 0.99:
                    dax = sum(o1 * _axis.np)
                    faces.append((o1, n1, dax, tag))
            
            faces = sorted(faces, key=lambda x: x[2])
            ftags = []
            if len(faces) >= 2:
                ftags.append(faces[0][3])
                ftags.append(faces[-1][3])
                self._add_face_pointer('front',faces[0][0], faces[0][1])
                self._add_face_pointer('back', faces[-1][0], faces[-1][1])
            elif len(faces)==1:
                ftags.append(faces[0][3])
                self._add_face_pointer('cap',faces[0][0], faces[0][1])
            
            ictr = 1
            for dim, tag in tags:
                if tag in ftags:
                    continue
                self._add_face_pointer(f'side{ictr}', tag=tag)
                ictr += 1

    def outside(self, *exclude: Literal['front','back']) -> FaceSelection:
        """Select all outside faces except for the once specified by outside

        Returns:
            FaceSelection: The resultant face selection
        """
        tagslist = [self._face_tags(name) for name in  self._face_pointers.keys() if name not in exclude]
        
        tags = list(reduce(lambda a,b: a+b, tagslist))
        return FaceSelection(tags)
    
    @property
    def front(self) -> FaceSelection:
        """The first local -Z face of the prism."""
        return self.face('front')
    
    @property
    def back(self) -> FaceSelection:
        """The back local +Z face of the prism."""
        return self.face('back')
    
    @property
    def sides(self) -> FaceSelection:
        """The outside faces excluding the top and bottom."""
        return self.boundary(exclude=('front','back'))

class XYPolygon:
    """This class generalizes a polygon in an un-embedded XY space that can be embedded in 3D space.
    """
    def __init__(self, 
                 xs: np.ndarray | list | tuple | None = None,
                 ys: np.ndarray | list | tuple | None = None,
                 cs: CoordinateSystem | None = None,
                 resolution: float = 1e-6):
        """Constructs an XY-plane placed polygon.

        Args:
            xs (np.ndarray): The X-points
            ys (np.ndarray): The Y-points
        """
        if xs is None:
            xs = []
        if ys is None:
            ys = []

        self.x: np.ndarray = np.asarray(xs)
        self.y: np.ndarray = np.asarray(ys)

        self.fillets: list[tuple[float, int]] = []
        
        self._cs: CoordinateSystem = cs
        self.resolution: float = resolution

    @property
    def center(self) -> tuple[float, float]:
        return np.mean(self.x), np.mean(self.y)
    
    @property
    def length(self):
        return sum([((self.x[i2]-self.x[i1])**2 + (self.y[i2]-self.y[i1])**2)**0.5 for i1, i2 in zip(range(self.N-1),range(1, self.N))])
        
    @property
    def N(self) -> int:
        """The number of polygon points

        Returns:
            int: The number of points
        """
        return len(self.x)
    
    def _check(self) -> None:
        """Checks if the last point is the same as the first point.
        The XYPolygon does not store redundant points p[0]==p[N] so if these are
        the same, this function will remove the last point.
        """
        if np.sqrt((self.x[-1]-self.x[0])**2 + (self.y[-1]-self.y[0])**2) < 1e-9:
            self.x = self.x[:-1]
            self.y = self.y[:-1]
        
    @property
    def area(self) -> float:
        """The Area of the polygon

        Returns:
            float: The area in square meters
        """
        return 0.5*np.abs(np.dot(self.x,np.roll(self.y,1))-np.dot(self.y,np.roll(self.x,1)))

    def incs(self, cs: CoordinateSystem) -> XYPolygon:
        self._cs = cs
        return self

    def extend(self, xpts: list[float], ypts: list[float]) -> XYPolygon:
        """Adds a series for x and y coordinates to the existing polygon.

        Args:
            xpts (list[float]): The list of x-coordinates
            ypts (list[float]): The list of y-coordinates

        Returns:
            XYPolygon: The same XYpolygon object
        """
        self.x = np.hstack([self.x, np.array(xpts)])
        self.y = np.hstack([self.y, np.array(ypts)])
        return self
    
    def iterate(self) -> Generator[tuple[float, float],None, None]:
        """ Iterates over the x,y coordinates as a tuple."""
        for i in range(self.N):
            yield (self.x[i], self.y[i])

    def fillet(self, radius: float, *indices: int) -> XYPolygon:
        """Add a fillet rounding with a given radius to the provided nodes.

        Example:
         >>> my_polygon.fillet(0.05, 2, 3, 4, 6)

        Args:
            radius (float): The radius
            *indices (int): The indices for which to apply the fillet.
        """
        for i in indices:
            self.fillets.append((radius, i))
        return self

    def _cleanup(self, resolution: float | None = None) -> None:
        # Compute differences between consecutive points
        if resolution is None:
            resolution = self.resolution
        dx = np.diff(self.x)
        dy = np.diff(self.y)

        # Distances between consecutive points
        dist = np.sqrt(dx**2 + dy**2)

        # Keep the first point, then points where distance >= threshold
        keep = np.insert(dist >= resolution, 1, True)

        # Apply mask
        self.x = self.x[keep]
        self.y = self.y[keep]
        
    def _make_wire(self, cs: CoordinateSystem) -> tuple[list[int], list[int], int]:
        """Turns the XYPolygon object into a GeoPolygon that is embedded in 3D space.

        The polygon will be placed in the XY-plane of the provided coordinate center.

        Args:
            cs (CoordinateSystem, optional): The coordinate system in which to put the polygon. Defaults to None.

        Returns:
            GeoPolygon: The resultant 3D GeoPolygon object.
        """
        self._check()

        ptags = []
        
        xg, yg, zg = cs.in_global_cs(self.x, self.y, 0*self.x)

        points = dict()
        for x,y,z in zip(xg, yg, zg):
            reuse = False
            for key, (px, py, pz) in points.items():
                if ((x-px)**2 + (y-py)**2 + (z-pz)**2)**0.5 < 1e-12:
                    ptags.append(key)
                    reuse = True
                    break
            if reuse:
                logger.warning(f'Reusing {ptags[-1]}')
                continue
            ptag = gmsh.model.occ.add_point(x,y,z)
            points[ptag] = (x,y,z)
            ptags.append(ptag)
        
        lines = []
        for i1, p1 in enumerate(ptags):
            p2 = ptags[(i1+1) % len(ptags)]
            lines.append(gmsh.model.occ.add_line(p1, p2))
        
        add = 0
        for radius, index in self.fillets:
            t1 = lines[(index + add-1) % len(lines)]
            t2 = lines[index + add]
            tag = gmsh.model.occ.fillet2_d(t1, t2, radius)
            lines.insert(index, tag)
            add += 1

        wiretag = gmsh.model.occ.add_wire(lines)
        return ptags, lines, wiretag
        
    def _finalize(self, cs: CoordinateSystem, name: str | None = 'GeoPolygon') -> GeoPolygon:
        """Turns the XYPolygon object into a GeoPolygon that is embedded in 3D space.

        The polygon will be placed in the XY-plane of the provided coordinate center.

        Args:
            cs (CoordinateSystem, optional): The coordinate system in which to put the polygon. Defaults to None.

        Returns:
            GeoPolygon: The resultant 3D GeoPolygon object.
        """
        self._cleanup()
        ptags, lines, wiretag = self._make_wire(cs)
        surftag = gmsh.model.occ.add_plane_surface([wiretag,])
        gmsh.model.occ.remove([(1,wiretag),]+[(1,t) for t in lines], recursive=True)
        poly = GeoPolygon([surftag,], name=name)
        poly.points = ptags
        poly.lines = lines
        return poly
    
    def extrude(self, length: float, cs: CoordinateSystem | None = None, name: str = 'Extrusion') -> GeoPrism:
        """Extrues the polygon along the Z-axis.
        The z-coordinates go from z1 to z2 (in meters). Then the extrusion
        is either provided by a maximum dz distance (in meters) or a number
        of sections N.

        Args:
            length (length): The length of the extrusion.

        Returns:
            GeoVolume: The resultant Volumetric object.
        """
        if cs is None:
            cs = GCS
        poly_fin = self._finalize(cs)
        zax = length*cs.zax.np
        poly_fin._exists = False
        volume = gmsh.model.occ.extrude(poly_fin.dimtags, zax[0], zax[1], zax[2])
        tags = [t for d,t in volume if d==3]
        surftags = [t for d,t in volume if d==2]
        return GeoPrism(tags, surftags[0], surftags, name=name)
    
    def geo(self, cs: CoordinateSystem | None = None, name: str = 'GeoPolygon') -> GeoPolygon:
        """Returns a GeoPolygon object for the current polygon.

        Args:
            cs (CoordinateSystem, optional): The Coordinate system of which the XY plane will be used. Defaults to None.

        Returns:
            GeoPolygon: The resultant object.
        """
        if self._cs is not None:
            return self._finalize(self._cs)
        if cs is None:
            cs = GCS
        return self._finalize(cs) 
    
    def revolve(self, cs: CoordinateSystem, origin: tuple[float, float, float], axis: tuple[float, float,float], angle: float = 360.0, name: str = 'Revolution') -> GeoPrism:
        """Applies a revolution to the XYPolygon along the provided rotation ais

        Args:
            cs (CoordinateSystem, optional): _description_. Defaults to None.
            angle (float, optional): _description_. Defaults to 360.0.

        Returns:
            Prism: The resultant 
        """
        if cs is None:
            cs = GCS
        poly_fin = self._finalize(cs)
        
        x,y,z = origin
        ax, ay, az = axis
        
        volume = gmsh.model.occ.revolve(poly_fin.dimtags, x,y,z, ax, ay, az, angle*np.pi/180)
        
        tags = [t for d,t in volume if d==3]
        poly_fin.remove()
        return GeoPrism(tags, _axis=axis, name=name)

    @staticmethod
    def circle(radius: float, 
               dsmax: float | None= None,
               tolerance: float | None = None,
               Nsections: int | None = None):
        """This method generates a segmented circle.

        The number of points along the circumpherence can be specified in 3 ways. By a maximum
        circumpherential length (dsmax), by a radial tolerance (tolerance) or by a number of 
        sections (Nsections).

        Args:
            radius (float): The circle radius
            dsmax (float, optional): The maximum circumpherential angle. Defaults to None.
            tolerance (float, optional): The maximum radial error. Defaults to None.
            Nsections (int, optional): The number of sections. Defaults to None.

        Returns:
            XYPolygon: The XYPolygon object.
        """
        if Nsections is not None:
            N = Nsections+1
        elif dsmax is not None:
            N = int(np.ceil((2*np.pi*radius)/dsmax))
        elif tolerance is not None:
            N = int(np.ceil(2*np.pi/np.arccos(1-tolerance)))

        angs = np.linspace(0,2*np.pi,N)

        xs = radius*np.cos(angs[:-1])
        ys = radius*np.sin(angs[:-1])
        return XYPolygon(xs, ys)

    @staticmethod
    def rect(width: float,
             height: float,
             origin: tuple[float, float],
             alignment: Alignment = Alignment.CORNER) -> XYPolygon:
        """Create a rectangle in the XY-plane as polygon

        Args:
            width (float): The width (X)
            height (float): The height (Y)
            origin (tuple[float, float]): The origin (x,y)
            alignment (Alignment, optional): What point the origin describes. Defaults to Alignment.CORNER.

        Returns:
            XYPolygon: A new XYpolygon object
        """
        if alignment is Alignment.CORNER:
            x0, y0 = origin
        else:
            x0 = origin[0]-width/2
            y0 = origin[1]-height/2
        xs = np.array([x0, x0, x0 + width, x0+width])
        ys = np.array([y0, y0+height, y0+height, y0])
        return XYPolygon(xs, ys)
    
    def parametric(self, xfunc: Callable,
                   yfunc: Callable,
                   xmin: float = 1e-3,
                   tolerance: float = 1e-5,
                   tmin: float = 0,
                   tmax: float = 1,
                   reverse: bool = False) -> XYPolygon:
        """Adds the points of a parametric curve to the polygon.
        The parametric curve is defined by two parametric functions of a parameter t that (by default) lives in the interval from [0,1].
        thus the curve x(t) = xfunc(t), and y(t) = yfunc(t).

        The tolerance indicates a maximum deviation from the true path.

        Args:
            xfunc (Callable): The x-coordinate function.
            yfunc (Callable): The y-coordinate function
            tolerance (float): A maximum distance tolerance. Defaults to 10um.
            tmin (float, optional): The start value of the t-parameter. Defaults to 0.
            tmax (float, optional): The end value of the t-parameter. Defaults to 1.
            reverse (bool, optional): Reverses the curve.

        Returns:
            XYPolygon: _description_
        """
        xs, ys = _discretize_curve(xfunc, yfunc, tmin, tmax, xmin, tolerance)

        if reverse:
            xs = xs[::-1]
            ys = ys[::-1]
        self.extend(xs, ys)
        return self
    
    def connect(self, other: XYPolygon, name: str = 'Connection') -> GeoVolume:
        """Connect two XYPolygons with a defined coordinate system

        The coordinate system must be defined before this function can be used. To add a coordinate systme without
        rendering the Polygon to a GeoVolume, use:
        >>> polygon.incs(my_cs_obj)
        
        Args:
            other (XYPolygon): _descrThe otheiption_

        Returns:
            GeoVolume: The resultant volume object
        """
        if self._cs is None:
            raise RuntimeError('Cannot connect XYPolygons without a defined coordinate system. Set this first using .incs()')
        if other._cs is None:
            raise RuntimeError('Cannot connect XYPolygons without a defined coordinate system. Set this first using .incs()')
        p1, l1, w1 = self._make_wire(self._cs)
        p2, l2, w2 = other._make_wire(other._cs)
        o1 = np.array(self._cs.in_global_cs(*self.center, 0)).flatten()
        o2 = np.array(other._cs.in_global_cs(*other.center, 0)).flatten()
        dts = gmsh.model.occ.addThruSections([w1, w2], True, parametrization="IsoParametric")
        vol = GeoVolume([t for d,t in dts if d==3], name=name)
        
        vol._add_face_pointer('front',o1, self._cs.zax.np)
        vol._add_face_pointer('back', o2, other._cs.zax.np)
        return vol
            
class Disc(GeoSurface):
    _default_name: str = 'Disc'
    
    def __init__(self, origin: tuple[float, float, float],
                 radius: float,
                 axis: tuple[float, float, float] = (0,0,1.0),
                 radius_opt: float | None = None,
                 axis_opt: tuple[float, float, float] | None = None,
                 name: str | None = None):
        """Creates a circular Disc surface.

        Args:
            origin (tuple[float, float, float]): The center of the disc
            radius (float): The radius of the disc
            axis (tuple[float, float, float], optional): The disc normal axis. Defaults to (0,0,1.0).
            radius_opt (float, None): Secondary radius in case where one wants to make an ellipse.
        """
        if radius_opt is None:
            radius_opt = radius
            axis_opt = []
        else:
            if axis_opt is None:
                raise ValueError('A secondary axis is required when making an ellipse')
        
        disc = gmsh.model.occ.addDisk(*origin, radius, radius_opt, zAxis=axis, xAxis=axis_opt)
        super().__init__(disc, name=name)
    
    
class Curve(GeoEdge):
    _default_name: str = 'Curve'
    
    def __init__(self, 
                 xpts: np.ndarray, 
                 ypts: np.ndarray, 
                 zpts: np.ndarray, 
                 degree: int = 3,
                 weights: list[float] | None = None,
                 knots: list[float] | None = None,
                 ctype: Literal['Spline','BSpline','Bezier'] = 'Spline',
                 name: str | None = None):
        """Generate a Spline/Bspline or Bezier curve based on a series of points

        This calls the different curve features in OpenCASCADE.
        
        The dstart parameter defines the departure direction of the curve. If not provided this is inferred as the
        discrete derivative from the first to second coordinate. 
        
        Args:
            xpts (np.ndarray): The X-coordinates
            ypts (np.ndarray): The Y-coordinates
            zpts (np.ndarray): The Z-coordinates
            degree (int, optional): The BSpline degree parameter. Defaults to 3.
            weights (list[float] | None, optional): An optional point weights list. Defaults to None.
            knots (list[float] | None, optional): A nkots list. Defaults to None.
            ctype (Literal['Spline','BSpline','Bezier'], optional): The type of curve. Defaults to 'Spline'.
            dstart (tuple[float, float, float] | None, optional): The departure direction. Defaults to None.
        """
        self.xpts: np.ndarray = xpts
        self.ypts: np.ndarray = ypts
        self.zpts: np.ndarray = zpts

        points = [gmsh.model.occ.add_point(x,y,z) for x,y,z in zip(xpts, ypts, zpts)]
        
        if ctype.lower()=='spline':
            tags = gmsh.model.occ.addSpline(points)
            
        elif ctype.lower()=='bspline':
            if weights is None:
                weights = []
            if knots is None:
                knots = []
            tags = gmsh.model.occ.addBSpline(points, degree=degree, weights=weights, knots=knots)
        else:
            tags = gmsh.model.occ.addBezier(points)
        
        tags = gmsh.model.occ.addWire([tags,])
        gmsh.model.occ.remove([(0,tag) for tag in points])
        super().__init__(tags, name=name)
    
        gmsh.model.occ.synchronize()
        p1 = gmsh.model.getValue(self.dim, self.tags[0], [0,])
        p2 = gmsh.model.getValue(self.dim, self.tags[0], [1e-6])
        self.dstart: tuple[float, float, float] = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
    
        
    @property
    def p0(self) -> tuple[float, float, float]:
        """The start coordinate
        """
        return (self.xpts[0], self.ypts[0], self.zpts[0])

    @staticmethod
    def helix_rh(pstart: tuple[float, float, float],
              pend: tuple[float, float, float],
              r_start: float,
              pitch: float,
              r_end: float | None = None,
              _narc: int = 8,
              startfeed: float = 0.0) -> Curve:
        """Generates a Helical curve

        Args:
            pstart (tuple[float, float, float]): The start of the center of rotation (not the start of the curve)
            pend (tuple[float, float, float]): The end of the center of rotation
            r_start (float): The (start) radius of the helix
            pitch (float): The pitch angle of the helix
            r_end (float | None, optional): The ending radius. If default, the same is used as the start. Defaults to None.
            _narc (int, optional): The number of Spline arc sections used. Defaults to 8.

        Returns:
            Curve: The Curve geometry object
        """
        if r_end is None:
            r_end = r_start
        
        pitch = pitch*np.pi/180
        
        R1, R2, DR = r_start, r_end, r_end-r_start
        p0 = np.array(pstart)
        p1 = np.array(pend)
        dp = (p1-p0)
        L = (dp[0]**2 + dp[1]**2 + dp[2]**2)**(0.5)
        dp = dp/L
        
        Z, X, Y = orthonormalize(dp)
        
        Q = L/np.tan(pitch)
        #a1 = Q/R1
        #a2 = Q*((1 - 1/R1 *(R2 + DR))/(2*R2 + DR))
        C = 0#Q/R1

        wtot = C/R2 + Q/R2
        nt = int(np.ceil(wtot/(2*np.pi)*_narc))
        
        t = np.linspace(0, 1, nt)
        Rt = R1 + DR*t
        #wt = (a1*t + a2*t**2)
        wt = C/Rt + (Q*t)/Rt
        
        xs = (R1 + DR*t)*np.cos(wt)
        ys = (R1 + DR*t)*np.sin(wt)
        zs = L*t

        xp = xs*X[0] + ys*Y[0] + zs*Z[0] + p0[0]
        yp = xs*X[1] + ys*Y[1] + zs*Z[1] + p0[1]
        zp = xs*X[2] + ys*Y[2] + zs*Z[2] + p0[2]
        
        dp = tuple(Y)
        if startfeed > 0:
            dpx, dpy, dpz = Y
            dx = Z[0]
            dy = Z[1]
            dz = Z[2]
            d = startfeed

            fx = np.array([xp[0] - dx*d/2 - d*dpx, xp[0] - dx*d*0.8/2 - d*dpx])
            fy = np.array([yp[0] - dy*d/2 - d*dpy, yp[0] - dy*d*0.8/2 - d*dpy])
            fz = np.array([zp[0] - dz*d/2 - d*dpz, zp[0] - dz*d*0.8/2 - d*dpz])
            
            xp = np.concat([fx ,xp])
            yp = np.concat([fy, yp])
            zp = np.concat([fz, zp])
            xp[2] += d/2*dx
            yp[2] += d/2*dy
            zp[2] += d/2*dz
            dp = tuple(Z)
        
        return Curve(xp, yp, zp, ctype='Spline')
    
    @staticmethod
    def helix_lh(pstart: tuple[float, float, float],
              pend: tuple[float, float, float],
              r_start: float,
              pitch: float,
              r_end: float | None = None,
              _narc: int = 8,
              startfeed: float = 0.0) -> Curve:
        """Generates a Helical curve

        Args:
            pstart (tuple[float, float, float]): The start of the center of rotation (not the start of the curve)
            pend (tuple[float, float, float]): The end of the center of rotation
            r_start (float): The (start) radius of the helix
            pitch (float): The pitch angle of the helix
            r_end (float | None, optional): The ending radius. If default, the same is used as the start. Defaults to None.
            _narc (int, optional): The number of Spline arc sections used. Defaults to 8.

        Returns:
            Curve: The Curve geometry object
        """
        if r_end is None:
            r_end = r_start
        
        pitch = pitch*np.pi/180
        
        R1, R2, DR = r_start, r_end, r_end-r_start
        p0 = np.array(pstart)
        p1 = np.array(pend)
        dp = (p1-p0)
        L = (dp[0]**2 + dp[1]**2 + dp[2]**2)**(0.5)
        dp = dp/L
        
        Z, X, Y = orthonormalize(dp)
        
        Q = L/np.tan(pitch)
        #a1 = Q/R1
        #a2 = Q*((1 - 1/R1 *(R2 + DR))/(2*R2 + DR))
        C = 0#Q/R1

        wtot = C/R2 + Q/R2
        nt = int(np.ceil(wtot/(2*np.pi)*_narc))
        
        t = np.linspace(0, 1, nt)
        Rt = R1 + DR*t
        #wt = (a1*t + a2*t**2)
        wt = C/Rt + (Q*t)/Rt
        
        xs = (R1 + DR*t)*np.cos(-wt)
        ys = (R1 + DR*t)*np.sin(-wt)
        zs = L*t

        xp = xs*X[0] + ys*Y[0] + zs*Z[0] + p0[0]
        yp = xs*X[1] + ys*Y[1] + zs*Z[1] + p0[1]
        zp = xs*X[2] + ys*Y[2] + zs*Z[2] + p0[2]
        
        dp = tuple(Y)
        if startfeed > 0:
            dpx, dpy, dpz = Y
            dx = Z[0]
            dy = Z[1]
            dz = Z[2]
            d = startfeed

            fx = np.array([xp[0] - dx*d/2 + d*dpx, xp[0] - dx*d*0.8/2 + d*dpx])
            fy = np.array([yp[0] - dy*d/2 + d*dpy, yp[0] - dy*d*0.8/2 + d*dpy])
            fz = np.array([zp[0] - dz*d/2 + d*dpz, zp[0] - dz*d*0.8/2 + d*dpz])
            
            xp = np.concat([fx ,xp])
            yp = np.concat([fy, yp])
            zp = np.concat([fz, zp])
            xp[2] += d/2*dx
            yp[2] += d/2*dy
            zp[2] += d/2*dz
            dp = tuple(Z)
        
        return Curve(xp, yp, zp, ctype='Spline')
        
    def pipe(self, crossection: GeoSurface | XYPolygon, 
             max_mesh_size: float | None = None,
             start_tangent: Axis | tuple[float, float, float] | np.ndarray | None = None,
             x_axis: Axis | tuple[float, float, float] | np.ndarray | None = None,
             name: str = 'PipedVolume') -> GeoVolume:
        """Extrudes a surface or XYPolygon object along the given curve

        If a GeoSurface object is used, make sure it starts at the center of the curve. This property
        can be accessed with curve_obj.p0.Alignment
        If an XYPolygon is used, it will be automatically centered with XY=0 at the start of the curve with
        the Z-axis align along the initial departure direction curve_obj.dstart.Alignment
        
        Args:
            crossection (GeoSurface | XYPolygon): The cross section definition to be used
            max_mesh_size (float, optional): The maximum mesh size. Defaults to None
            start_tangent (Axis, tuple, ndarray, optional): The input polygon plane normal direction. Defaults to None
            x_axis (Axis, tuple, ndarray optional): The reference X-axis to align the input polygon. Defaults to None
        Returns:
            GeoVolume: The resultant volume object
        """
        if isinstance(crossection, XYPolygon):
            if start_tangent is None:
                start_tangent = self.dstart
            if x_axis is not None:
                xax = _parse_axis(x_axis)
                zax = _parse_axis(self.dstart)
                yax = zax.cross(xax)
                cs = CoordinateSystem(xax, yax, zax, self.p0)
            else:
                zax = self.dstart
                cs = Axis(np.array(zax)).construct_cs(self.p0)
                surf = crossection.geo(cs)
        else:
            surf = crossection
        x1, y1, z1, x2, y2, z2 = gmsh.model.occ.getBoundingBox(*surf.dimtags[0])
        diag = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(0.5)
        pipetag = gmsh.model.occ.addPipe(surf.dimtags, self.tags[0], 'GuidePlan')
        self.remove()
        surf.remove()
        volume = GeoVolume(pipetag[0][1], name=name)
        volume.max_meshsize = diag/2
        return volume
        