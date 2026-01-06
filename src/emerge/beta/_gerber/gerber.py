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

from pygerber.gerberx3.api.v2 import GerberFile, ParsedFile
from pygerber.gerberx3.parser2.commands2.region2 import Region2
from pygerber.gerberx3.parser2.commands2.line2 import Line2
from pygerber.gerberx3.parser2.commands2.flash2 import Flash2
from pygerber.gerberx3.parser2.commands2.arc2 import Arc2, CCArc2
from pygerber.gerberx3.parser2.apertures2.polygon2 import Polygon2
from pygerber.gerberx3.parser2.apertures2.circle2 import Circle2
from pygerber.gerberx3.parser2.apertures2.rectangle2 import Rectangle2

from math import cos, sin, pi
import numpy as np
from itertools import groupby
import math

from ..._emerge import geo
from ..._emerge.cs import CoordinateSystem, GCS
from ..._emerge.geometry import GeoSurface, GeoPolygon
from ..._emerge.mth.loopsplit import Loop

from loguru import logger

N_CIRC_MIN = 6
N_CIRC_MAX = 21

def _calc_via_segs(diameter: float, nsegments: int, edge_size: float | None = None) -> int:
    if edge_size is not None:
        N = int(np.ceil(np.pi*diameter/edge_size))
    else:
        N = nsegments
    return max(min(N,N_CIRC_MAX),N_CIRC_MIN)
    
def zero_runs_indices_gb(xs: list[int], min_len=3):
    """Returns all index sequences of the input list of subsequent ids that form a segmen of at least min_len zeros.

    Args:
        xs (_type_): _description_
        min_len (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    runs = []
    idx = 0
    for val, grp in groupby(xs):
        length = sum(1 for _ in grp)
        if val == 0 and length >= min_len:
            runs.append((idx, idx + length - 1))  # inclusive end
        idx += length
    return runs


############################################################
#                    POLYGON FUNCTIONS                    #
############################################################

def semi_circ(p: tuple[float, float], u: tuple[float, float], w: float, N: int, max_seg: float | None) -> list[tuple[float, float]]:
    N = _calc_via_segs(w, N, max_seg)
    x2, y2 = p
    ux, uy = u
    urx, ury = uy, -ux
    ths = np.linspace(0.01,np.pi-0.01,N)#[1:-1]
    pes = []
    ex = 0*w*np.sin(np.pi/(2*N))*ux 
    ey = 0*w*np.sin(np.pi/(2*N))*uy
    for th in ths:
        c = (w*urx + w*ury*1j)*np.exp(1j*th)
        px = x2 + c.real
        py = y2 + c.imag
        pes.append((px+ex, py+ey))
    return pes

def parse_region(cmd: Region2) -> list[tuple[float, float]]:
    ring = []
    for seg in cmd.command_buffer:   # nested boundary commands
        if isinstance(seg, Line2):
            # points are Vector2D with unit helpers; use millimeters
            if not ring:
                ring.append((float(seg.start_point.x.as_millimeters()),
                            float(seg.start_point.y.as_millimeters())))
            ring.append((float(seg.end_point.x.as_millimeters()),
                        float(seg.end_point.y.as_millimeters())))
            continue
    return ring

def xy_to_c(points: list[tuple, tuple]) -> list[complex]:
    return [p[0]+1j*p[1] for p in points]

def c_to_xy(points: list[complex]) -> list[tuple[float, float]]:
    return [(p.real, p.imag) for p in points]

def dphi_cw(dr) -> tuple[float, float]:
    rx, ry = dr
    R = (rx**2 + ry**2)**0.5
    return (ry/R, -rx/R)

def dphi_ccw(dr) -> tuple[float, float]:
    rx, ry = dr
    R = (rx**2 + ry**2)**0.5
    return (-ry/R, rx/R)

def parse_arc(cmd: Arc2, ds: float, reverse: bool, Nseg: int, max_size: float | None) -> list[tuple[float, float]]:
    xc =  float(cmd.center_point.x.as_millimeters())
    yc =  float(cmd.center_point.y.as_millimeters())
    dx0 = float(cmd.get_relative_start_point().x.as_millimeters())
    dy0 = float(cmd.get_relative_start_point().y.as_millimeters())
    dx1 = float(cmd.get_relative_end_point().x.as_millimeters())
    dy1 = float(cmd.get_relative_end_point().y.as_millimeters())
    
    if reverse:
        dx0, dy0, dx1, dy1 = dx1, dy1, dx0, dy0
        
    w = float(cmd.aperture.diameter.as_millimeters())/2
    
    print(w)
    dang = np.angle((dx1+1j*dy1)/(dx0+1j*dy0))
    if dang > 0:
        dang = dang - 2*np.pi
        #raise ValueError(f'Dang is positive {dang}, {xc},{yc},{dx0},{dy0},{dx1},{dy1}')
    
    R = (dx0**2 + dy0**2)**0.5
    Nang = abs(int(np.ceil(1.5*dang*R/ds)))
    angs = np.linspace(0,dang,Nang)
    
    arm = dx0 + 1j*dy0
    cc = xc + 1j*yc
    psc1 = c_to_xy([cc+arm*np.exp(1j*th)*(R-w)/R for th in angs])
    psc2 = c_to_xy([cc+arm*np.exp(1j*th)*(R+w)/R for th in angs][::-1])
    
    
    begin_cap = semi_circ((xc+dx0, yc+dy0), dphi_ccw((dx0, dy0)), w, Nseg, max_size)
    end_cap = semi_circ((xc+dx1, yc+dy1), dphi_cw((dx1, dy1)), w, Nseg, max_size)
    
    points = psc1+end_cap+psc2+begin_cap
    return points

def parse_line(cmd: Line2, min_dist: float, ncirc:int , size_circ) -> list[tuple[float, float]]:
    diam = float(cmd.aperture.diameter.as_millimeters())
    x1 = float(cmd.start_point.x.as_millimeters())
    y1 = float(cmd.start_point.y.as_millimeters())
    x2 = float(cmd.end_point.x.as_millimeters())
    y2 = float(cmd.end_point.y.as_millimeters())
    
    L = ((x2-x1)**2 + (y2-y1)**2)**0.5
    
    if L < min_dist/10:
        diam = float(cmd.aperture.diameter.as_millimeters())
        x = float(cmd.start_point.x.as_millimeters())
        y = float(cmd.start_point.y.as_millimeters())
        R = diam/2 #+ diam/2*np.sin(np.pi/12)
        points = [(x+R*np.cos(th), y+R*np.sin(th)) for th in np.linspace(0,2*np.pi,_calc_via_segs(diam, ncirc, size_circ))]
        return points
    
    ux = (x2-x1)/L
    uy = (y2-y1)/L

    urx = uy
    ury = -ux
    w = diam/2
    
    
    p1 = (x1+w*urx, y1+w*ury)
    p2 = (x2+w*urx, y2+w*ury)
    p3 = (x2-w*urx, y2-w*ury)
    p4 = (x1-w*urx, y1-w*ury)
    
    pes = semi_circ((x2, y2), (ux, uy), w, ncirc, size_circ)
    pss = semi_circ((x1, y1), (-ux, -uy), w, ncirc, size_circ)
    
    points = [p1, p2] + pes + [p3, p4] + pss
    return points

def parse_flash_circ(cmd: Flash2, nsegments: int, segsize: float | None = None) -> list[tuple[float, float]]:
    diam = float(cmd.aperture.diameter.as_millimeters())
    x = float(cmd.flash_point.x.as_millimeters())
    y = float(cmd.flash_point.y.as_millimeters())
    
    N = _calc_via_segs(diam, nsegments, segsize)
    points = [(x+diam/2*np.cos(th), y+diam/2*np.sin(th)) for th in np.linspace(0,2*np.pi,N+1)]
    return points

def parse_flash_rect(cmd: Rectangle2) -> list[tuple[float, float]]:
    x = float(cmd.flash_point.x.as_millimeters())
    y = float(cmd.flash_point.y.as_millimeters())
    sizeX = float(cmd.aperture.x_size.as_millimeters())
    sizeY = float(cmd.aperture.y_size.as_millimeters())
    rotation = float(cmd.aperture.rotation)
    
    # Half sizes
    hx, hy = sizeX / 2.0, sizeY / 2.0

    # Local (unrotated) corners relative to center
    corners = [
        (-hx, -hy),
        ( hx, -hy),
        ( hx,  hy),
        (-hx,  hy),
    ]

    cos_r, sin_r = np.cos(rotation), np.sin(rotation)

    # Rotate + translate
    points = [
        (
            x + dx * cos_r - dy * sin_r,
            y + dx * sin_r + dy * cos_r
        )
        for dx, dy in corners
    ]

    return points

############################################################
#                          CLASSES                         #
############################################################


class ClosedPoly:
    
    def __init__(self, points: list[tuple[float, float]], clear: bool, simplify: bool):
        points = [(x*0.001, y*0.001) for x,y in points]
        if points[0] != points[-1]:
            points.append(points[0])
            
        points = self._ensure_clockwise(points)
        
        self.points: list[tuple[float, float]] = points
        self.clear: bool = clear
        x, y = zip(*self.points)
        self.xs: list[float] = x
        self.ys: list[float] = y
        self.do_simplify: bool = simplify
    
    @staticmethod
    def _ensure_clockwise(points: list[tuple[float, float]]):
        # Signed area (shoelace formula)
        area = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            area += (x1 - x0) * (y1 + y0)

        # area > 0 → counter-clockwise → reverse
        if area > 0.0:
            return points[::-1]
        return points
    
    @property
    def dark(self) -> bool:
        return not self.clear
    
    @property
    def xy(self) -> tuple[list[float], list[float]]:
        return self.xs, self.ys
    
    def simplify(self, min_distance: float, min_incl_angle: float = 5, min_remove_angle: float = 10):
        """
        Simplify a closed polygon (last point is NOT repeated as first) by iteratively
        removing vertices according to the following rules:

        For each point i:
            dist_till_next = distance(i, i+1)
            dist_till_prev = distance(i, i-1)
            angle_next_prev = angle between vectors (prev -> i) and (i -> next)
                            in degrees, 0° = straight, 180° = U-turn

        A point is removed iff:
            |angle_next_prev| < min_incl_angle
        OR
            dist_till_next < min_distance AND
            dist_till_prev < min_distance AND
            |angle_next_prev| < min_remove_angle

        After any removal, the process restarts until no more points are removed.
        """

        # Optional guard: do nothing for very small polygons
        if len(self.points) <= 6:
            return

        # Work on local copies so we can pop safely
        xs = list(self.xs)
        ys = list(self.ys)
        n = len(xs)

        # Need at least a triangle
        if n < 3:
            return

        changed = True
        while changed and n > 6:
            changed = False

            for i in range(n):
                ip = (i - 1) % n        # previous index
                inext = (i + 1) % n     # next index

                x_i, y_i = xs[i], ys[i]
                x_p, y_p = xs[ip], ys[ip]
                x_n, y_n = xs[inext], ys[inext]

                # Distances to previous and next
                dx_prev = x_i - x_p
                dy_prev = y_i - y_p
                dx_next = x_n - x_i
                dy_next = y_n - y_i

                dist_prev = math.hypot(dx_prev, dy_prev)
                dist_next = math.hypot(dx_next, dy_next)

                # If either adjacent segment is degenerate, skip angle computation
                if dist_prev == 0.0 or dist_next == 0.0:
                    continue

                # Angle between (prev -> current) and (current -> next)
                dot = dx_prev * dx_next + dy_prev * dy_next
                denom = dist_prev * dist_next
                cos_angle = dot / denom

                # Clamp for numerical safety
                if cos_angle > 1.0:
                    cos_angle = 1.0
                elif cos_angle < -1.0:
                    cos_angle = -1.0

                angle = math.degrees(math.acos(cos_angle))  # in [0, 180]

                remove = False
                if abs(angle) < min_incl_angle:
                    remove = True
                elif (
                    dist_next < min_distance
                    and dist_prev < min_distance
                    and abs(angle) < min_remove_angle
                ):
                    remove = True

                # Do not reduce below a triangle
                if remove and n - 1 >= 6:
                    xs.pop(i)
                    ys.pop(i)
                    n -= 1
                    changed = True
                    break  # restart outer while-loop with updated polygon

        # Write back to object
        self.xs = xs
        self.ys = ys
        self.points = list(zip(xs, ys))



############################################################
#                       GERBER CLASS                      #
############################################################

class GerberLayer:
    
    def __init__(self, 
                 filename: str, 
                 res_mm: float,
                 n_circ_segments: int = 8,
                 seg_size_mm: float | None = None,
                 ignore_via_pads: bool = True,
                 cs: CoordinateSystem = GCS,
                 ):
        
        self.fname: str = filename
        self.pf: ParsedFile = GerberFile.from_file(filename).parse()
        self.buffer = self.pf._command_buffer
        self.polies: list[ClosedPoly] = []
        self.res_mm: float = res_mm
        
        self._nseg: int = n_circ_segments
        self._seg_size_mm: float = seg_size_mm
        
        self.xs = []
        self.ys = []
        
        self.cs: CoordinateSystem = cs
        self.parse_file()
        self.simplify()
    
    def bounds(self, margin: float | tuple[float, float, float, float] = 0.0) -> tuple[float, float, float, float]:
        if isinstance(margin, (float, int)):
            return min(self.xs)-margin, min(self.ys)-margin, max(self.xs)+margin, max(self.ys)+margin
        else:
            return min(self.xs)-margin[0], min(self.ys)-margin[1], max(self.xs)+margin[2], max(self.ys)+margin[3]
        
    def _add_poly(self, points: list[tuple], clear: bool, simplify: bool) -> None:
        if points[-1]==points[0]:
            points = points[:-1]
        poly = ClosedPoly(points, clear, simplify)
        self.polies.append(poly)
        self.xs.extend(poly.xs)
        self.ys.extend(poly.ys)
        
    
    def parse_file(self):
        for cmd in self.buffer:
            
            if 'dark' in str(cmd.transform.polarity).lower():
                clear = False
            else:
                clear = True
            
            if isinstance(cmd, Region2):
                poly = parse_region(cmd)
                self._add_poly(poly, clear, True)
                continue
            
            if isinstance(cmd, CCArc2):
                points = parse_arc(cmd, self.res_mm, False, self._nseg, self._seg_size_mm)
                self._add_poly(points, clear, True)
                continue
            
            if isinstance(cmd, Arc2):
                points = parse_arc(cmd, self.res_mm, True, self._nseg, self._seg_size_mm)
                self._add_poly(points, clear, True)
                continue
            
            if isinstance(cmd, Line2):
                points = parse_line(cmd, self.res_mm, self._nseg, self._seg_size_mm)
                self._add_poly(points, clear, True)
                continue
            
            if not isinstance(cmd, Flash2):
                print('Not a Flash2, Uprocessed command:', cmd)
                continue
            
            if isinstance(cmd.aperture, Polygon2):
                ap = cmd.aperture
                # Field names are available on the dataclass; inspect if unsure:
                # print(ap.model_dump())
                n = int(ap.vertices)             # number of sides
                R = float(ap.diameter.as_millimeters()) / 2.0
                # Flash placement lives in the command transform:
                tr = cmd.transform               # translation + rotation
                cx = float(tr.translation.x.as_millimeters())
                cy = float(tr.translation.y.as_millimeters())
                # Rotation is in degrees in PyGerber’s apertures; convert to radians if needed
                rot = float(ap.rotation) * pi / 180.0
                verts = [(cx + R * cos(2*pi*i/n + rot), cy + R * sin(2*pi*i/n + rot))
                        for i in range(n)]
                self._add_poly(verts, clear, True)
                continue
            
            if isinstance(cmd.aperture, Circle2):
                points = parse_flash_circ(cmd, self._nseg, self._seg_size_mm)
                self._add_poly(points, clear, True)
                continue
            
            if isinstance(cmd.aperture, Rectangle2):
                points = parse_flash_rect(cmd)
                self._add_poly(points, clear, True)
                continue
            logger.error(f'Unparsable command found! {cmd}. Please contact the EMerge developers with this Gerber file to get this command supported.')
     
    def simplify(self):
        for poly in self.polies:
            poly.simplify(self.res_mm)
    
    def flatten(self, z: float = 0.0) -> GeoSurface:
        """Merge a GerberLayer into a GeoSurface

        Raises:
            Exception: _description_

        Returns:
            GeoSurface: _description_
        """
        poly = None
        
        xall = []
        yall = []
        import gmsh

        for isdark, polygons in groupby(self.polies, key=lambda x: x.dark):
            if poly is None and not isdark:
                raise Exception('First polygon is a clear:(')
            surfs = []
            for pol in polygons:
                x, y = pol.xy
                loop = Loop(np.array(x),np.array(y))
                add, remove = loop.split()
                LA = []
                LR = []
                for ax, ay in add:
                    LA.append(geo.XYPolygon(ax, ay, resolution=1e-7).geo(self.cs.displace(0,0,z)))
                    xall.extend(ax)
                    yall.extend(ay)
                for ax, ay in remove:
                    LR.append(geo.XYPolygon(ax, ay, resolution=1e-7).geo(self.cs.displace(0,0,z)))
                
                if len(LA)==1 and len(LR)==0:
                    surfs.append(LA[0])
                    continue
                
                if len(LR)==0:
                    surf = geo.unite(*LA)
                    surfs.append(surf)
                    continue
                
                surf = geo.subtract(geo.unite(*LA), geo.unite(*LR))
                surfs.append(surf)
           
            self.xs = xall
            self.ys = yall
            gmsh.fltk.run()
            if poly is None:
                poly = geo.unite(*surfs)
                continue
            
            if isdark:
                poly = geo.unite(poly, *surfs)
            else:
                poly = geo.subtract(poly, geo.unite(*surfs))
            
            
        return poly
            