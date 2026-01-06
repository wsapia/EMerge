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

# Last Cleanup: 2026-01-04

from __future__ import annotations
import numpy as np
from typing import Callable

def gauss3_composite(x: np.ndarray, y: np.ndarray) -> float:
    """
    Composite 3-point Gauss (order-2) integral
    on *equally spaced* 1-D data.

    Sections:
        (x0,x1,x2), (x2,x3,x4), (x4,x5,x6), ...

    The rule on one section [x_i, x_{i+2}] is

        ∫ f dx  ≈  (h/2) Σ_k w_k · f( x̄ + (h/2) ξ_k )

    where  h = x_{i+2}−x_i  and
           ξ = (−√3/5, 0, +√3/5),
           w = (8/9, 5/9, 8/9)

    """
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1-D arrays of equal length.")
    if x.size % 2 == 0:
        raise ValueError("Number of samples must be odd (… 0,1,2; 2,3,4; …).")

    xi = np.sqrt(3/5)
    nodes   = np.array([-xi, 0.0, +xi])
    weights = np.array([5/9, 8/9, 5/9])

    total = 0.0
    for i in range(0, x.size - 2, 2):
        y0, y1, y2 = y[i:i+3]
        a = y0*0.5 - y1 + 0.5*y2
        b = -y0*0.5 + 0.5*y2
        c = y1
        poly_vals = a*nodes**2 + b*nodes + c
        total += np.dot(weights, poly_vals)
    return total

class Line:
    """ A Line class used for convenient definition of integration lines"""
    def __init__(self, xpts: np.ndarray,
                 ypts: np.ndarray,
                 zpts: np.ndarray):
        
        self.xs: np.ndarray = xpts
        self.ys: np.ndarray = ypts
        self.zs: np.ndarray = zpts
        self.dxs: np.ndarray = xpts[1:] - xpts[:-1]
        self.dys: np.ndarray = ypts[1:] - ypts[:-1]
        self.dzs: np.ndarray = zpts[1:] - zpts[:-1]
        self.dl = np.sqrt(self.dxs**2 + self.dys**2 + self.dzs**2)
        self.length: float = np.sum(np.sqrt(self.dxs**2 + self.dys**2 + self.dzs**2))
        self.l: np.ndarray = np.concatenate((np.array([0,]), np.cumsum(self.dl))) 
        self.xmid: np.ndarray = 0.5*(xpts[:-1] + xpts[1:])
        self.ymid: np.ndarray = 0.5*(ypts[:-1] + ypts[1:])
        self.zmid: np.ndarray = 0.5*(zpts[:-1] + zpts[1:])

        self.dx: float = self.dxs[0]
        self.dy: float = self.dys[0]
        self.dz: float = self.dzs[0]
    
    @property
    def cmid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The midpoints of the line segments.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The midpoint coordinates (xmid, ymid, zmid).
        """
        return self.xmid, self.ymid, self.zmid

    @property
    def cpoint(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The center points of the line segments."""
        return self.xs, self.ys, self.zs
    
    @staticmethod
    def from_points(start: np.ndarray, end: np.ndarray, Npts: int) -> Line:
        """Create a Line object from start to end with Npts points.

        Args:
            start (np.ndarray): start point coordinates (x,y,z).
            end (np.ndarray): end point coordinates (x,y,z).
            Npts (int): Number of points along the line.

        Returns:
            Line: The created Line object.
        """
        x1, y1, z1 = start
        x2, y2, z2 = end
        xs = np.linspace(x1, x2, Npts)
        ys = np.linspace(y1, y2, Npts)
        zs = np.linspace(z1, z2, Npts)
        return Line(xs, ys, zs)
    
    @staticmethod
    def from_path(*points: tuple[float,float,float] | np.ndarray, ds: float) -> Line:
        """Creates a Line object from a series of points with approximate spacing ds.

        Args:
            ds (float): Approximate spacing between points.

        Returns:
            Line: The created Line object.
        """
        xpts = []
        ypts = []
        zpts = []
        points = [np.array(p) for p in points]
        xl = None
        yl = None
        zl = None
        for p1, p2 in zip(points[:-1], points[1:]):
            N = max(3,int(np.ceil(np.linalg.norm(p2-p1)/ds)))
            *xs, xl = list(np.linspace(p1[0], p2[0], N))
            *ys, yl = list(np.linspace(p1[1], p2[1], N))
            *zs, zl = list(np.linspace(p1[2], p2[2], N))
            xpts.extend(xs)
            ypts.extend(ys)
            zpts.extend(zs)
        xpts.append(xl)
        ypts.append(yl)
        zpts.append(zl)
        return Line(np.array(xpts), np.array(ypts), np.array(zpts))
            
    def line_integral(self, evalfunc: Callable) -> complex:
        """Compute the line integral for a complex vector field function evalfunc."""
        Ex, Ey, Ez = evalfunc(*self.cpoint)
        EdotL = Ex*self.dx + Ey*self.dy + Ez*self.dz
        return gauss3_composite(self.l, EdotL)
    
    def _integrate(self, quantity: np.ndarray) -> complex:
        """Integrates a quantity of values defined along the line.

        Args:
            quantity (np.ndarray): The quantity values along the line.

        Returns:
            complex: The integrated value.
        """
        return gauss3_composite(self.l, quantity)