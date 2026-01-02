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
from ...mesh3d import Mesh3D
from ...simstate import SimState
from ...geometry import GeoObject
from ...selection import FaceSelection, DomainSelection, EdgeSelection, Selection, encode_data
from ...physics.microwave.microwave_bc import PortBC, ModalPort

from emsutil.pyvista import EMergeDisplay, setdefault, cmap_names, _AnimObject
import numpy as np
from typing import Iterable, Literal

import pyvista as pv

def _min_distance(xs, ys, zs):
    """
    Compute the minimum Euclidean distance between any two points
    defined by the 1D arrays xs, ys, zs.
    
    Parameters:
        xs (np.ndarray): x-coordinates of the points
        ys (np.ndarray): y-coordinates of the points
        zs (np.ndarray): z-coordinates of the points
    
    Returns:
        float: The minimum Euclidean distance between any two points
    """
    # Stack the coordinates into a (N, 3) array
    points = np.stack((xs, ys, zs), axis=-1)

    # Compute pairwise squared distances using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_squared = np.sum(diff ** 2, axis=-1)

    # Set diagonal to infinity to ignore zero distances to self
    np.fill_diagonal(dists_squared, np.inf)

    # Get the minimum distance
    min_dist = np.sqrt(np.min(dists_squared))
    return min_dist

def _select(obj: GeoObject | Selection) -> Selection:
    if isinstance(obj, GeoObject):
        return obj.selection
    return obj

def _merge(lst: Iterable[GeoObject | Selection]) -> Selection:
    selections = [_select(item) for item in lst]
    dim = selections[0].dim
    all_tags = []
    for item in lst:
        all_tags.extend(_select(item).tags)
    
    if dim==1:
        return EdgeSelection(all_tags)
    elif dim==2:
        return FaceSelection(all_tags)
    elif dim==3:
        return DomainSelection(all_tags)
    else:
        return Selection(all_tags)

class PVDisplay(EMergeDisplay):

    def __post_init__(self, state: SimState):
        self._state: SimState = state
        self._selector._set_encoder_function(encode_data)
        
    def _get_edge_length(self):
        return max(1e-3, min(self._mesh.edge_lengths))
    ############################################################
    #                       SPECIFIC METHODS                  #
    ############################################################
    
    def _volume_edges(self, obj: GeoObject | Selection) -> pv.UnstructuredGrid:
        """Adds the edges of objects

        Args:
            obj (DomainSelection | None, optional): _description_. Defaults to None.

        Returns:
            pv.UnstructuredGrid: The unstrutured grid object
        """
        edge_ids = self._mesh.domain_edges(obj.dimtags)
        nedges = edge_ids.shape[0]
        cells = np.zeros((nedges,3), dtype=np.int64)
        cells[:,1:] = self._mesh.edges[:,edge_ids].T
        cells[:,0] = 2
        celltypes = np.full(nedges, fill_value=pv.CellType.CUBIC_LINE, dtype=np.uint8)
        points = self._mesh.nodes.copy().T
        return pv.UnstructuredGrid(cells, celltypes, points)
    
    def mesh_surface(self, surface: FaceSelection) -> pv.UnstructuredGrid:
        tris = self._mesh.get_triangles(surface.tags)
        ntris = tris.shape[0]
        cells = np.zeros((ntris,4), dtype=np.int64)
        cells[:,1:] = self._mesh.tris[:,tris].T
        cells[:,0] = 3
        celltypes = np.full(ntris, fill_value=pv.CellType.TRIANGLE, dtype=np.uint8)
        points = self._mesh.nodes.copy().T
        points[:,2] += self.set.z_boost
        return pv.UnstructuredGrid(cells, celltypes, points)
    
    def mesh(self, obj: GeoObject | Selection | Iterable) -> pv.UnstructuredGrid | None:
        if isinstance(obj, Iterable):
            obj = _merge(obj)
        else:
            obj = _select(obj)
        
        if isinstance(obj, DomainSelection):
            return self.mesh_volume(obj)
        elif isinstance(obj, FaceSelection):
            return self.mesh_surface(obj)
        else:
            return None


    ############################################################
    #                        EMERGE METHODS                    #
    ############################################################

    def mesh_volume(self, volume: DomainSelection) -> pv.UnstructuredGrid:
        tets = self._mesh.get_tetrahedra(volume.tags)
        ntets = tets.shape[0]
        cells = np.zeros((ntets,5), dtype=np.int64)
        cells[:,1:] = self._mesh.tets[:,tets].T
        cells[:,0] = 4
        celltypes = np.full(ntets, fill_value=pv.CellType.TETRA, dtype=np.uint8)
        points = self._mesh.nodes.copy().T
        return pv.UnstructuredGrid(cells, celltypes, points)
    
    
    @property
    def _mesh(self) -> Mesh3D:
        return self._state.mesh
    
    def add_object(self, obj: GeoObject | Selection, 
                   mesh: bool = False, 
                   volume_mesh: bool = True, 
                   label: bool = False, 
                   label_text: str | None = None, 
                   texture: str | None = None, *args, **kwargs):
        
        if isinstance(obj, GeoObject):
            if obj._hidden:
                return
        
        
        self._add_obj(self.mesh(obj), obj.dim, 
                      plot_mesh=mesh, 
                      volume_mesh=volume_mesh, 
                      metal=obj._metal, 
                      opacity=obj.opacity,
                      color=obj.color_rgb, 
                      texture=texture)

        self._plot.add_mesh(self._volume_edges(_select(obj)), line_width=self.set.theme.geo_edge_width, color=self.set.theme.geo_edge_color, show_edges=True)

        if label:
            points = []
            labels = []
            label_text = obj.name if label_text is None else label_text
            for dim, tag in obj.dimtags:
                if dim==2:
                    points.append(self._mesh.ftag_to_point[tag])
                else:
                    points.append(self._mesh.dimtag_to_center[(dim, tag)])
                labels.append(label_text)
            self._plot.add_point_labels(points, labels, shape_color=self.set.theme.label_color)
            
    def add_objects(self, *objects, **kwargs) -> None:
        """Add a series of objects provided as a list of arguments
        """
        for obj in objects:
            self.add_object(obj, **kwargs)
        
    def add_scatter(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        """Adds a scatter point cloud

        Args:
            xs (np.ndarray): The X-coordinate
            ys (np.ndarray): The Y-coordinate
            zs (np.ndarray): The Z-coordinate
        """
        cloud = pv.PolyData(np.array([xs,ys,zs]).T)
        self._data_sets.append(cloud)
        self._plot.add_points(cloud)

    def add_portmode(self, port: PortBC, 
                     Npoints: int = 10, 
                     dv=(0,0,0), 
                     XYZ=None,
                     field: Literal['E','H'] = 'E', 
                     k0: float | None = None,
                     mode_number: int | None = None) -> None:
        
        if XYZ:
            X,Y,Z = XYZ
        else:
            tris = self._mesh.get_triangles(port.selection.tags)
            ids = np.sort(np.unique(self._mesh.tris[:,tris].flatten()))
            X = self._mesh.tri_centers[0,tris]
            Y = self._mesh.tri_centers[1,tris]
            Z = self._mesh.tri_centers[2,tris]
            X2 = self._mesh.nodes[0,ids]
            Y2 = self._mesh.nodes[1,ids]
            Z2 = self._mesh.nodes[2,ids]
            X = np.concatenate((X,X2))
            Y = np.concatenate((Y,Y2))
            Z = np.concatenate((Z,Z2))
        
        X = X+dv[0]
        Y = Y+dv[1]
        Z = Z+dv[2]
        xf = X.flatten()
        yf = Y.flatten()
        zf = Z.flatten()

        d = _min_distance(xf, yf, zf)

        if port.vintline is not None:
            for line in port.vintline:
                xs, ys, zs = line.cpoint
                p_line = pv.Line(
                    pointa=(xs[0], ys[0], zs[0]),
                    pointb=(xs[-1], ys[-1], zs[-1]),
                )
                self._plot.add_mesh(
                    p_line,
                    color='red',
                    pickable=False,
                    line_width=3.0,
                )
            
        if k0 is None:
            if isinstance(port, ModalPort):
                k0 = port.get_mode(0).k0
            else:
                k0 = 1
        
        if isinstance(mode_number, int):
            port.selected_mode = mode_number
        
        F = port.port_mode_3d_global(xf,yf,zf,k0, which=field)

        Fx = F[0,:].reshape(X.shape).T
        Fy = F[1,:].reshape(X.shape).T
        Fz = F[2,:].reshape(X.shape).T

        if field=='H':
            F = np.imag(F.T)
            Fnorm = np.sqrt(Fx.imag**2 + Fy.imag**2 + Fz.imag**2).T
        else:
            F = np.real(F.T)
            Fnorm = np.sqrt(Fx.real**2 + Fy.real**2 + Fz.real**2).T
        
        if XYZ is not None:
            grid = pv.StructuredGrid(X,Y,Z)
            self.add_surf(X,Y,Z,Fnorm, _fieldname = 'portfield')
            self._wrap_plot(grid, scalars = Fnorm.T, opacity=0.8, pickable=False)

        Emag = F/np.max(Fnorm.flatten())*d*3
        actor = self._plot.add_arrows(np.array([xf,yf,zf]).T, Emag)
        self._data_sets.append(actor.mapper.dataset)
        
    def add_surf(self, 
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 field: np.ndarray,
                 scale: Literal['lin','log','symlog'] = 'lin',
                 cmap: cmap_names | None = None,
                 clim: tuple[float, float] | None = None,
                 opacity: float = 1.0,
                 symmetrize: bool = False,
                 _fieldname: str | None = None,
                 **kwargs,) -> pv.DataSet:
        """Add a surface plot to the display
        The X,Y,Z coordinates must be a 2D grid of data points. The field must be a real field with the same size.

        Args:
            x (np.ndarray): The X-grid array
            y (np.ndarray): The Y-grid array
            z (np.ndarray): The Z-grid array
            field (np.ndarray): The scalar field to display
            scale (Literal["lin","log","symlog"], optional): The colormap scaling¹. Defaults to 'lin'.
            cmap (cmap_names, optional): The colormap. Defaults to 'coolwarm'.
            clim (tuple[float, float], optional): Specific color limits (min, max). Defaults to None.
            opacity (float, optional): The opacity of the surface. Defaults to 1.0.
            symmetrize (bool, optional): Wether to force a symmetrical color limit (-A,A). Defaults to True.
        
        (¹): lin: f(x)=x, log: f(x)=log₁₀(|x|), symlog: f(x)=sgn(x)·log₁₀(1+|x·ln(10)|)
        """
        
        grid = pv.StructuredGrid(x,y,z)
        field_flat = field.flatten(order='F')
        
        if scale=='log':
            T = lambda x: np.log10(np.abs(x+1e-12))
        elif scale=='symlog':
            T = lambda x: np.sign(x) * np.log10(1 + np.abs(x*np.log(10)))
        else:
            T = lambda x: x
        
        static_field = T(np.real(field_flat))
        
        if _fieldname is None:
            name = 'anim'+str(self._ctr)
        else:
            name = _fieldname
        self._ctr += 1
        
        has_nan = np.any(np.isnan(static_field))
        if has_nan:
            nan_opacity = 0.0
        else:
            nan_opacity = 1.0
            
        grid[name] = static_field
        
        grid_no_nan = grid.threshold(scalars=name)
        
        default_cmap = self.set.theme.default_amplitude_cmap
        # Determine color limits
        if clim is None:
            if self._cbar_lim is not None:
                clim = self._cbar_lim
            else:
                fmin = np.nanmin(static_field)
                fmax = np.nanmax(static_field)
                clim = (fmin, fmax)
        
        if symmetrize:
            lim = max(abs(clim[0]), abs(clim[1]))
            clim = (-lim, lim)
            default_cmap = self.set.theme.default_wave_cmap
        
        if cmap is None:
            cmap = default_cmap
        else:
            cmap = self.set.theme.parse_cmap_name(cmap)
            
        kwargs = setdefault(kwargs, cmap=cmap, clim=clim, opacity=opacity, pickable=False, multi_colors=True, nan_opacity=nan_opacity)
        actor = self._wrap_plot(grid_no_nan, scalars=name, scalar_bar_args=self._cbar_args, **kwargs)
        
        if self._animate_next:
            def on_update(obj: _AnimObject, phi: complex):
                field_anim = obj.T(np.real(obj.field * phi))
                obj.grid[name] = field_anim
                obj.fgrid[name] = obj.grid.threshold(scalars=name)[name]
                #obj.fgrid replace with thresholded scalar data.
            self._objs.append(_AnimObject(field_flat, T, grid, grid_no_nan, actor, on_update))
            self._animate_next = False
        self._reset_cbar()
        return grid_no_nan