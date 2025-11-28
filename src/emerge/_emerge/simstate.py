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
from .mesh3d import Mesh3D
from .geometry import GeoObject, _GeometryManager, _GEOMANAGER
from .dataset import SimulationDataset
from loguru import logger
from typing import Any
import numpy as np
from .selection import _CALC_INTERFACE
import gmsh

class _SimStateCollection:
    
    def __init__(self):
        self.states: list[SimState] = []
        self.active: SimState | None = None

    def sign_on(self, state: SimState) -> None:
        if state not in self.states:
            self.states.append(state)
        self.active = state
            
_GLOBAL_SIMSTATES = _SimStateCollection()

class SimState:
    
    def __init__(self, modelname: str):
        self.modelname: str = modelname
        self.mesh: Mesh3D = Mesh3D()
        self.geos: list[GeoObject] = []
        self.data: SimulationDataset = SimulationDataset()
        self.params: dict[str, float] = dict()
        self._stashed: SimulationDataset | None = None
        self.manager: _GeometryManager = _GEOMANAGER
        self.sign_on()
    
    def sign_on(self):
        _GLOBAL_SIMSTATES.sign_on(self)
        _CALC_INTERFACE._ifobj = self

    @property
    def current_geo_state(self) -> list[GeoObject]:
        return self.manager.all_geometries()
    
    def reset_geostate(self) -> None:
        _GEOMANAGER.reset(self.modelname)
        self.clear_mesh()
        
    def init(self) -> None:
        """Initializes the Simstate to a clean starting point.
        """
        self.mesh = Mesh3D()
        self.geos = []
        self.reset_geostate()
        self.init_data()
        self.sign_on()
        
    def stash(self) -> None:
        """Stashes the simstate data to run simulations and restore it later.
        """
        self._stashed = self.data
        self.data = SimulationDataset()
    
    def set_parameters(self, parameters: dict[str, float]) -> None:
        """ Define the simulation parameter sweep to a set of outer variables
        defined by a dictionary of string, float pairs.

        Args:
            parameters (dict[str, float]): The parameter sweep slice.
        """
        self.params = parameters
        
    def init_data(self) -> None:
        """Initializes a the dataset with the current parameters
        """
        self.data.sim.new(**self.params)
        
    def reload(self) -> SimulationDataset:
        """Reload stashed data into the simstate memory

        Returns:
            SimulationDataset: _description_
        """
        old = self._stashed
        self.data = self._stashed
        self._stashed = None
        return old
    
    def reset_mesh(self) -> None:
        """Resets and clears the current mesh.
        """
        self.mesh = Mesh3D()
        
    def set_mesh(self, mesh: Mesh3D) -> None:
        """ Overwrite the mesh with a new one."""
        self.mesh = mesh
        
    def set_geos(self, geos: list[GeoObject]) -> None:
        """Activate a set of geometry objects to the simstate geometry field.

        Args:
            geos (list[GeoObject]): _description_
        """
        self.geos = geos
        _GEOMANAGER.set_geometries(geos)

    def clear_mesh(self) -> None:
        """resets the current mesh object to an empty one.
        """
        self.mesh = Mesh3D()
        
    def store_geometry_data(self) -> None:
        """Saves the current geometry state to the simulatin dataset
        """
        logger.trace('Storing geometries in data.sim')
        self.geos = self.current_geo_state
        self.data.sim['geos'] = self.geos
        self.data.sim['mesh'] = self.mesh
        
    def get_dataset(self) -> dict[str, Any]:
        """Create a dict of the file data to store to the harddrive.

        Returns:
            dict[str, Any]: _description_
        """
        return dict(simdata=self.data, mesh=self.mesh)
    
    def load_dataset(self, dataset: dict[str, Any]):
        """Load the data from a dataset

        Args:
            dataset (dict[str, Any]): _description_
        """
        self.data = dataset['simdata']
        self.mesh = dataset['mesh']
        
    def activate(self, _indx: int | None = None, **variables):
        """Searches for the permutaions of parameter sweep variables and sets the current geometry to the provided set."""
        if _indx is not None:
            dataset = self.data.sim.index(_indx)
        else:
            dataset = self.data.sim.find(**variables)
        
        variables = ', '.join([f'{key}={value}' for key,value in dataset.vars.items()])
        logger.info(f'Activated entry with variables: {variables}')
        self.set_mesh(dataset['mesh'])
        self.set_geos(dataset['geos'])
        
        return self
    

    ############################################################
    #                       GMSH LIKE METHODS                  #
    ############################################################

    def getCenterOfMass(self, dim: int, tag: int) -> np.ndarray:
        if self.mesh.defined is False:
            return gmsh.model.occ.getCenterOfMass(dim, tag)
        return self.mesh.dimtag_to_center[(dim, tag)]
    
    def getPoints(self, dimtags: list[tuple[int, int]]) -> list[np.ndarray]:
        """Returns a

        Args:
            dimtags (list[tuple[int, int]]): _description_

        Returns:
            list[np.ndarray]: _description_
        """
        points = []
        id_set = []
        for dt in dimtags:
            id_set.append(self.mesh.dimtag_to_nodes[dt])
        ids = np.unique(np.concatenate(id_set))
        points = [self.mesh.nodes[:,i] for i in ids]
        return points
    
    def getBoundingBox(self, dim: int, tag: int) -> tuple[float, float, float, float, float, float]:
        """Returns the bounding box corresponding to entity defined by (dim,tag)

        Args:
            dim (int): _description_
            tag (int): _description_

        Returns:
            tuple[float, float, float, float, float, float]: _description_
        """
        if self.mesh.defined is False:
            return gmsh.model.occ.getBoundingBox(dim, tag)
        return self.mesh.dimtag_to_bb[(dim, tag)]
    
    def getNormal(self, facetag: int) -> np.ndarray:
        """Returns the normal vector of a facetag

        Args:
            facetag (int): _description_

        Returns:
            np.ndarray: _description_
        """
        if self.mesh.defined:
            return self.mesh.ftag_to_normal[facetag]
        else:
            return np.array(gmsh.model.getNormal(facetag, (0,0)))
    
    def getCharPoint(self, facetag: int) -> np.ndarray:
        """Returns a coordinate that is always on the surface and roughly in the center

        Args:
            facetag (int): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.mesh.ftag_to_point[facetag]
    
    def getArea(self, tag: int) -> float:
        if self.mesh.defined is True:
            area = sum([self.mesh.areas[tri] for tri in self.mesh.ftag_to_tri[tag]])
            return area
        else:
            area = gmsh.model.occ.getMass(2,tag)
            return area