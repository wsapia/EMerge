
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

from ..._emerge.geo import PCB, extrude
from ..._emerge.geometry import GeoSurface
from .gerber import GerberLayer
from .excellon import parse_excellon_file
from collections import defaultdict

class FileBasedPCB(PCB):
    """Adds PCB construction from Gerber and Excellon files.

    Args:
        PCB (_type_): _description_
    """
    def layer_from_file(self, layer: int, filename: str, res_mm: float = 0.01, n_circ_segments: int = 8, segment_size_mm: float | None = None) -> GeoSurface:
        """Create a layer from a Gerber file.

        Args:
            layer (int): The layer number.
            filename (str): The path to the Gerber file.
            res_mm (float, optional): The resolution in mm. Defaults to 0.01.
            n_circ_segments (int, optional): The number of circular segments. Defaults to 8.
            segment_size_mm (float | None, optional): The size of the segments in mm. Defaults to None.

        Returns:
            GeoSurface: The generated surface.
        """
        gerber = GerberLayer(filename, 0.01, 8, segment_size_mm, cs=self.cs, ignore_via_pads=True)
        z = self.z(layer)*self.unit
        surf = gerber.flatten(z)
        
        self.xs.extend([x/self.unit for x in gerber.xs])
        self.ys.extend([y/self.unit for y in gerber.ys])
        
        if self._thick_traces:
            dx, dy, dz = self.cs.zax.np*self.unit
            vol = extrude(surf, dx, dy, dz).set_material(self.trace_material)
            return vol
        else:
            surf.set_material(self.trace_material)
        return surf
        
    def vias_from_file(self, filename: str, z1: float | None = None, z2: float | None = None, Nsections: int = 6) -> None:
        """Add vias from an Excellon file.

        Args:
            filename (str): The path to the Excellon file.
            z1 (float | None, optional): The starting z-coordinate. Defaults to None.
            z2 (float | None, optional): The ending z-coordinate. Defaults to None.
            Nsections (int, optional): The number of sections for the vias. Defaults to 6.
        """
        vias = []
        via_buffer = defaultdict(list)
        
        for via in parse_excellon_file(filename):
            rad = via['radius']*0.001/self.unit
            x = via['x']*0.001/self.unit
            y = via['y']*0.001/self.unit
            via_buffer[rad].append((x,y))
        
        for size, vias in via_buffer.items():
            self.add_vias(*vias, radius=size, z1=z1, z2=z2, segments=Nsections)
        