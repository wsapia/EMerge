
from ..._emerge.geo import PCB, extrude, Cylinder
from ..._emerge.geometry import GeoSurface
from ..._emerge.cs import GCS, CoordinateSystem
from ...lib import COPPER
from .gerber import GerberLayer
from pathlib import Path
from .excellon import parse_excellon_file
import gmsh
from collections import defaultdict

class FileBasedPCB(PCB):

    def layer_from_file(self, layer: int, filename: str, res_mm: float = 0.01, n_circ_segments: int = 8, segment_size_mm: float | None = None) -> GeoSurface:
        
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
        vias = []
        via_buffer = defaultdict(list)
        
        for via in parse_excellon_file(filename):
            rad = via['radius']*0.001/self.unit
            x = via['x']*0.001/self.unit
            y = via['y']*0.001/self.unit
            via_buffer[rad].append((x,y))
        
        for size, vias in via_buffer.items():
            self.add_vias(*vias, radius=size, z1=z1, z2=z2, segments=Nsections)
        