
from typing import Any
from ..mesh3d import Mesh3D
from ..geometry import GeoObject
from ..selection import Selection
from ..physics.microwave.microwave_3d import Microwave3D
from ..settings import Settings

class DataSet:
    def __init__(self):
        pass

class SimState:
    
    def __init__(self, settings: Settings, params: dict[str, float]):
        self.params: dict[str, float] = params
        self.set: Settings = settings
        self.mesh: Mesh3D = Mesh3D()
        self.selections: dict[str, Selection] = dict()
        self.geos: dict[str, GeoObject] = dict()
        self.tasks: list[Any] = []
        self.data: DataSet = DataSet()
        self.mw: Microwave3D = Microwave3D()
        
class SimData:
    
    def __init__(self):
        self.states: list[SimState] = []