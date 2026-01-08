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

from typing import TypeVar, Generic, Any
from .physics.microwave.microwave_data import MWData
from .simulation_data import DataContainer

class SimulationDataset:
    """This simple class contains the different kinds of data sets in the Simulation Model. It includes

    Attributes:
      self.mw: MWData              - The Microwave physics data
      self.globals: dict[str, Any] - Any globally defined data of choice in the Simulation
      self.sim: DataContainer      - Generic simulation data associated with different instantiation of your at parameter level.
    """
    def __init__(self):
        self.mw: MWData = MWData()
        self.globals: dict[str, Any] = dict()
        self.sim: DataContainer = DataContainer()
        