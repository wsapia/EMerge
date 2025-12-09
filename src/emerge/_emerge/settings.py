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


class Settings:
    def __init__(self):
        self._mw_2dbc: bool = True 
        self._mw_2dbc_lim: float = 10.0
        self._mw_2dbc_peclim: float = 1e8
        self._mw_3d_peclim: float = 1e7
        self._mw_cap_sp_single: bool = True
        self._mw_cap_sp_col: bool = True
        self._mw_recip_sp: bool = False
        self._size_check: bool = True
        self._auto_save: bool = False
        self._save_after_sim: bool = True

    ############################################################
    #                            GETTERS                       #
    ############################################################

    @property
    def mw_2dbc(self) -> bool:
        """ This variable determines is 2D boundary conditions will be automatically assigned based on material properties.
        """
        return self._mw_2dbc
    
    @property
    def mw_2dbc_lim(self) -> float:
        """This variable is the bulk conductivity limit in S/m beyond which a surface material will automatically be assigned as a SurfaceImpedance boundary condition."""
        return self._mw_2dbc_lim
    
    @property
    def mw_2dbc_peclim(self) -> float:
        """This variable determines a bulk conductivity limit in S/m beyond which a conductor is assigned PEC instead of a SurfaceImpedance boundary condition."""
        return self._mw_2dbc_peclim
    
    @property
    def mw_3d_peclim(self) -> float:
        """This variable determines if bulk conductors with a bulk conductivity beyond a limit (.mw_3d_peclim) are considered PEC.

        """
        return self._mw_3d_peclim
    
    @property
    def size_check(self) -> bool:
        """If a total volume check should be considered (100,000 tetrahedra) to hard crash the simulation assuming that the problem size will be too high to solver.
        100.000 Tetrahedra would yield approximately 700k Degrees of Freedom
        """
        return self._size_check
    
    @property
    def mw_cap_sp_single(self) -> bool:
        """If Single S-parameters should be capped with their magnitude to at most 1.0"""
        return self._mw_cap_sp_single
    
    @property
    def mw_cap_sp_col(self) -> bool:
        """If Single S-parameters columns should be power normalized to 1.0"""
        return self._mw_cap_sp_col
    
    @property
    def mw_recip_sp(self) -> bool:
        """If reciprodicty should be explicitly enforced"""
        return self._mw_recip_sp
    
    @property
    def auto_save(self) -> bool:
        """If the simulation should automatically be saved upon a detected abortion of the simulation.
        """
        return self._auto_save
    
    @property
    def save_after_sim(self) -> bool:
        """It the simulation should be saved only if a simulation is completed.

        """
        return self._save_after_sim
    ############################################################
    #                            SETTERS                       #
    ############################################################

    @mw_2dbc.setter
    def mw_2dbc(self, value: bool) -> None:
        """ This variable determines is 2D boundary conditions will be automatically assigned based on material properties.
        """
        self._mw_2dbc = value
        
    @mw_2dbc_lim.setter
    def mw_2dbc_lim(self, value: float):
        """This variable is the bulk conductivity limit in S/m beyond which a surface material will automatically be assigned as a SurfaceImpedance boundary condition."""
        self._mw_2dbc_lim = value
    
    @mw_2dbc_peclim.setter
    def mw_2dbc_peclim(self, value: float):
        """This variable determines a bulk conductivity limit in S/m beyond which a conductor is assigned PEC instead of a SurfaceImpedance boundary condition."""
        
        self._mw_2dbc_peclim = value
    
    @mw_3d_peclim.setter
    def mw_3d_peclim(self, value: float):
        """This variable determines if bulk conductors with a bulk conductivity beyond a limit (.mw_3d_peclim) are considered PEC.

        """
        self._mw_3d_peclim = value
        
    @size_check.setter
    def size_check(self, value: bool):
        """If a total volume check should be considered (100,000 tetrahedra) to hard crash the simulation assuming that the problem size will be too high to solver.
        100.000 Tetrahedra would yield approximately 700k Degrees of Freedom
        """
        self._size_check = value
        
    @mw_cap_sp_single.setter
    def mw_cap_sp_single(self, value: bool) -> None:
        """If Single S-parameters should be capped with their magnitude to at most 1.0"""
        self._mw_cap_sp_single = value
    
    @mw_cap_sp_col.setter
    def mw_cap_sp_col(self, value: bool) -> None:
        """If Single S-parameters columns should be power normalized to 1.0"""
        self._mw_cap_sp_col = value
    
    @mw_recip_sp.setter
    def mw_recip_sp(self, value: bool) -> None:
        """If reciprodicty should be explicitly enforced"""
        self._mw_recip_sp = value
        
    @auto_save.setter
    def auto_save(self, value: bool) -> None:
        """If the simulation should automatically be saved upon a detected abortion of the simulation.
      
        """
        self._auto_save = value
        
    @save_after_sim.setter
    def save_after_sim(self, value: bool) -> None:
        """It the simulation should be saved only if a simulation is completed.

        """
        self._save_after_sim = value
    
DEFAULT_SETTINGS = Settings()