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

from ...mesher import Mesher
from emsutil import Material
from ...mesh3d import Mesh3D
from ...coord import Line
from ...geometry import GeoSurface, GeoVolume
from ...elements.femdata import FEMBasis
from ...elements.nedelec2 import Nedelec2
from ...solver import DEFAULT_ROUTINE, SolveRoutine
from ...system import called_from_main_function
from ...selection import FaceSelection
from ...settings import Settings
from ...simstate import SimState
from ...logsettings import DEBUG_COLLECTOR

from .microwave_bc import MWBoundaryConditionSet, PEC, ModalPort, LumpedPort, PortBC
from .microwave_data import MWData
from .assembly.assembler import Assembler
from .port_functions import compute_avg_power_flux
from .simjob import SimJob

from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from typing import Callable, Literal, Any
import multiprocessing as mp
from cmath import sqrt as csqrt
from itertools import product
import numpy as np
import threading
import time
from collections import defaultdict

class SimulationError(Exception):
    pass

def run_job_multi(job: SimJob) -> SimJob:
    """The job launcher for Multi-Processing environements

    Args:
        job (SimJob): The Simulation Job

    Returns:
        SimJob: The solved SimJob
    """
    nr = int(mp.current_process().name.split('-')[1])
    routine = DEFAULT_ROUTINE._configure_routine('MP', proc_nr=nr)
    for A, b, ids, reuse, aux in job.iter_Ab():
        solution, report = routine.solve(A, b, ids, reuse, id=job.id)
        report.add(**aux)
        job.submit_solution(solution, report)
    return job


def _dimstring(data: list[float] | np.ndarray) -> str:
    """A String formatter for dimensions in millimeters

    Args:
        data (list[float]): The list of floating point dimensions

    Returns:
        str: The formatted string
    """
    return '(' + ', '.join([f'{x*1000:.1f}mm' for x in data]) + ')'

def shortest_path(xyz1: np.ndarray, xyz2: np.ndarray, Npts: int) -> np.ndarray:
    """
    Finds the pair of points (one from xyz1, one from xyz2) that are closest in Euclidean distance,
    and returns a (3, Npts) array of points linearly interpolating between them.

    Parameters:
    xyz1 : np.ndarray of shape (3, N1)
    xyz2 : np.ndarray of shape (3, N2)
    Npts : int, number of points in the output path

    Returns:
    np.ndarray of shape (3, Npts)
    """
    # Compute pairwise distances (N1 x N2)
    diffs = xyz1[:, :, np.newaxis] - xyz2[:, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=0)  # shape (N1, N2)

    # Find indices of closest pair
    i1, i2 = np.unravel_index(np.argmin(dists), dists.shape)
    p1 = xyz1[:, i1]
    p2 = xyz2[:, i2]

    # Interpolate linearly between p1 and p2
    t = np.linspace(0, 1, Npts)
    path = (1 - t) * p1[:, np.newaxis] + t * p2[:, np.newaxis]

    return path
    
class Microwave3D:
    """The Electrodynamics time harmonic physics class.

    This class contains all physics dependent features to perform EM simuation in the time-harmonic
    formulation.

    """
    def __init__(self, state: SimState, mesher: Mesher, settings: Settings, order: int = 2):
        
        self._settings: Settings = settings
        
        self.frequencies: list[float] = []
        self.current_frequency = 0
        self.order: int = order
        self.resolution: float = 0.33
        
        self.mesher: Mesher = mesher
        self._state: SimState = state

        self.assembler: Assembler = Assembler(self._settings)
        self.bc: MWBoundaryConditionSet = MWBoundaryConditionSet(None)
        self.basis: Nedelec2 | None = None
        self.solveroutine: SolveRoutine = DEFAULT_ROUTINE
        self.cache_matrices: bool = True

        ## States
        self._bc_initialized: bool = False
        self._simstart: float = 0.0
        self._simend: float = 0.0
        self._container: dict[str, Any] = dict()
        self._completed: bool = False
        
    @property
    def _params(self) -> dict[str, float]:
        return self._state.params
    
    @property
    def mesh(self) -> Mesh3D:
        return self._state.mesh
    
    @property
    def data(self) -> MWData:
        return self._state.data.mw
    
    def reset(self, _reset_bc: bool = True):
        if _reset_bc:
            self.bc = MWBoundaryConditionSet(None)
        else:
            for bc in self.bc.oftype(ModalPort):
                bc.reset()
            
        self.basis: FEMBasis = None
        self.solveroutine.reset()
        self.assembler.cached_matrices = None

    @property
    def nports(self) -> int:
        """The number of ports in the physics.

        Returns:
            int: The number of ports
        """
        return self.bc.count(PortBC)
    
    def ports(self) -> list[PortBC]:
        """A list of all port boundary conditions.

        Returns:
            list[PortBC]: A list of all port boundary conditions
        """
        return sorted(self.bc.oftype(PortBC), key=lambda x: x.number) # type: ignore
    
    
    def _initialize_bcs(self, surfaces: list[GeoSurface]) -> None:
        """Initializes the boundary conditions to set PEC as all exterior boundaries.
        """
        logger.debug('Initializing boundary conditions.')

        tags = self.mesher.domain_boundary_face_tags
        
        # Assigning surface impedance boundary condition
        if self._settings.mw_2dbc:
            for surf in surfaces:
                if surf.material.cond.scalar(1e9) > self._settings.mw_2dbc_peclim:
                    logger.debug(f'Assinging PEC to {surf}')
                    self.bc.PEC(surf)
                elif surf.material.cond.scalar(1e9) > self._settings.mw_2dbc_lim:
                    self.bc.SurfaceImpedance(surf, surf.material)
                
        
        tags = [tag for tag in tags if tag not in self.bc.assigned(2)]
        
        self.bc.PEC(FaceSelection(tags))
        
        logger.info(f'Adding PEC boundary condition with tags {tags}.')
        if self.mesher.periodic_cell is not None:
            self.mesher.periodic_cell.generate_bcs()
            for bc in self.mesher.periodic_cell.bcs:
                self.bc.assign(bc)

    def set_frequency(self, frequency: float | list[float] | np.ndarray ) -> None:
        """Define the frequencies for the frequency sweep

        Args:
            frequency (float | list[float] | np.ndarray): The frequency points.
        """
        logger.info(f'Setting frequency as {frequency}Hz.')
        if isinstance(frequency, (tuple, list, np.ndarray)):
            self.frequencies = list(frequency)
        else:
            self.frequencies = [frequency,]
            
        # Safety tests
        if len(self.frequencies) > 200:
            DEBUG_COLLECTOR.add_report(f'More than 200 frequency points are detected ({len(self.frequencies)}). This may cause slow simulations. Consider using Vector Fitting to subsample S-parameters.')
        if min(self.frequencies) < 1e6:
            DEBUG_COLLECTOR.add_report(f'A frequency smaller than 1MHz has been detected ({min(self.frequencies)} Hz). Perhaps you forgot to include usints like 1e6 for MHz etc.')
        if max(self.frequencies) > 1e12:
            DEBUG_COLLECTOR.add_report(f'A frequency greater than THz has been detected ({min(self.frequencies)} Hz). Perhaps you double counted frequency units like twice 1e6 for MHz etc.')
        
        self.mesher.max_size = self.resolution * 299792458 / max(self.frequencies)
        self.mesher.min_size = 0.1 * self.mesher.max_size

        logger.debug(f'Setting global mesh size range to: {self.mesher.min_size*1000:.3f}mm - {self.mesher.max_size*1000:.3f}mm')
    
    set_frequencies = set_frequency
    
    def set_frequency_range(self, fmin: float, fmax: float, Npoints: int) -> None:
        """Set the frequency range using the np.linspace syntax

        Args:
            fmin (float): The starting frequency
            fmax (float): The ending frequency
            Npoints (int): The number of points
        """
        self.set_frequency(np.linspace(fmin, fmax, Npoints))
    
    def fdense(self, Npoints: int) -> np.ndarray:
        """Return a resampled version of the current frequency range

        Args:
            Npoints (int): The new number of points

        Returns:
            np.ndarray: The new frequency axis
        """
        if len(self.frequencies) == 1:
            raise ValueError('Only 1 frequency point known. At least two need to be defined.')
        fmin = min(self.frequencies)
        fmax = max(self.frequencies)
        return np.linspace(fmin, fmax, Npoints)
    
    def set_resolution(self, resolution: float) -> None:
        """Define the simulation resolution as the fraction of the wavelength.

        To define the wavelength as ¼λ, call .set_resolution(0.25)

        Args:
            resolution (float): The desired wavelength fraction.
            
        """
        if resolution > 0.5:
            logger.warning('Resolutions greater than 0.5 cannot yield accurate results, capping resolution to 0.4')
            resolution = 0.4
        elif resolution > 0.334:
            logger.warning('A resolution greater than 0.33 may cause accuracy issues.')
        
        self.resolution = resolution
        logger.trace(f'Resolution set to {self.resolution}')

    def set_conductivity_limit(self, condutivity: float) -> None:
        """Sets the limit of a material conductivity value beyond which
        the assembler considers it PEC. By default this value is
        set to 1·10⁷S/m which means copper conductivity is ignored.

        Args:
            condutivity (float): The conductivity level in S/m
        """
        if condutivity < 0:
            raise ValueError('Conductivity values must be above 0. Ignoring assignment')

        self.assembler.settings.mw_2dbc_peclim = condutivity
        self.assembler.settings.mw_3d_peclim = condutivity
        logger.trace(f'Set conductivity limit to {condutivity} S/m')
    
    def get_discretizer(self) -> Callable:
        """Returns a discretizer function that defines the maximum mesh size.

        Returns:
            Callable: The discretizer function
        """
        def disc(material: Material):
            return 299792458/(max(self.frequencies) * np.real(material.neff(max(self.frequencies))))
        return disc
    
    def _initialize_field(self):
        """Initializes the physics basis to the correct FEMBasis object.
        
        Currently it defaults to Nedelec2. Mixed basis are used for modal analysis. 
        This function does not have to be called by the user. Its automatically invoked.
        """
        if self.basis is not None:
            return
        if self.order == 1:
            raise NotImplementedError('Nedelec 1 is currently not supported')
            from ...elements import Nedelec1
            self.basis = Nedelec1(self.mesh)
        elif self.order == 2:
            from ...elements.nedelec2 import Nedelec2
            self.basis = Nedelec2(self.mesh)

    def _initialize_bc_data(self):
        ''' Initializes auxilliary required boundary condition information before running simulations.
        '''
        logger.debug('Initializing boundary conditions')
        self.bc.cleanup()
        for port in self.bc.oftype(LumpedPort):
            self.define_lumped_port_integration_points(port)
    
    def _check_physics(self) -> None:
        """ Executes a physics check before a simulation can be run."""
        if not self.bc._is_excited():
            raise SimulationError('The simulation has no boundary conditions that insert energy. Make sure to include at least one Port into your simulation.')
    
    def define_lumped_port_integration_points(self, port: LumpedPort) -> None:
        """Sets the integration points on Lumped Port objects for voltage integration

        Args:
            port (LumpedPort): The LumpedPort object

        Raises:
            SimulationError: An error if there are no nodes associated with the port.
        """
        if len(port.vintline) > 0:
            return
        logger.debug(' - Finding Lumped Port integration points')
        field_axis = port.Vdirection.np

        points = self.mesh.get_nodes(port.tags)

        if points.size==0:
            raise SimulationError(f'The lumped port {port} has no nodes associated with it')
        
        xs = self.mesh.nodes[0,points]
        ys = self.mesh.nodes[1,points]
        zs = self.mesh.nodes[2,points]

        dotprod = xs*field_axis[0] + ys*field_axis[1] + zs*field_axis[2]

        start_id = np.argwhere(dotprod == np.min(dotprod)).flatten()
        
        xs = xs[start_id]
        ys = ys[start_id]
        zs = zs[start_id]
        

        for x,y,z in zip(xs, ys, zs):
            start = np.array([x,y,z])
            end = start + port.Vdirection.np*port.height
            port.vintline.append(Line.from_points(start, end, 21))
            logger.trace(f' - Port[{port.port_number}] integration line {start} -> {end}.')
        
        port.v_integration = True

    def _find_tem_conductors(self, port: ModalPort, sigtri: np.ndarray) -> tuple[list[int], list[int]]:
        ''' Returns two lists of global node indices corresponding to the TEM port conductors.
        
        This method is invoked during modal analysis with TEM modes. It looks at all edges
        exterior to the boundary face triangulation and finds two small subsets of nodes that
        lie on different exterior boundaries of the boundary face.

        Args:
            port (ModalPort): The modal port object.
            
        Returns:
            list[int]: A list of node integers of island 1.
            list[int]: A list of node integers of island 2.
        '''
        if self.basis is None:
            raise ValueError('The field basis is not yet defined.')

        logger.debug(' - Finding PEC TEM conductors')
        mesh = self.mesh
        
        # Find all BC conductors
        pecs: list[PEC] = self.bc.get_conductors() # type: ignore

        # Process all PEC Boundary Conditions
        pec_edges = []
        for pec in pecs:
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
            pec_edges.extend(edge_ids)
        
        # Process conductivity
        for itri in mesh.get_triangles(port.tags):
            if sigtri[itri] > 1e6:
                edge_ids = list(mesh.tri_to_edge[:,itri].flatten())
                pec_edges.extend(edge_ids)

        # All PEC edges
        pec_edges = list(set(pec_edges))
        
        # Port mesh data
        tri_ids = mesh.get_triangles(port.tags)
        edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
        
        port_pec_edges = np.array([i for i in pec_edges if i in set(edge_ids)])
        
        pec_islands = mesh.find_edge_groups(port_pec_edges)

        
        logger.debug(f' - Found {len(pec_islands)} PEC islands.')

        if len(pec_islands) != 2:
            pec_island_tags = {i: self.mesh._get_dimtags(edges=pec_edge_group) for i,pec_edge_group in enumerate(pec_islands)}
            plus_term = None
            min_term = None
            
            for i, dimtags in pec_island_tags.items():
                if not set(dimtags).isdisjoint(port.plus_terminal):
                    plus_term = i
                
                if not set(dimtags).isdisjoint(port.minus_terminal):
                    min_term = i
            
            if plus_term is None or min_term is None:
                logger.error(f' - Found {len(pec_islands)} PEC islands without a terminal definition. Please use .set_terminals() to define which conductors are which polarity, or define the integration line manually.') 
                return None, None  
            logger.debug(f'Positive island = {pec_island_tags[plus_term]}')
            logger.debug(f'Negative island = {pec_island_tags[min_term]}')
            pec_islands = [pec_islands[plus_term], pec_islands[min_term]]
        
        groups = []
        for island in pec_islands:
            group = set()
            for edge in island:
                group.add(mesh.edges[0,edge])
                group.add(mesh.edges[1,edge])
            groups.append(sorted(list(group)))
        
        group1 = groups[0]
        group2 = groups[1]

        return group1, group2
    
    def _compute_modes(self, freq: float):
        """Compute the modal port modes for a given frequency. Used internally by the frequency domain study.

        Args:
            freq (float): The simulation frequency
        """
        for bc in self.bc.oftype(ModalPort):
            
            # If there is a port mode (at least one) and the port does not have mixed materials. No new analysis is needed
            if not bc.mixed_materials and bc.initialized:
                continue
            
            if bc.forced_modetype=='TEM':
                TEM = True
            else:
                TEM = False
            self.modal_analysis(bc, 1, direct=False, freq=freq, TEM=TEM)

    def modal_analysis(self, 
                       port: ModalPort, 
                       nmodes: int = 6, 
                       direct: bool = True,
                       TEM: bool = False,
                       target_kz = None,
                       target_neff = None,
                       freq: float | None = None) -> None:
        ''' Execute a modal analysis on a given ModalPort boundary condition.
        
        Parameters:
        -----------
            port : ModalPort
                The port object to execute the analysis for.
            direct : bool
                Whether to use the direct solver (LAPACK) if True. Otherwise it uses the iterative
                ARPACK solver. The ARPACK solver required an estimate for the propagation constant and is faster
                for a large number of Degrees of Freedom.
            TEM : bool = True
                Whether to estimate the propagation constant assuming its a TEM transmisison line.
            target_k0 : float
                The expected propagation constant to find a mode for (direct = False).
            target_neff : float
                The expected effective mode index defined as kz/k0 (1.0 = free space, <1 = TE/TM, >1=slow wavees)
            freq : float = None
                The desired frequency at which the mode is solved. If None then it uses the lowest frequency of the provided range.
        '''
        T0 = time.time()
        logger.info(f'Starting Mode Analysis for port {port}.')
        
        if self.bc._initialized is False:
            raise SimulationError('Cannot run a modal analysis because no boundary conditions have been assigned.')
        
        self._initialize_field()
        self._initialize_bc_data()

        if self.basis is None:
            raise SimulationError('Cannot proceed, the current basis class is undefined.')

        logger.debug(' - retreiving material properties.')
        
        if freq is None:
            freq = self.frequencies[0]
        
        materials = self._get_material_assignment(self.mesher.volumes)

        ertet = np.zeros((3,3,self.mesh.n_tets), dtype=np.complex128)
        tandtet = np.zeros((3,3,self.mesh.n_tets), dtype=np.complex128)
        urtet = np.zeros((3,3,self.mesh.n_tets), dtype=np.complex128)
        condtet = np.zeros((3,3,self.mesh.n_tets), dtype=np.complex128)
        
        for mat in materials:
            ertet = mat.er(freq, ertet)
            tandtet = mat.tand(freq, tandtet)
            urtet = mat.ur(freq, urtet)
            condtet = mat.cond(freq, condtet)
        
        ertet = ertet * (1-1j*tandtet)
        
        er = np.zeros((3,3,self.mesh.n_tris,), dtype=np.complex128)
        ur = np.zeros((3,3,self.mesh.n_tris,), dtype=np.complex128)
        cond = np.zeros((self.mesh.n_tris,), dtype=np.complex128)

        for itri in range(self.mesh.n_tris):
            itet = self.mesh.tri_to_tet[0,itri]
            er[:,:,itri] = ertet[:,:,itet]
            ur[:,:,itri] = urtet[:,:,itet]
            cond[itri] = condtet[0,0,itet]

        itri_port = self.mesh.get_triangles(port.tags)

        ermean = np.mean(er[er>0].flatten()[itri_port])
        urmean = np.mean(ur[ur>0].flatten()[itri_port])
        ermax = np.max(er[:,:,itri_port].flatten())
        urmax = np.max(ur[:,:,itri_port].flatten())

        k0 = 2*np.pi*freq/299792458
        
        logger.debug(f' - mean(max): εr = {ermean:.2f}({ermax:.2f}), μr = {urmean:.2f}({urmax:.2f})')
        
        Amatrix, Bmatrix, solve_ids, nlf = self.assembler.assemble_bma_matrices(self.basis, er, ur, cond, k0, port, self.bc)
        
        logger.debug(f'Total of {Amatrix.shape[0]} Degrees of freedom.')
        logger.debug(f'Applied frequency: {freq/1e9:.2f}GHz')
        logger.debug(f'K0 = {k0} rad/m')

        F = -1

        if target_neff is not None:
            target_kz = k0*target_neff
        
        if target_kz is None:
            if TEM or port.forced_modetype=='TEM':
                target_kz = ermean*urmean*1.1*k0
            else:
                
                target_kz = ermean*urmean*0.7*k0
                
        logger.debug(f'Solving for {solve_ids.shape[0]} degrees of freedom.')

        eigen_values, eigen_modes, report = self.solveroutine.eig_boundary(Amatrix, Bmatrix, solve_ids, nmodes, direct, target_kz, sign=-1)
        
        logger.debug(f'Eigenvalues: {np.sqrt(F*eigen_values)} rad/m')

        port._er = er
        port._ur = ur

        nmodes_found = eigen_values.shape[0]

        for i in range(nmodes_found):
            
            Emode = np.zeros((nlf.n_field,), dtype=np.complex128)
            eigenmode = eigen_modes[:,i]
            Emode[solve_ids] = np.squeeze(eigenmode)
            Emode = Emode * np.exp(-1j*np.angle(Emode[np.argmax(np.abs(Emode))]))

            beta_base = np.emath.sqrt(-eigen_values[i])
            beta = min(k0*np.sqrt(ermax*urmax), beta_base)

            residuals = -1

            if port._get_alignment_vector(i) is not None:
                vec = port._get_alignment_vector(i)
                xyz_centers = self.mesh.tri_centers[:,self.mesh.get_triangles(port.tags)]
                E_centers = np.mean(nlf.interpolate_Ef(Emode)(xyz_centers[0,:], xyz_centers[1,:], xyz_centers[2,:]), axis=1)
                EdotVec = vec[0]*E_centers[0] + vec[1]*E_centers[1] + vec[2]*E_centers[2]
                if EdotVec.real < 0:
                    logger.debug(f'Mode polarization along alignment axis {vec} = {EdotVec.real:.3f}, inverting.')
                    Emode = -Emode
                  
            portfE = nlf.interpolate_Ef(Emode)
            portfH = nlf.interpolate_Hf(Emode, k0, ur, beta)
            P = compute_avg_power_flux(nlf, Emode, k0, ur, beta)

            mode = port.add_mode(Emode, portfE, portfH, beta, k0, residuals, number=i, freq=freq)
            
            if mode is None:
                continue
            
            Efxy = Emode[:nlf.n_xy]
            Efz = Emode[nlf.n_xy:]
            Ez = np.max(np.abs(Efz))
            Exy = np.max(np.abs(Efxy))
            
            if port.forced_modetype == 'TEM' or TEM:
                mode.modetype = 'TEM'
                
                if len(port.vintline)>0:
                    line = port.vintline[0]
                else:  
                    G1, G2 = self._find_tem_conductors(port, sigtri=cond)
                    if G1 is None or G2 is None:
                        logger.warning('Skipping characteristic impedance calculation.')
                        continue
                    
                    nodes1 = self.mesh.nodes[:,G1]
                    nodes2 = self.mesh.nodes[:,G2]
                    path = shortest_path(nodes1, nodes2, 2)
                    line = Line.from_points(path[:,0], path[:,1], 21)
                    port.vintline.append(line)
                
                cs = np.array(line.cmid)

                logger.debug(f'Integrating portmode from {cs[:,0]} to {cs[:,-1]}')
                voltage = line.line_integral(portfE)
                # Align mode polarity to positive voltage
                if voltage < 0:
                    mode.polarity = mode.polarity * -1
                    
                mode.Z0 = abs(voltage**2/(2*P))
                logger.debug(f'Port Z0 = {mode.Z0}')
            elif Ez/Exy < 1e-1 or port.forced_modetype=='TE':
                logger.debug('Low Ez/Et ratio detected, assuming TE mode')
                mode.modetype = 'TE'
            elif Ez/Exy > 1e-1 or port.forced_modetype=='TM':
                logger.debug('High Ez/Et ratio detected, assuming TM mode')
                mode.modetype = 'TM'
            

            mode.set_power(P*port._qmode(k0)**2)
        
        port.sort_modes()

        logger.info(f'Total of {port.nmodes} found')

        T2 = time.time()    
        logger.info(f'Elapsed time = {(T2-T0):.2f} seconds.')
        return None
    
    def run_sweep(self, 
                parallel: bool = False,
                n_workers: int = 2, 
                harddisc_threshold: int | None = None,
                harddisc_path: str = 'EMergeSparse',
                frequency_groups: int = -1,
                multi_processing: bool = False,
                automatic_modal_analysis: bool = True) -> MWData:
        """Executes a frequency domain study

        The study is distributed over "n_workers" workers.
        As optional parameter you may set a harddisc_threshold as integer. This determines the maximum
        number of degrees of freedom before which the jobs will be cahced to the harddisk. The
        path that will be used to cache the sparse matrices can be specified.
        Additionally the term frequency_groups may be specified. This number will define in how
        many groups the matrices will be pre-computed before they are send to workers. This can minimize
        the total amound of RAM memory used. For example with 11 frequencies in gruops of 4, the following
        frequency indices will be precomputed and then solved: [[1,2,3,4],[5,6,7,8],[9,10,11]]

        Args:
            n_workers (int, optional): The number of workers. Defaults to 2.
            harddisc_threshold (int, optional): The number of DOF limit. Defaults to None.
            harddisc_path (str, optional): The cached matrix path name. Defaults to 'EMergeSparse'.
            frequency_groups (int, optional): The number of frequency points in a solve group. Defaults to -1.
            automatic_modal_analysis (bool, optional): Automatically compute port modes. Defaults to False.
            multi_processing (bool, optional): Whether to use multiprocessing instead of multi-threaded (slower on most machines).

        Raises:
            SimulationError: An error associated witha a problem during the simulation.

        Returns:
            MWSimData: The dataset.
        """
        self._completed = False
        self._simstart = time.time()
        if self.bc._initialized is False:
            raise SimulationError('Cannot run a modal analysis because no boundary conditions have been assigned.')
        
        self._initialize_field()
        self._initialize_bc_data()
        self._check_physics()
        
        if self.basis is None:
            raise SimulationError('Cannot proceed, the simulation basis class is undefined.')

        materials = self._get_material_assignment(self.mesher.volumes)

        ### Does this move
        logger.debug('Initializing frequency domain sweep.')
        
        #### Port settings
        all_ports = self.bc.oftype(PortBC)

        ##### FOR PORT SWEEP SET ALL ACTIVE TO FALSE. THIS SHOULD BE FIXED LATER
        ### COMPUTE WHICH TETS ARE CONNECTED TO PORT INDICES

        for port in all_ports:
            port.active=False
            

        logger.info(f'Pre-assembling matrices of {len(self.frequencies)} frequency points.')

        thread_local = None
        if parallel:
            # Thread-local storage for per-thread resources
            thread_local = threading.local()

        ## DEFINE SOLVE FUNCTIONS
        def get_routine():
            if not hasattr(thread_local, "routine"):
                worker_nr = int(threading.current_thread().name.split('_')[1])+1
                thread_local.routine = self.solveroutine.duplicate()._configure_routine('MT', thread_nr=worker_nr)
            return thread_local.routine

        def run_job(job: SimJob):
            routine = get_routine()
            for A, b, ids, reuse, aux in job.iter_Ab():
                solution, report = routine.solve(A, b, ids, reuse, id=job.id)
                report.add(**aux)
                job.submit_solution(solution, report)
            return job
        
        def run_job_single(job: SimJob):
            for A, b, ids, reuse, aux in job.iter_Ab():
                solution, report = self.solveroutine.solve(A, b, ids, reuse, id=job.id)
                report.add(**aux)
                job.submit_solution(solution, report)
            return job
        
        ## GROUP FREQUENCIES
        # Each frequency group will be pre-assembled before submitting them to the parallel pool
        freq_groups = []
        if frequency_groups == -1:
            freq_groups=[self.frequencies,]
        else:
            n = frequency_groups
            freq_groups = [self.frequencies[i:i+n] for i in range(0, len(self.frequencies), n)]

        results: list[SimJob] = []
        matset: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        
        logger.trace(f'Frequency groups: {freq_groups}')
        ## Single threaded
        job_id = 1

        self._compute_modes(sum(self.frequencies)/len(self.frequencies))

        if not parallel:
            # ITERATE OVER FREQUENCIES
            freq_groups
            for i_group, fgroup in enumerate(freq_groups):
                logger.info(f'Precomputing group {i_group}.')
                jobs = []
                ## Assemble jobs
                for ifreq, freq in enumerate(fgroup):
                    logger.debug(f'Simulation frequency = {freq/1e9:.3f} GHz') 
                    if automatic_modal_analysis:
                        self._compute_modes(freq)
                    job, mats = self.assembler.assemble_freq_matrix(self.basis, materials, 
                                                            self.bc.boundary_conditions, 
                                                            freq, 
                                                            cache_matrices=self.cache_matrices)
                    job.store_limit = harddisc_threshold
                    job.relative_path = harddisc_path
                    job.id = job_id
                    job_id += 1
                    jobs.append(job)
                    matset.append(mats)
                
                logger.info(f'Starting single threaded solve of {len(jobs)} jobs.')
                group_results = [run_job_single(job) for job in jobs]
                results.extend(group_results)
        elif not multi_processing:
             # MULTI THREADED
            with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix='WKR') as executor:
                # ITERATE OVER FREQUENCIES
                for i_group, fgroup in enumerate(freq_groups):
                    logger.info(f'Precomputing group {i_group}.')
                    jobs = []
                    ## Assemble jobs
                    for freq in fgroup:
                        logger.debug(f'Simulation frequency = {freq/1e9:.3f} GHz') 
                        if automatic_modal_analysis:
                            self._compute_modes(freq)
                        job, mats = self.assembler.assemble_freq_matrix(self.basis, materials, 
                                                                self.bc.boundary_conditions, 
                                                                freq, 
                                                                cache_matrices=self.cache_matrices)
                        job.store_limit = harddisc_threshold
                        job.relative_path = harddisc_path
                        job.id = job_id
                        job_id += 1
                        jobs.append(job)
                        matset.append(mats)
                    
                    logger.info(f'Starting distributed solve of {len(jobs)} jobs with {n_workers} threads.')
                    group_results = list(executor.map(run_job, jobs))
                    results.extend(group_results)
                executor.shutdown()
        else:
            ### MULTI PROCESSING
            # Check for if __name__=="__main__" Guard
            if not called_from_main_function():
                raise SimulationError(
                    "Multiprocess support must be launched from your "
                    "if __name__ == '__main__' guard in the top-level script."
                )
            # Start parallel pool
            with mp.Pool(processes=n_workers) as pool:
                for i_group, fgroup in enumerate(freq_groups):
                    logger.debug(f'Precomputing group {i_group}.')
                    jobs = []
                    # Assemble jobs
                    for freq in fgroup:
                        logger.debug(f'Simulation frequency = {freq/1e9:.3f} GHz')
                        if automatic_modal_analysis:
                            self._compute_modes(freq)
                        
                        job, mats = self.assembler.assemble_freq_matrix(
                            self.basis, materials,
                            self.bc.boundary_conditions,
                            freq,
                            cache_matrices=self.cache_matrices
                        )

                        job.store_limit = harddisc_threshold
                        job.relative_path = harddisc_path
                        job.id = job_id
                        job_id += 1
                        jobs.append(job)
                        matset.append(mats)

                    logger.info(
                        f'Starting distributed solve of {len(jobs)} jobs '
                        f'with {n_workers} processes in parallel'
                    )
                    # Distribute taks
                    group_results = pool.map(run_job_multi, jobs)
                    results.extend(group_results)
        if parallel:
            thread_local.__dict__.clear()
        logger.info('Solving complete')

        for freq, job in zip(self.frequencies, results):
            self.data.setreport(job.reports, freq=freq, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f'Sim variable: {variables}')
            for item in data['report']:
                item.logprint(logger.trace)

        self.solveroutine.reset()
        ### Compute S-parameters and return
        self._post_process(results, matset)
        self._completed = True
        return self.data
    
    def _get_material_assignment(self, volumes: list[GeoVolume]) -> list[Material]:
        '''Retrieve the material properties of the geometry'''
        
        # Reset index assingments
        for vol in volumes:
            vol.material.reset()
        
        # collect all materials
        materials = []
        assignment_dict: dict[int, list[GeoVolume]] = defaultdict(list)
        i = 0
        for vol in volumes:
            for tag in vol.tags:
                assignment_dict[tag].append(vol)
            if vol.material not in materials:
                materials.append(vol.material)
                vol.material._hash_key = i
                i += 1
        
        # Check competing priorities!
        for domaintag, volumelist in assignment_dict.items():
            priolist = [vol._priority for vol in volumelist]
            maxprio = max(priolist)
            if priolist.count(maxprio) > 1:
                vols = [vol for vol in volumelist if vol._priority==maxprio]
                logger.warning(f'Domain with tag {domaintag} has multiple geometries imposing a material to them: {vols}. Consider setting priorities to decide which volume is more important.')
                DEBUG_COLLECTOR.add_report(f'Domain with tag {domaintag} has multiple geometries imposing a material to them: {vols}. Consider setting priorities to decide which volume is more important.')
            
        xs = self.mesh.centers[0,:]
        ys = self.mesh.centers[1,:]
        zs = self.mesh.centers[2,:]
        
        matassign = -1*np.ones((self.mesh.n_tets,), dtype=np.int64)
        
        
        for volume in sorted(volumes, key=lambda x: x._priority):
        
            for dimtag in volume.dimtags:
                
                tet_ids = self.mesh.get_tetrahedra(dimtag[1])
                
                matassign[tet_ids] = volume.material._hash_key
        
        if np.any(matassign==-1):
            raise SimulationError(f'Tetrahedra detected with unassigned materials: {np.argwhere(matassign==-1)}')
        
        for mat in materials:
            ids = np.argwhere(matassign==mat._hash_key).flatten()
            mat.initialize(xs[ids], ys[ids], zs[ids], ids)
                    
        
        return materials
    
    def _run_adaptive_mesh(self,
                iteration: int, 
                frequency: float,
                automatic_modal_analysis: bool = True) -> tuple[MWData, list[int]]:
        """Executes a frequency domain study

        The study is distributed over "n_workers" workers.
        As optional parameter you may set a harddisc_threshold as integer. This determines the maximum
        number of degrees of freedom before which the jobs will be cahced to the harddisk. The
        path that will be used to cache the sparse matrices can be specified.
        Additionally the term frequency_groups may be specified. This number will define in how
        many groups the matrices will be pre-computed before they are send to workers. This can minimize
        the total amound of RAM memory used. For example with 11 frequencies in gruops of 4, the following
        frequency indices will be precomputed and then solved: [[1,2,3,4],[5,6,7,8],[9,10,11]]

        Args:
            iteration (int): The iteration number
            frequency (float): The simulation frequency

        Raises:
            SimulationError: An error associated witha a problem during the simulation.

        Returns:
            MWSimData: The dataset.
        """
        
        self._simstart = time.time()
        if self.bc._initialized is False:
            raise SimulationError('Cannot run a modal analysis because no boundary conditions have been assigned.')
        
        self._initialize_field()
        self._initialize_bc_data()
        self._check_physics()
        
        if self.basis is None:
            raise SimulationError('Cannot proceed, the simulation basis class is undefined.')

        materials = self._get_material_assignment(self.mesher.volumes)

        ### Does this move
        logger.debug('Initializing single frequency settings.')
        
        #### Port settings
        all_ports = self.bc.oftype(PortBC)

        ##### FOR PORT SWEEP SET ALL ACTIVE TO FALSE. THIS SHOULD BE FIXED LATER
        ### COMPUTE WHICH TETS ARE CONNECTED TO PORT INDICES

        for port in all_ports:
            port.active=False
    
    
        def run_job_single(job: SimJob):
            for A, b, ids, reuse, aux in job.iter_Ab():
                solution, report = self.solveroutine.solve(A, b, ids, reuse, id=job.id)
                report.add(**aux)
                job.submit_solution(solution, report)
            return job
        
        
        #matset: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []     

        self._compute_modes(frequency)

        logger.debug(f'Simulation frequency = {frequency/1e9:.3f} GHz') 
        
        #if automatic_modal_analysis:
        #    self._compute_modes(frequency)
            
        job, mats = self.assembler.assemble_freq_matrix(self.basis, materials, 
                                                self.bc.boundary_conditions, 
                                                frequency, 
                                                cache_matrices=self.cache_matrices)

        job.id = 0

        logger.info('Starting solve')
        job = run_job_single(job)

    
        logger.info('Solving complete')

        self.data.setreport(job.reports, freq=frequency, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f'Sim variable: {variables}')
            for item in data['report']:
                item.logprint(logger.trace)

        self.solveroutine.reset()
        ### Compute S-parameters and return
        self._post_process([job,], [mats,])
        return self.data, job._pec_tris
    
    def eigenmode(self, search_frequency: float,
                        nmodes: int = 6,
                        k0_limit: float = 1,
                        direct: bool = False,
                        deep_search: bool = False,
                        mode: Literal['LM','LR','SR','LI','SI']='LM') -> MWData:
        """Executes an eigenmode study

       

        Args:
            search_frequency (float): The frequency around which you would like to search
            nmodes (int, optional): The number of jobs. Defaults to 6.
            k0_limit (float): The lowest k0 value before which a mode is considered part of the null space. Defaults to 1e-3
        Raises:
            SimulationError: An error associated witha a problem during the simulation.

        Returns:
            MWSimData: The dataset.
        """
        
        self._simstart = time.time()
        if self.bc._initialized is False:
            raise SimulationError('Cannot run a modal analysis because no boundary conditions have been assigned.')
        
        self._initialize_field()
        self._initialize_bc_data()
        
        if self.basis is None:
            raise SimulationError('Cannot proceed. The simulation basis class is undefined.')

        materials = self._get_material_assignment(self.mesher.volumes)
        
        ### Does this move
        logger.debug('Initializing frequency domain sweep.')
            
        logger.info(f'Pre-assembling matrices of {len(self.frequencies)} frequency points.')
        
        job, matset = self.assembler.assemble_eig_matrix(self.basis, materials, 
                                                            self.bc.boundary_conditions, search_frequency)
        
        er, ur, cond = matset
        logger.info('Solving complete')

        A, C, solve_ids = job.yield_AC()

        target_k0 = 2*np.pi*search_frequency/299792458

        eigen_values, eigen_modes, report = self.solveroutine.eig(A, C, solve_ids, nmodes, direct, target_k0, which=mode)

        eigen_modes = job.fix_solutions(eigen_modes)

        logger.debug(f'Eigenvalues: {np.sqrt(eigen_values)} rad/m')

        nmodes_found = eigen_values.shape[0]

        for i in range(nmodes_found):
            

            eig_k0 = np.sqrt(eigen_values[i])
            if eig_k0 < k0_limit:
                logger.debug(f'Ignoring mode due to low k0: {eig_k0} < {k0_limit}')
                continue
            eig_freq = eig_k0*299792458/(2*np.pi)

            logger.debug(f'Found k0={eig_k0:.2f}, f0={eig_freq/1e9:.2f} GHz')
            Emode = eigen_modes[:,i]

            scalardata = self.data.scalar.new(**self._params)
            scalardata.k0 = eig_k0
            scalardata.freq = eig_freq

            fielddata = self.data.field.new(**self._params)
            fielddata.freq = eig_freq
            fielddata._der = np.squeeze(er[0,0,:])
            fielddata._dur = np.squeeze(ur[0,0,:])
            fielddata._dsig = np.squeeze(cond[0,0,:])
            fielddata._mode_field = Emode
            fielddata.basis = self.basis
        ### Compute S-parameters and return
        
        return self.data

    def _post_process(self, results: list[SimJob], materials: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """Compute the S-parameters after Frequency sweep

        Args:
            results (list[SimJob]): The set of simulation results
            er (np.ndarray): The domain εᵣ
            ur (np.ndarray): The domain μᵣ
            cond (np.ndarray): The domain conductivity
        """
        if self.basis is None:
            raise SimulationError('Cannot post-process. Simulation basis function is undefined.')
        
        mesh = self.mesh
        all_ports = self.bc.oftype(PortBC)
        port_numbers = [port.port_number for port in all_ports]
        
        logger.info('Computing S-parameters')
        
        not_conserved = False
        conserve_margin = 0.0
        
        single_corr = self._settings.mw_cap_sp_single
        col_corr = self._settings.mw_cap_sp_col
        recip_corr = self._settings.mw_recip_sp
        
        for job, mats in zip(results, materials):
            freq = job.freq
            er, ur, cond = mats
            ertri = np.zeros((3,3,self.mesh.n_tris), dtype=np.complex128)
            urtri = np.zeros((3,3,self.mesh.n_tris), dtype=np.complex128)
            condtri = np.zeros((self.mesh.n_tris,), dtype=np.complex128)

            er_scal = (er[0,0,:] + er[1,1,:] + er[2,2,:])/3
            ur_scal = (ur[0,0,:] + ur[1,1,:] + ur[2,2,:])/3
            cond_scal = (cond[0,0,:] + cond[1,1,:] + cond[2,2,:])/3
            
            for itri in range(self.mesh.n_tris):
                itet = self.mesh.tri_to_tet[0,itri]
                ertri[:,:,itri] = er[:,:,itet]
                urtri[:,:,itri] = ur[:,:,itet]
                condtri[itri] = cond[0,0,itet]
                
            k0 = 2*np.pi*freq/299792458

            scalardata = self.data.scalar.new(freq=freq, **self._params)
            scalardata.k0 = k0
            scalardata.freq = freq
            scalardata.init_sp(port_numbers) # type: ignore
            
            fielddata = self.data.field.new(freq=freq, **self._params)
            fielddata.freq = freq
            fielddata._der = np.squeeze(er_scal)
            fielddata._dur = np.squeeze(ur_scal)
            fielddata._dsig = np.squeeze(cond_scal)

            logger.info(f'Post Processing simulation frequency = {freq/1e9:.3f} GHz') 

            # Recording port information
            for active_port in all_ports:
                port_tets = self.mesh.get_face_tets(active_port.tags)
                
                fielddata.add_port_properties(active_port.port_number,
                                         mode_number=active_port.mode_number,
                                         k0 = k0,
                                         beta = active_port.get_beta(k0),
                                         Z0 = active_port.portZ0(k0),
                                         Pout = active_port.power)
                scalardata.add_port_properties(active_port.port_number,
                                         mode_number=active_port.mode_number,
                                         k0 = k0,
                                         beta = active_port.get_beta(k0),
                                         Z0 = active_port.portZ0(k0),
                                         Pout= active_port.power)

                # Set port as active and add the port mode to the forcing vector
                active_port.active = True
                
                solution = job._fields[active_port.port_number]

                fielddata._fields = job._fields
                fielddata.basis = self.basis
                # Compute the S-parameters
                # Define the field interpolation function
                fieldf = self.basis.interpolate_Ef(solution, tetids=port_tets)

                # Active port power
                tris = mesh.get_triangles(active_port.tags)
                tri_vertices = mesh.tris[:,tris]
                EdotF_act, EdotE_act = self._compute_s_data(active_port, fieldf, tri_vertices, k0, ertri[:,:,tris], urtri[:,:,tris])
                logger.debug(f'[{active_port.port_number}] Active port amplitude = {np.abs(EdotF_act):.3f} (Excitation = {np.abs(EdotE_act):.2f})')
                Amp_act = np.sqrt(active_port.power)
                
                #Passive ports
                for bc in all_ports:
                    port_tets = self.mesh.get_face_tets(bc.tags)
                    fieldf = self.basis.interpolate_Ef(solution, tetids=port_tets)
                    tris = mesh.get_triangles(bc.tags)
                    tri_vertices = mesh.tris[:,tris]
                    EdotF_pas, EdotE_pas = self._compute_s_data(bc, fieldf,tri_vertices, k0, ertri[:,:,tris], urtri[:,:,tris])
                    Amp_pas = EdotF_pas/EdotE_pas
                    logger.debug(f'[{bc.port_number}] Passive amplitude = {np.abs(EdotF_pas):.3f}')
                    scalardata.write_S(bc.port_number, active_port.port_number, Amp_pas/Amp_act)
                    if abs(Amp_pas/Amp_act) > 1.0:
                        logger.warning(f'S-parameter ({bc.port_number},{active_port.port_number}) > 1.0 detected: {np.abs(Amp_pas/Amp_act)}')
                        not_conserved = True
                        conserve_margin = abs(Amp_pas/Amp_act) - 1.0
                active_port.active=False
            
            
            fielddata.set_field_vector()
            
            N = scalardata.Sp.shape[1]
            
            # Enforce reciprocity
            if recip_corr:
                scalardata.Sp = (scalardata.Sp + scalardata.Sp.T)/2
            
            # Enforce energy conservation
            if col_corr:
                for j in range(N):
                    scalardata.Sp[:,j] = scalardata.Sp[:,j] / max(1.0, np.sum(np.abs(scalardata.Sp[:,j])**2))
            
            # Enforce S-parameter limit to 1.0
            if single_corr:
                for i,j in product(range(N), range(N)):
                    scalardata.Sp[i,j] = scalardata.Sp[i,j] / max(1.0, np.abs(scalardata.Sp[i,j]))
                    
                    

        if not_conserved and conserve_margin > 0.001:
            DEBUG_COLLECTOR.add_report('S-parameters with an amplitude greater than 1.0 detected. This could be due to a ModalPort with the wrong mode type.\n' +
                                       'Specify the type of mode (TE/TM/TEM) in the constructor using ModalPort(..., modetype=\'TE\') for example.')
        if not_conserved and conserve_margin < 0.001:
            DEBUG_COLLECTOR.add_report(f'S-parameters with a total column power slightly greater than 1.0 detected ({20*np.log10(conserve_margin):.2f}dB error).\n' +
                                       'This is compatible with the numerical accuracy of EMerge.')
        logger.info('Simulation Complete!')
        self._simend = time.time()    
        logger.info(f'Elapsed time = {(self._simend-self._simstart):.2f} seconds.')

    
    def _compute_s_data(self, bc: PortBC, 
                       fieldfunction: Callable, 
                       tri_vertices: np.ndarray, 
                       k0: float,
                       erp: np.ndarray,
                       urp: np.ndarray,) -> tuple[complex, complex]:
        """ Computes the S-parameter data for a given boundary condition and field function.

        Args:
            bc (PortBC): The port boundary condition
            fieldfunction (Callable): The field function that interpolates the solution field.
            tri_vertices (np.ndarray): The triangle vertex indices of the port face
            k₀ (float): The simulation phase constant
            erp (np.ndarray): The εᵣ of the port face triangles
            urp (np.ndarray): The μᵣ of the port face triangles.

        Returns:
            tuple[complex, complex]: _description_
        """
        from .sparam import sparam_field_power, sparam_mode_power
        
        if bc.v_integration:
            if bc.vintline is None:
                raise SimulationError('Trying to compute characteristic impedance but no integration line is defined.')
            if bc.Z0 is None:
                raise SimulationError('Trying to compute the impedance of a boundary condition with no characteristic impedance.')
            
            Voltages = [line.line_integral(fieldfunction) for line in bc.vintline]
            V = sum(Voltages)/len(Voltages)
            
            if bc.active:
                if bc.voltage is None:
                    raise ValueError('Cannot compute port S-paramer with a None port voltage.')
                a = bc.voltage
                b = (V-bc.voltage)
            else:
                a = bc.voltage
                b = V
            
            a_sig = a*csqrt(1/(2*bc.Z0))
            b_sig = b*csqrt(1/(2*bc.Z0))

            return b_sig, a_sig
        else:
            if bc.modetype(k0) == 'TEM':
                const = 1/(np.sqrt((urp[0,0,:] + urp[1,1,:] + urp[2,2,:])/(erp[0,0,:] + erp[1,1,:] + erp[2,2,:])))
            elif bc.modetype(k0) == 'TE':
                const = 1/((urp[0,0,:] + urp[1,1,:] + urp[2,2,:])/3)
            elif bc.modetype(k0) == 'TM':
                const = 1/((erp[0,0,:] + erp[1,1,:] + erp[2,2,:])/3)
            const = np.squeeze(const)
            field_p = sparam_field_power(self.mesh.nodes, tri_vertices, bc, k0, fieldfunction, const, 4)
            mode_p = sparam_mode_power(self.mesh.nodes, tri_vertices, bc, k0, const, 4)
            return field_p, mode_p


    ############################################################
    #                     DEPRICATED FUNCTIONS                #
    ############################################################

    def frequency_domain(self, *args, **kwargs):
        """DEPRICATED VERSION: Use run_sweep() instead.
        """
        logger.warning('This function is depricated. Please use run_sweep() instead')
        return self.run_sweep(*args, **kwargs)
    
    
class MW3D:
    """The New Electrodynamics time harmonic physics class.

    This class contains all physics dependent features to perform EM simuation in the time-harmonic
    formulation.

    """
    def __init__(self, settings: Settings, order: int = 2):
        
        self._settings: Settings = settings
        self.order: int = order
        self.bc: MWBoundaryConditionSet = MWBoundaryConditionSet(None)
        self.basis: Nedelec2 | None = None

        ## Parameters
        self.resolution: float = 0.33
        
        ## States
        self._bc_initialized: bool = False
        self._simstart: float = 0.0
        self._simend: float = 0.0
        self._container: dict[str, Any] = dict()
        self._completed: bool = False
        
    @property
    def _params(self) -> dict[str, float]:
        return self._state.params
    
    @property
    def mesh(self) -> Mesh3D:
        return self._state.mesh
    
    @property
    def data(self) -> MWData:
        return self._state.data.mw
    
    def reset(self, _reset_bc: bool = True):
        if _reset_bc:
            self.bc = MWBoundaryConditionSet(None)
        else:
            for bc in self.bc.oftype(ModalPort):
                bc.reset()
            
        self.basis: FEMBasis = None
        self.solveroutine.reset()
        self.assembler.cached_matrices = None

    @property
    def nports(self) -> int:
        """The number of ports in the physics.

        Returns:
            int: The number of ports
        """
        return self.bc.count(PortBC)
    
    def ports(self) -> list[PortBC]:
        """A list of all port boundary conditions.

        Returns:
            list[PortBC]: A list of all port boundary conditions
        """
        return sorted(self.bc.oftype(PortBC), key=lambda x: x.number) # type: ignore
    
    
    def _initialize_bcs(self, surfaces: list[GeoSurface]) -> None:
        """Initializes the boundary conditions to set PEC as all exterior boundaries.
        """
        logger.debug('Initializing boundary conditions.')

        tags = self.mesher.domain_boundary_face_tags
        
        # Assigning surface impedance boundary condition
        if self._settings.mw_2dbc:
            for surf in surfaces:
                if surf.material.cond.scalar(1e9) > self._settings.mw_2dbc_peclim:
                    logger.debug(f'Assinging PEC to {surf}')
                    self.bc.PEC(surf)
                elif surf.material.cond.scalar(1e9) > self._settings.mw_2dbc_lim:
                    self.bc.SurfaceImpedance(surf, surf.material)
                
        
        tags = [tag for tag in tags if tag not in self.bc.assigned(2)]
        
        self.bc.PEC(FaceSelection(tags))
        
        logger.info(f'Adding PEC boundary condition with tags {tags}.')
        if self.mesher.periodic_cell is not None:
            self.mesher.periodic_cell.generate_bcs()
            for bc in self.mesher.periodic_cell.bcs:
                self.bc.assign(bc)

    def get_discretizer(self) -> Callable:
        """Returns a discretizer function that defines the maximum mesh size.

        Returns:
            Callable: The discretizer function
        """
        def disc(material: Material):
            return 299792458/(max(self.frequencies) * np.real(material.neff(max(self.frequencies))))
        return disc

    def _initialize_bc_data(self):
        ''' Initializes auxilliary required boundary condition information before running simulations.
        '''
        logger.debug('Initializing boundary conditions')
        self.bc.cleanup()
        for port in self.bc.oftype(LumpedPort):
            self.define_lumped_port_integration_points(port)
    
    def _check_physics(self) -> None:
        """ Executes a physics check before a simulation can be run."""
        if not self.bc._is_excited():
            raise SimulationError('The simulation has no boundary conditions that insert energy. Make sure to include at least one Port into your simulation.')
    
    def define_lumped_port_integration_points(self, port: LumpedPort) -> None:
        """Sets the integration points on Lumped Port objects for voltage integration

        Args:
            port (LumpedPort): The LumpedPort object

        Raises:
            SimulationError: An error if there are no nodes associated with the port.
        """
        if len(port.vintline) > 0:
            return
        logger.debug(' - Finding Lumped Port integration points')
        field_axis = port.Vdirection.np

        points = self.mesh.get_nodes(port.tags)

        if points.size==0:
            raise SimulationError(f'The lumped port {port} has no nodes associated with it')
        
        xs = self.mesh.nodes[0,points]
        ys = self.mesh.nodes[1,points]
        zs = self.mesh.nodes[2,points]

        dotprod = xs*field_axis[0] + ys*field_axis[1] + zs*field_axis[2]

        start_id = np.argwhere(dotprod == np.min(dotprod)).flatten()
        
        xs = xs[start_id]
        ys = ys[start_id]
        zs = zs[start_id]
        

        for x,y,z in zip(xs, ys, zs):
            start = np.array([x,y,z])
            end = start + port.Vdirection.np*port.height
            port.vintline.append(Line.from_points(start, end, 21))
            logger.trace(f' - Port[{port.port_number}] integration line {start} -> {end}.')
        
        port.v_integration = True

    def _find_tem_conductors(self, port: ModalPort, sigtri: np.ndarray) -> tuple[list[int], list[int]]:
        ''' Returns two lists of global node indices corresponding to the TEM port conductors.
        
        This method is invoked during modal analysis with TEM modes. It looks at all edges
        exterior to the boundary face triangulation and finds two small subsets of nodes that
        lie on different exterior boundaries of the boundary face.

        Args:
            port (ModalPort): The modal port object.
            
        Returns:
            list[int]: A list of node integers of island 1.
            list[int]: A list of node integers of island 2.
        '''
        if self.basis is None:
            raise ValueError('The field basis is not yet defined.')

        logger.debug(' - Finding PEC TEM conductors')
        mesh = self.mesh
        
        # Find all BC conductors
        pecs: list[PEC] = self.bc.get_conductors() # type: ignore

        # Process all PEC Boundary Conditions
        pec_edges = []
        for pec in pecs:
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
            pec_edges.extend(edge_ids)
        
        # Process conductivity
        for itri in mesh.get_triangles(port.tags):
            if sigtri[itri] > 1e6:
                edge_ids = list(mesh.tri_to_edge[:,itri].flatten())
                pec_edges.extend(edge_ids)

        # All PEC edges
        pec_edges = list(set(pec_edges))
        
        # Port mesh data
        tri_ids = mesh.get_triangles(port.tags)
        edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
        
        port_pec_edges = np.array([i for i in pec_edges if i in set(edge_ids)])
        
        pec_islands = mesh.find_edge_groups(port_pec_edges)

        
        logger.debug(f' - Found {len(pec_islands)} PEC islands.')

        if len(pec_islands) != 2:
            pec_island_tags = {i: self.mesh._get_dimtags(edges=pec_edge_group) for i,pec_edge_group in enumerate(pec_islands)}
            plus_term = None
            min_term = None
            
            for i, dimtags in pec_island_tags.items():
                if not set(dimtags).isdisjoint(port.plus_terminal):
                    plus_term = i
                
                if not set(dimtags).isdisjoint(port.minus_terminal):
                    min_term = i
            
            if plus_term is None or min_term is None:
                logger.error(f' - Found {len(pec_islands)} PEC islands without a terminal definition. Please use .set_terminals() to define which conductors are which polarity, or define the integration line manually.') 
                return None, None  
            logger.debug(f'Positive island = {pec_island_tags[plus_term]}')
            logger.debug(f'Negative island = {pec_island_tags[min_term]}')
            pec_islands = [pec_islands[plus_term], pec_islands[min_term]]
        
        groups = []
        for island in pec_islands:
            group = set()
            for edge in island:
                group.add(mesh.edges[0,edge])
                group.add(mesh.edges[1,edge])
            groups.append(sorted(list(group)))
        
        group1 = groups[0]
        group2 = groups[1]

        return group1, group2
    
    
    def _get_material_assignment(self, volumes: list[GeoVolume]) -> list[Material]:
        '''Retrieve the material properties of the geometry'''
        
        # Reset index assingments
        for vol in volumes:
            vol.material.reset()
        
        # collect all materials
        materials = []
        assignment_dict: dict[int, list[GeoVolume]] = defaultdict(list)
        i = 0
        for vol in volumes:
            for tag in vol.tags:
                assignment_dict[tag].append(vol)
            if vol.material not in materials:
                materials.append(vol.material)
                vol.material._hash_key = i
                i += 1
        
        # Check competing priorities!
        for domaintag, volumelist in assignment_dict.items():
            priolist = [vol._priority for vol in volumelist]
            maxprio = max(priolist)
            if priolist.count(maxprio) > 1:
                vols = [vol for vol in volumelist if vol._priority==maxprio]
                logger.warning(f'Domain with tag {domaintag} has multiple geometries imposing a material to them: {vols}. Consider setting priorities to decide which volume is more important.')
                DEBUG_COLLECTOR.add_report(f'Domain with tag {domaintag} has multiple geometries imposing a material to them: {vols}. Consider setting priorities to decide which volume is more important.')
            
        xs = self.mesh.centers[0,:]
        ys = self.mesh.centers[1,:]
        zs = self.mesh.centers[2,:]
        
        matassign = -1*np.ones((self.mesh.n_tets,), dtype=np.int64)
        
        
        for volume in sorted(volumes, key=lambda x: x._priority):
        
            for dimtag in volume.dimtags:
                
                tet_ids = self.mesh.get_tetrahedra(dimtag[1])
                
                matassign[tet_ids] = volume.material._hash_key
        
        if np.any(matassign==-1):
            raise SimulationError(f'Tetrahedra detected with unassigned materials: {np.argwhere(matassign==-1)}')
        for mat in materials:
            ids = np.argwhere(matassign==mat._hash_key).flatten()
            mat.initialize(xs[ids], ys[ids], zs[ids], ids)
                    
        
        return materials
    