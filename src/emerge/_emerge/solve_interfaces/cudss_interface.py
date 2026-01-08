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

import warnings
from loguru import logger

# Catch the Cuda warning and print it with Loguru.
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import cupy as cp
    for warn in w:
        logger.debug(f"{warn.category.__name__}: {warn.message}")

import nvmath.bindings.cudss as cudss # ty: ignore
from nvmath import CudaDataType # ty: ignore

from scipy.sparse import csr_matrix
import numpy as np
from typing import Literal

############################################################
#                         CONSTANTS                        #
############################################################

ALG_NEST_DISS_METIS = cudss.AlgType.ALG_DEFAULT
ALG_COLAMD = cudss.AlgType.ALG_1
ALG_COLAMD_BLOCK_TRI = cudss.AlgType.ALG_2
ALG_AMD = cudss.AlgType.ALG_3

FLOAT64 = CudaDataType.CUDA_R_64F
FLOAT32 = CudaDataType.CUDA_R_32F
COMPLEX128 = CudaDataType.CUDA_C_64F
COMPLEX64 = CudaDataType.CUDA_C_32F
INT64 = CudaDataType.CUDA_R_64I
INT32 = CudaDataType.CUDA_R_32I

INDEX_BASE = cudss.IndexBase.ZERO

def _c_pointer(arry) -> int:
    return int(arry.data.ptr)



############################################################
#                         FUNCTIONS                        #
############################################################

def is_complex_symmetric(A, rtol=1e-12, atol=0.0):
    D = A-A.T
    D.sum_duplicates()
    if D.nnz == 0:
        return True
    max_diff = float(np.abs(D.data).max())
    max_A = float(np.abs(A.data).max()) if A.nnz else 0.0
    return (max_diff <= atol) if max_A == 0.0 else (max_diff <= atol + rtol * max_A)

############################################################
#                         INTERFACE                        #
############################################################

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class CuDSSInterface(metaclass=Singleton):
    """the CuDSSInterface class implements the nvmath bindings and cupy
    control for EMerge.
    """
    AlgType = cudss.AlgType

    def __init__(self):
        self.A_cu = None
        self.b_cu = None
        self.x_cu = None
        self.A_cobj = None
        self.b_cobj = None
        self.x_cobj = None
        self.A_pattern = None

        self._handle = cudss.create()
        self._config = cudss.config_create()
        self._data = cudss.data_create(self._handle)

        self.MTYPE = cudss.MatrixType.GENERAL
        self.MVIEW = cudss.MatrixViewType.FULL
        self.RALG = cudss.AlgType.ALG_DEFAULT
        self.VTYPE = CudaDataType.CUDA_R_64F

        self._INDPTR = None
        self._ROW_START: int | None = None
        self._ROW_END: int | None = None
        self._IND = None
        self._VAL = None
        self._NNZ: int | None = None
        self._COMP: bool = True
        self._PRES: int = 2
        self._COL_IDS = None

        self._initialized = False

        param = cudss.ConfigParam.REORDERING_ALG
        dtype = cudss.get_config_param_dtype(int(param))
        reorder_alg = np.array(self.RALG, dtype=dtype)

        cudss.config_set(
            self._config,   
            int(param),
            reorder_alg.ctypes.data, 
            reorder_alg.nbytes
        )

    def set_algorithm(self, alg_type: Literal['METIS','COLAMD','COLAM_BT','AMD']):
        """Define fill-in reduction column permuation algorithm. The options are:

         - "METIS" (Default) = NVidia's own Nested Dissection METIS sorter
         - "COLAMD" = Column approximate minimum degree
         - "COLAM_BT" = Column Approximate Minimum Degree Block Triangular
         - "AMD" = Approximate Minimum Degree

        Args:
            alg_type (str): The chosen type
        """
        if alg_type=='METIS':
            self.RALG = ALG_NEST_DISS_METIS
        elif alg_type =='COLAMD':
            self.RALG = ALG_COLAMD
        elif alg_type == 'COLAMD_BT':
            self.RALG = ALG_COLAMD_BLOCK_TRI
        elif alg_type == 'AMD':
            self.RALG = ALG_AMD
        else:
            logger.warning(f'Algorithm type {alg_type} is not of the chosen set. Ignoring setting.')
    
    def init_type(self):
        """Initializes the value data type of the solver (float vs complex, single vs double).
        """
        if self._PRES == 1:
            if self._COMP:
                self.c_dtype = cp.complex64
                self.VTYPE = COMPLEX64
            else:
                self.c_dtype = cp.float32
                self.VTYPE = FLOAT32
        else:
            if self._COMP:
                self.c_dtype = cp.complex128
                self.VTYPE = COMPLEX128
            else:
                self.c_dtype = cp.float64
                self.VTYPE = FLOAT64

    def submit_matrix(self, A: csr_matrix):
        """Sets the given csr_matrix as the matrix to be solved.

        Args:
            A (csr_matrix): The csr_format matrix for the problem Ax=b
        """
        self.N = A.shape[0]

        if is_complex_symmetric(A, rtol=1e-12, atol=1e-15):
            self.MTYPE = cudss.MatrixType.SYMMETRIC
        else:
            self.MTYPE = cudss.MatrixType.GENERAL
        
        if np.iscomplexobj(A):
            self._COMP = True
        else:
            self._COMP = False
        
        self.init_type()

        self.A_cu = cp.sparse.csr_matrix(A).astype(self.c_dtype)

        self._INDPTR = cp.ascontiguousarray(self.A_cu.indptr.astype(cp.int32))
        self._IND = cp.ascontiguousarray(self.A_cu.indices.astype(cp.int32))
        self._VAL = cp.ascontiguousarray(self.A_cu.data)
        self._NNZ = int(self._VAL.size)
        self._ROW_START = self._INDPTR[:-1]
        self._ROW_END = self._INDPTR[1:]
        self._COL_IDS = self.A_cu.indices.astype(cp.int32)

    def submit_vector(self, b: np.ndarray):
        """Submits the dense vector b to be solved.

        Args:
            b (np.ndarray): The dense vector for the problem Ax=b
        """
        self.b_cu = cp.array(b).astype(self.c_dtype)
    
    def create_solvec(self):
        """Initializes a solution vector that the nvmath binding can access.
        """
        self.x_cu = cp.empty_like(self.b_cu)

    def _update_dss_data(self):
        """Updates the currently defined matrix data into the existing memory.ALG_AMD
        """
        cudss.matrix_set_values(self.A_cobj, _c_pointer(self._VAL))
        
        self.b_cobj = cudss.matrix_create_dn(self.N, 1, self.N, _c_pointer(self.b_cu),
                                    int(self.VTYPE), int(cudss.Layout.COL_MAJOR))
        self.x_cobj = cudss.matrix_create_dn(self.N, 1, self.N, _c_pointer(self.x_cu),
                                    int(self.VTYPE), int(cudss.Layout.COL_MAJOR))

    def _create_dss_data(self):
        """Creates a new memory slot for the CSR matrix of the matrix A"""
        self.A_cobj = cudss.matrix_create_csr(
            self.N,self.N,self._NNZ,
            _c_pointer(self._ROW_START),
            _c_pointer(self._ROW_END),
            _c_pointer(self._COL_IDS),
            _c_pointer(self._VAL),
            int(INT32), 
            int(self.VTYPE),
            int(self.MTYPE),
            int(self.MVIEW),
            int(INDEX_BASE),
        )

        self.b_cobj = cudss.matrix_create_dn(self.N, 1, self.N, _c_pointer(self.b_cu),
                                    int(self.VTYPE), int(cudss.Layout.COL_MAJOR))
        self.x_cobj = cudss.matrix_create_dn(self.N, 1, self.N, _c_pointer(self.x_cu),
                                    int(self.VTYPE), int(cudss.Layout.COL_MAJOR))

    def from_symbolic(self, A: csr_matrix, b: np.ndarray) -> np.ndarray:
        """Solves Ax=b starting from the symbolic factorization

        Args:
            A (csr_matrix): The input sparse matrix
            b (np.ndarray): The solution vector b

        Returns:
            np.ndarray: The solved vector
        """
        self.submit_matrix(A)
        self.submit_vector(b)
        self.create_solvec()
        self._create_dss_data()
        self._symbolic()
        self._numeric(False)
        return self._solve()

    def from_numeric(self, A: csr_matrix, b: np.ndarray) -> np.ndarray:
        """Solves Ax=b starting from the Numeric factorization

        Args:
            A (csr_matrix): The input sparse matrix
            b (np.ndarray): The solution vector b

        Returns:
            np.ndarray: The solved vector
        """
        self.submit_matrix(A)
        self.submit_vector(b)
        self.create_solvec()
        self._update_dss_data()
        self._numeric(True)
        return self._solve()

    def from_solve(self, b: np.ndarray) -> np.ndarray:
        """Solves Ax=b only with a new b vector.

        Args:
            A (csr_matrix): The input sparse matrix
            b (np.ndarray): The solution vector b

        Returns:
            np.ndarray: The solved vector
        """
        self.submit_vector(b)
        self.create_solvec()
        return self._solve()

    def _symbolic(self):
        logger.trace('Executing symbolic factorization')
        cudss.execute(self._handle, cudss.Phase.ANALYSIS, self._config, self._data, 
                      self.A_cobj, self.x_cobj, self.b_cobj)

    def _numeric(self, refactorize: bool = False):
        if refactorize:
            logger.trace('Refactoring matrix')
            phase = cudss.Phase.REFACTORIZATION
        else:
            phase = cudss.Phase.FACTORIZATION
        logger.trace('Executing numerical factorization')
        cudss.execute(self._handle, phase, self._config, self._data, 
                      self.A_cobj, self.x_cobj, self.b_cobj)
                      
    def _solve(self) -> np.ndarray:
        logger.trace('Solving matrix problem')
        cudss.execute(self._handle, cudss.Phase.SOLVE, self._config, self._data, 
                      self.A_cobj, self.x_cobj, self.b_cobj)
        cp.cuda.runtime.deviceSynchronize()
        x_host = cp.asnumpy(self.x_cu).ravel()
        return x_host
