from mumps import ZMumpsContext
from scipy import sparse
import numpy as np

# Reuse factorizations by running `job=3` with new right hand sides
# or analyses by supplying new values and running `job=2` to repeat
# the factorization process.

class MUMPSInterface:
    def __init__(self):
        self.ctx = ZMumpsContext(1,0)
        self.ctx.set_silent()
        self._analysed = False
        self._factorized = False
        # Parallel analysis
        #self.ctx.id.icntl[27] = 1   # sequential analysis
        #self.ctx.id.icntl[47] = 1
        self.ctx.id.icntl[6]  = 5   # METIS
        self.ctx.id.icntl[27] = 1      # ICNTL(28)
        self.ctx.id.icntl[28] = 0      # ICNTL(29) automatic (safe)
        
    def analyse_matrix(self, A: sparse.csr_matrix):
        # Convert to COO
        A = A.tocoo()
        n = A.shape[0]

        if self.ctx.myid == 0:
            # 1-based indices
            irn = A.row.astype(np.int32) + 1
            jcn = A.col.astype(np.int32) + 1
            a = A.data.astype(np.complex128)

            self.ctx.set_shape(n)
            self.ctx.set_centralized_assembled(irn, jcn, a)

        # Analysis
        self.ctx.run(job=1)
        self._analysed = True
        self._factorized = False

    def factorize(self, A: sparse.csr_matrix):
        if not self._analysed:
            raise RuntimeError("Matrix must be analysed before factorization")
        A = A.tocoo()

        n = A.shape[0]

        if self.ctx.myid == 0:
            # 1-based indices
            irn = A.row.astype(np.int32) + 1
            jcn = A.col.astype(np.int32) + 1
            a = A.data.astype(np.complex128)

            self.ctx.set_shape(n)
            self.ctx.set_centralized_assembled(irn, jcn, a)
            
        self.ctx.run(job=2)
        self._factorized = True

    def solve(self, b):
        if not self._factorized:
            raise RuntimeError("Matrix must be factorized before solve")
        if self.ctx.myid == 0:
            x = np.asarray(b, dtype=np.complex128).copy()
            self.ctx.set_rhs(x)
        else:
            x = None

        self.ctx.run(job=3)

        return x, 0

    def destroy(self):
        self.ctx.destroy()
        