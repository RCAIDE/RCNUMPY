# src.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Oct 2024 M. Clarke
# Modified: 

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORTS
# ----------------------------------------------------------------------------------------------------------------------  

import RNUMPY as rp

j   = rp.jax_handle
np  = rp.numpy_handle
sp  = rp.scipy_handle
jnp = j.numpy
 
    
#def trapezoid( ): 
    #if not rp.use_jax: return np.trapezoid( )
    #else: return jnp.trapezoid( )
        
                         
def block_diag():       raise NotImplementedError 
def cho_factor():       raise NotImplementedError 
def cho_solve():        raise NotImplementedError 
def cholesky():         raise NotImplementedError 
def det():              raise NotImplementedError 
def eigh():             raise NotImplementedError 
def eigh_tridiagonal(): raise NotImplementedError 
def expm():             raise NotImplementedError 
def expm_frechet():     raise NotImplementedError 
def funm():             raise NotImplementedError 
def hessenberg():       raise NotImplementedError 
def hilbert():          raise NotImplementedError 
def inv():              raise NotImplementedError 
def lu_factor():        raise NotImplementedError 
def lu():               raise NotImplementedError 
def lu_solve():         raise NotImplementedError
def polar():            raise NotImplementedError
def qr():               raise NotImplementedError
def rsf2csf():          raise NotImplementedError
def schur():            raise NotImplementedError
def solve():            raise NotImplementedError
def solve_triangular(): raise NotImplementedError
def sqrt():             raise NotImplementedError
def svd():              raise NotImplementedError
def toepliz():          raise NotImplementedError