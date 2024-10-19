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
 
def block_diag(*arrs): 
    if not rp.use_jax: return np.block_diag(arrs)
    else: return jnp.block_diag(arrs)   
                  
def cho_factor(a, lower=False, overwrite_a=False, check_finite=True): 
    if not rp.use_jax: return np.cho_factor(a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)
    else: return jnp.cho_factor(a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)
 
def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True): 
    if not rp.use_jax: return np.cho_solve(c_and_lower, b, overwrite_b=overwrite_b, check_finite=check_finite)
    else: return jnp.cho_solve(c_and_lower, b, overwrite_b=overwrite_b, check_finite=check_finite) 


def eigh_tridiagonal(d, e, *, eigvals_only=False, select='a', select_range=None, tol=None): 
    if not rp.use_jax: return np.eigh_tridiagonal(d, e, eigvals_only=eigvals_only, select=select, select_range=select_range, tol=tol)
    else: return jnp.eigh_tridiagonal(d, e, eigvals_only=eigvals_only, select=select, select_range=select_range, tol=tol)
           

def det(a, overwrite_a=False, check_finite=True): 
    if not rp.use_jax: return np.det(a, overwrite_a=overwrite_a, check_finite=check_finite)
    else: return jnp.det(a, overwrite_a=overwrite_a, check_finite=check_finite)


def eigh(a, b = None, lower = True, eigvals_only = False, overwrite_a= False, overwrite_b = False, turbo = True,
         eigvals = None, type = 1, check_finite = True): 
    if not rp.use_jax: return np.eigh(a, b, lower = lower, eigvals_only = eigvals_only, overwrite_a= overwrite_a,
                          overwrite_b =overwrite_b, type = type, check_finite= check_finite)
    else: return jnp.eigh(a, b, lower = lower, eigvals_only = eigvals_only, overwrite_a= overwrite_a,
                          overwrite_b =overwrite_b, turbo = turbo, eigvals= eigvals, type = type,
                          check_finite= check_finite)

def expm(A, *, upper_triangular=False, max_squarings=16): 
    if not rp.use_jax: return np.expm(A)
    else: return jnp.expm(A, upper_triangular=upper_triangular, max_squarings=max_squarings)
 
def expm_frechet(A, E, *, method = None, compute_expm= True): 
    if not rp.use_jax: return np.expm_frechet(A, E, method=method, compute_expm=compute_expm )
    else: return jnp.expm_frechet(A, E,method = method, compute_expm= compute_expm)
           
def hessenberg(a, calc_q= False, overwrite_a = False, check_finite= True): 
    if not rp.use_jax: return np.hessenberg(a, calc_q= calc_q, overwrite_a = overwrite_a, check_finite= check_finite )
    else: return jnp.hessenberg(a, calc_q= calc_q, overwrite_a = overwrite_a, check_finite= check_finite )

def hilbert(n): 
    if not rp.use_jax: return np.hilbert(n)
    else: return jnp.hilbert(n)

def inv(a, overwrite_a=False, check_finite=True ): 
    if not rp.use_jax: return np.inv(a, overwrite_a=overwrite_a, check_finite=check_finite)
    else: return jnp.inv(a, overwrite_a=overwrite_a, check_finite=check_finite )

def lu_factor(a, overwrite_a=False, check_finite=True): 
    if not rp.use_jax: return np.lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite )
    else: return jnp.lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite )
    
def lu(a, permute_l=False, overwrite_a=False, check_finite=True, p_indices=False ): 
    if not rp.use_jax: return np.lu(a, permute_l=permute_l, overwrite_a=overwrite_a, check_finite=check_finite, p_indices=p_indices)
    else: return jnp.lu(a, permute_l=permute_l, overwrite_a=overwrite_a, check_finite=check_finite, p_indices=p_indices )

def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True): 
    if not rp.use_jax: return np.lu_solve(lu_and_piv, b, trans=trans, overwrite_b=overwrite_b, check_finite=check_finite )
    else: return jnp.lu_solve(lu_and_piv, b, trans=trans, overwrite_b=overwrite_b, check_finite=check_finite )

def polar(a, side='right', *, method='qdwh', eps=None, max_iterations=None): 
    if not rp.use_jax: return np.polar(a, side=side)
    else: return jnp.polar(a, side=side, method=method, eps=eps, max_iterations=max_iterations)
                
def qr(a, overwrite_a = False, lwork  = None, mode = 'full', pivoting = False, check_finite = True): 
    if not rp.use_jax: return np.qr(a, overwrite_a = overwrite_a, lwork  = lwork, mode = mode, pivoting = pivoting, check_finite = check_finite)   
    else: return jnp.qr(a, overwrite_a = overwrite_a, lwork  = lwork, mode = mode, pivoting = pivoting, check_finite = check_finite)      
 
def rsf2csf(T, Z, check_finite=True): 
    if not rp.use_jax: return np.rsf2csf(T, Z, check_finite=check_finite)
    else: return jnp.rsf2csf(T, Z, check_finite=check_finite )

def schur(a, output='real'): 
    if not rp.use_jax: return np.schur(a, output=output)
    else: return jnp.schur(a, output=output)

def solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, debug=False, check_finite=True, assume_a='gen'): 
    if not rp.use_jax: return np.solve(a, b, lower=lower, overwrite_a=overwrite_a, overwrite_b=overwrite_b,  check_finite=check_finite, assume_a=assume_a)
    else: return jnp.solve(a, b, lower=lower, overwrite_a=overwrite_a, overwrite_b=overwrite_b, debug=debug, check_finite=check_finite, assume_a=assume_a)

def solve_triangular( ): 
    if not rp.use_jax: return np.solve_triangular( )
    else: return jnp.solve_triangular( )

def sqrtm(A, blocksize=1): 
    if not rp.use_jax: return np.sqrt(A, blocksize=blocksize)
    else: return jnp.sqrt(A, blocksize=blocksize)
 
def svd(a, full_matrices= True, compute_uv = True, overwrite_a = False, check_finite= True, lapack_driver= 'gesdd'): 
    if not rp.use_jax: return np.svd(a, full_matrices= full_matrices, compute_uv = compute_uv, overwrite_a =overwrite_a, check_finite= check_finite, lapack_driver= lapack_driver)
    else: return jnp.svd(a, full_matrices= full_matrices, compute_uv = compute_uv, overwrite_a =overwrite_a, check_finite= check_finite, lapack_driver= lapack_driver)

def toeplitz(c, r=None): 
    if not rp.use_jax: return np.toeplitz(c, r=r)
    else: return jnp.toeplitz(c, r=r)

def funm(A, func, disp=True): 
    if not rp.use_jax: return np.funm(A, func, disp=disp)
    else: return jnp.funm(A, func, disp=disp)        
      