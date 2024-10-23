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
  
def minimize(fun, x0, args=(), *, method='BFGS', tol=None, options=None): 
    if not rp.use_jax: return np.minimize(fun, x0, args=args, method=method,tol=tol, options=options)
    else: return jnp.minimize(fun, x0, args=args,method=method, tol=tol, options=options)
    
def OptimizeResults(x, success, status, fun, jac, hess_inv, nfev, njev, nit): 
    if not rp.use_jax: return np.OptimizeResults(x, success, status, fun, jac, hess_inv, nfev, njev, nit)
    else: return jnp.OptimizeResults(x, success, status, fun, jac, hess_inv, nfev, njev, nit)
 