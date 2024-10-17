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

def trapezoid(y, x=None, dx=1.0, axis=-1): 
    if not rp.use_jax: return np.trapezoid(y=y, x=x, dx=dx, axis=axis)
    else: return jnp.trapezoid(y=y, x=x, dx=dx, axis=axis)
    
     