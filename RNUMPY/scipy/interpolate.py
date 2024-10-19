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


def RegularGridInterpolator(points, values, method='linear', bounds_error=False, fill_value= np.nan ): 
    if not rp.use_jax: return np.RegularGridInterpolator(points = points, values = values, method=method, bounds_error=bounds_error, fill_value=fill_value)
    else: return jnp.RegularGridInterpolator(points = points, values = values, method=method, bounds_error=bounds_error, fill_value=fill_value )
 