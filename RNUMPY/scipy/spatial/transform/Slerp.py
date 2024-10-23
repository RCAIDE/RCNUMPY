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
    
def Slerp(times, timedelta, rotations, rotvecs): 
    if not rp.use_jax: return np.Slerp(times=times, rotations=rotations)
    else: return jnp.Slerp(times=times, timedelta=timedelta, rotations=rotations, rotvecs=rotvecs)
   