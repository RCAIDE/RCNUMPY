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
    
def dct(x,type=2, n=None, axis=-1, norm=None): 
    if not rp.use_jax: return np.dct(x,type=type, n=n, axis=axis, norm=norm)
    else: return jnp.dct(x,type=type,n=n,axis=axis, norm=norm)
    
def dctn(x, type=2, s=None, axes=None, norm=None): 
    if not rp.use_jax: return np.dctn(x,type=type, s=s, axes=axes, norm=norm)
    else: return jnp.dctn(x, type=type, s=s, axes=axes, norm=norm)
     
def idct(x, type=2, n=None, axis=-1, norm=None): 
    if not rp.use_jax: return np.idct(x, type=type, n=n, axis=axis, norm=norm)
    else: return jnp.ictn(x, type=type, n=n, axis=axis, norm=norm)
     
def idctn(x, type=2, s=None, axes=None, norm=None): 
    if not rp.use_jax: return np.idctn(x,type=type, s=s, axes=axes, norm=norm)
    else: return jnp.idctn(x, type=type, s=s, axes=axes, norm=norm)
   