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

#def dot(x,y):
    #if not rp.use_jax: return np.dot(x,y)
    #else: return jnp.dot(x,y)
    
def dct(x,type=2, n=None, axis=-1, norm=None): 
    if not rp.use_jax: return np.dct(x,type=type, n=n, axis=axis, norm=norm)
    else: return jnp.dct(x,type=type,n=n,axis=axis, norm=norm)
        
def dctn():  raise NotImplementedError
def idct():  raise NotImplementedError
def idctn(): raise NotImplementedError 