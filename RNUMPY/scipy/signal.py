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

def fftconvolve(in1, in2, mode='full', axes=None): 
    if not rp.use_jax: return np.fftconvolve(in1, in2, mode=mode, axes=axes)
    else: return jnp.fftconvolve(in1, in2, mode=mode, axes=axes)
    
def convolve(in1, in2, mode='full', method='auto'): 
    if not rp.use_jax: return np.convolve(in1, in2, mode=mode, method=method)
    else: return jnp.convolve(in1, in2, mode=mode, method=method) 

def convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0, precision=None): 
    if not rp.use_jax: return np.convolve2d(in1, in2, mode=mode, boundary=boundary, fillvalue=fillvalue)
    else: return jnp.convolve2d(in1, in2, mode=mode, boundary=boundary, fillvalue=fillvalue, precision=precision)        
 
def correlate():   raise NotImplementedError
def correlate2d(): raise NotImplementedError
def csd():         raise NotImplementedError
def detrend():     raise NotImplementedError
def istft():       raise NotImplementedError
def stft():        raise NotImplementedError
def welch():       raise NotImplementedError 

