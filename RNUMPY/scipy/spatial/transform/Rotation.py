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

class Rotation(): 
    
    def __init__():  
        return
    
    def apply(R, vectors, inverse=False): 
        if not rp.use_jax: return R.apply(vectors,inverse=vectors)
        else: return R.apply()
        
    def as_euler(R, seq, degrees=False): 
        if not rp.use_jax: return R.as_euler(seq=seq, degrees=degrees )
        else: return R.as_euler(seq=seq, degrees=degrees)
        
    def as_matrix(R): 
        if not rp.use_jax: return R.as_matrix()
        else: return R.as_matrix()
        
    def as_mrp(R): 
        if not rp.use_jax: return R.as_mrp()
        else: return R.as_mrp()
        
    def as_quat(R, canonical=False, scalar_first=False): 
        if not rp.use_jax: return R.as_quat(canonical=canonical, scalar_first=scalar_first)
        else: return R.as_quat(canonical=canonical, scalar_first=scalar_first)
        
    def as_rotvec(R, degrees=False): 
        if not rp.use_jax: return R.as_rotvec(degrees=degrees)
        else: return R.as_rotvec(degrees=degrees)
        
    def concatenate(R,rotations): 
        if not rp.use_jax: return R.concatenate(rotations=rotations)
        else: return R.concatenate(rotations=rotations)
         
    def from_euler(R,seq, angles, degrees=False): 
        if not rp.use_jax: return R.from_euler(seq=seq, angles= angles, degrees=degrees)
        else: return R.from_euler(seq=seq, angles= angles, degrees=degrees)
        
    def from_matrix(R,matrix): 
        if not rp.use_jax: return R.from_matrix(matrix=matrix)
        else: return R.from_matrix(matrix=matrix)
        
    def from_mrp(R,mrp): 
        if not rp.use_jax: return R.from_mrp(mrp=mrp)
        else: return R.from_mrp(mrp=mrp)
        
    def from_quat(R,quat): 
        if not rp.use_jax: return R.from_quat(quat=quat)
        else: return R.from_quat(quat=quat)
        
    def from_rotvec(R,rotvec, degrees=False): 
        if not rp.use_jax: return R.from_rotvec(rotvec=rotvec, degrees=degrees)
        else: return R.from_rotvec(rotvec=rotvec, degrees=degrees)
        
    def identity(R, num=None): 
        if not rp.use_jax: return R.identity(num=num)
        else: return R.identity(num=num) 
        
    def inv(R): 
        if not rp.use_jax: return R.inv()
        else: return R.inv()
        
    def magnitude(R): 
        if not rp.use_jax: return R.magnitude()
        else: return R.magnitude()
        
    def mean(R, weights=None): 
        if not rp.use_jax: return R.mean(weights=weights)
        else: return R.mean(weights=weights)
        
    def random(R, num=None, random_state=None): 
        if not rp.use_jax: return R.random()
        else: return R.random( num=num, random_state=random_state)     
        
    
    def count():    raise NotImplementedError 
    def index():    raise NotImplementedError     