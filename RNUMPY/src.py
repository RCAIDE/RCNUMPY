# src.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Aug 2024 E. Botero
# Modified: 

# ----------------------------------------------------------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------------------------------------------------------  

import RNUMPY as rp

j   = rp.jax_handle
np  = rp.numpy_handle
jnp = j.numpy

# ----------------------------------------------------------------------------------------------------------------------
#  Debug Print Function
# ----------------------------------------------------------------------------------------------------------------------  

def debugprint(fmt, *args, ordered=False, **kwargs):
    if not rp.use_jax: print(fmt.format(*args, **kwargs))
    else: j.debug.print(fmt,*args,ordered=ordered,**kwargs)


# ----------------------------------------------------------------------------------------------------------------------
#  .at functions
# ----------------------------------------------------------------------------------------------------------------------  

def set(a,indices,b):
    if not rp.use_jax: a[indices] = b; return a 
    else: return a.at[indices].set(b)
        
def get(a,indices):
    if not rp.use_jax: return a[indices]
    else: return a.at[indices].get()

def pequals(a,indices,b):
    if not rp.use_jax: a[indices] += b; return a
    else: return a.at[indices].add(b)
         
def sequals(a,indices,b):
    if not rp.use_jax: a[indices] -= b; return a
    else: return a.at[indices].minus(b)

def mequals(a,indices,b):
    if rp.use_jax: a[indices] *= b; return a
    else: return a.at[indices].multiply(b)

def dequals(a,indices,b):
    if not rp.use_jax: a[indices] /= b; return a
    else: return a.at[indices].divide(b)

# ----------------------------------------------------------------------------------------------------------------------
#  ndarray
# ----------------------------------------------------------------------------------------------------------------------  

class ndarray(np.ndarray, j.Array):
    pass

# ----------------------------------------------------------------------------------------------------------------------
#  Numpy Functions
# ----------------------------------------------------------------------------------------------------------------------  

def dot(x,y):
    if not rp.use_jax: return np.dot(x,y)
    else: return jnp.dot(x,y)

def sin(x,/):
    if not rp.use_jax: return np.sin(x)
    else: return jnp.sin(x)

def cos(x,/): 
    if not rp.use_jax: return np.cos(x)
    else: return jnp.cos(x)

def tan(x,/):
    if not rp.use_jax: return np.tan(x)
    else: return jnp.tan(x)
    
def arcsin(x,/):
    if not rp.use_jax: return np.arcsin(x)
    else: return jnp.arcsin(x)

def asin(x,/):
    if not rp.use_jax: return np.asin(x)
    else: return jnp.asin(x)

def arccos(x,/):
    if not rp.use_jax: return np.arccos(x)
    else: return jnp.arccos(x)

def acos(x,/):
    if not rp.use_jax: return np.acos(x)
    else: return jnp.acos(x)

def arctan(x,/): 
    if not rp.use_jax: return np.arctan(x)
    else: return jnp.arctan(x)

def atan(x,/):
    if not rp.use_jax: return np.atan(x)
    else: return jnp.atan(x)

def hypot(x1,x2,/):
    if not rp.use_jax: return np.hypot(x1,x2)
    else: return jnp.hypot(x1,x2)

def arctan2(x1,x2,/):
    if not rp.use_jax: return np.arctan2(x1,x2)
    else: return jnp.arctan2(x1,x2)

def atan2(x1,x2,/):
    if not rp.use_jax: return np.atan2(x1,x2)
    else: return jnp.atan2(x1,x2)

def degrees(x,/):
    if not rp.use_jax: return np.degrees(x)
    else: return jnp.degrees(x)

def radians(x,/):
    if not rp.use_jax: return np.radians(x)
    else: return jnp.radians(x)

def unwrap(p,discont=None,axis=-1,period=6.283185307179586): 
    if not rp.use_jax: return np.radians(p,discont=discont,axis=axis,period=period)
    else: return jnp.unwrap(p,discont=discont,axis=axis,period=period)

def deg2rad(x,/):
    if not rp.use_jax: return np.deg2rad(x)
    else: return jnp.deg2rad(x)

def rad2deg(x,/): 
    if not rp.use_jax: return np.rad2deg(x)
    else: return jnp.rad2deg(x)

def sinh(x,/):
    if not rp.use_jax: return np.sinh(x)
    else: return jnp.sinh(x)

def cosh(x,/):
    if not rp.use_jax: return np.cosh(x)
    else: return jnp.cosh(x)

def tanh(x,/):
    if not rp.use_jax: return np.tanh(x)
    else: return jnp.tanh(x)

def arcsinh(x,/):
    if not rp.use_jax: return np.arcsinh(x)
    else: return jnp.arcsinh(x)

def asinh(x,/):
    if not rp.use_jax: return np.asinh(x)
    else: return jnp.asinh(x)

def arccosh(x,/):
    if not rp.use_jax: return np.arccosh(x)
    else: return jnp.arccosh(x)

def acosh(x,/):
    if not rp.use_jax: return np.acosh(x)
    else: return jnp.acosh(x)

def arctanh(x,/):
    if not rp.use_jax: return np.arctanh(x)
    else: return jnp.arctanh(x)

def atanh(x,/):
    if not rp.use_jax: return np.atanh(x)
    else: return jnp.atanh(x)

def round(a,decimals=0,out=None):
    if not rp.use_jax: return np.round(a,decimals=decimals,out=out)
    else: return jnp.round(a,decimals=decimals,out=out)

def around(a,decimals=0,out=None):
    if not rp.use_jax: return np.around(a,decimals=decimals,out=out)
    else: return jnp.around(a,decimals=decimals,out=out)

def rint(x,/):
    if not rp.use_jax: return np.rint(x)
    else: return jnp.rint(x)

def fix(x,out=None):
    if not rp.use_jax: return np.fix(x,out=out)
    else: return jnp.fix(x,out=out)

def floor(x,/):
    if not rp.use_jax: return np.floor(x)
    else: return jnp.floor(x)

def ceil(x,/):
    if not rp.use_jax: return np.ceil(x)
    else: return jnp.ceil(x)

def trunc(x):
    if not rp.use_jax: return np.trunc(x)
    else: return jnp.trunc(x)

def prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None, promote_integers=True):
    if not rp.use_jax: return np.prod(a,axis=axis,dtype=dtype,out=out,keepdims=keepdims,initial=initial,where=where)
    else: return jnp.prod(a,axis=axis,dtype=dtype,out=out,keepdims=keepdims,initial=initial,where=where,promote_integers=promote_integers)

def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None, promote_integers=True):
    if not rp.use_jax: return np.sum(a,axis=axis,dtype=dtype,out=out,keepdims=keepdims,initial=initial,where=where)
    else: return jnp.sum(a,axis=axis,dtype=dtype,out=out,keepdims=keepdims,initial=initial,where=where,promote_integers=promote_integers)

def nanprod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None, promote_integers=True):
    if not rp.use_jax: return np.nanprod(a,axis=axis,dtype=dtype,out=out,keepdims=keepdims,initial=initial,where=where)
    else: return jnp.nanprod(a,axis=axis,dtype=dtype,out=out,keepdims=keepdims,initial=initial,where=where,promote_integers=promote_integers)

def nansum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None, promote_integers=True):
    if not rp.use_jax: return np.nansum(a,axis=axis,dtype=dtype,out=out,keepdims=keepdims,initial=initial,where=where)
    else: return jnp.nansum(a,axis=axis,dtype=dtype,out=out,keepdims=keepdims,initial=initial,where=where,promote_integers=promote_integers)

def cumprod(a, axis=None, dtype=None, out=None):
    if not rp.use_jax: return np.cumprod(a,axis=axis,dtype=dtype,out=out)
    else: return jnp.cumprod(a,axis=axis,dtype=dtype,out=out)

def cumsum(a, axis=None, dtype=None, out=None):
    if not rp.use_jax: return np.cumsum(a,axis=axis,dtype=dtype,out=out)
    else: return jnp.cumsum(a,axis=axis,dtype=dtype,out=out)

def nancumprod(a, axis=None, dtype=None, out=None):
    if not rp.use_jax: return np.nancumprod(a,axis=axis,dtype=dtype,out=out)
    else: return jnp.nancumprod(a,axis=axis,dtype=dtype,out=out)

def nancumsum(a, axis=None, dtype=None, out=None):
    if not rp.use_jax: return np.nancumsum(a,axis=axis,dtype=dtype,out=out)
    else: return jnp.nancumsum(a,axis=axis,dtype=dtype,out=out)

def diff(a, n=1, axis=-1, prepend=None, append=None):
    if not rp.use_jax: return np.diff(a,n=n,prepend=prepend,append=append)
    else: return jnp.diff(a,n=n,prepend=prepend,append=append)

def ediff1d(ary, to_end=None, to_begin=None):
    if not rp.use_jax: return np.ediff1d(ary,to_end=to_end,to_begin=to_begin)
    else: return jnp.ediff1d(ary,to_end=to_end,to_begin=to_begin)

def gradient(f, *varargs, axis=None, edge_order=None):
    if not rp.use_jax: return np.gradient(f,*varargs,axis=axis,edge_order=edge_order)
    else: return jnp.gradient(f,*varargs,axis=axis,edge_order=edge_order)

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if not rp.use_jax: return np.gradient(a,b,axisa=axisa,axisb=axisb,axisc=axisc,axis=axis)
    else: return jnp.gradient(a,b,axisa=axisa,axisb=axisb,axisc=axisc,axis=axis)

def exp(x,/): 
    if not rp.use_jax: return np.exp(x)
    else: return jnp.exp(x)

def expm1(x,/): 
    if not rp.use_jax: return np.expm1(x)
    else: return jnp.expm1(x)

def exp2(x,/): 
    if not rp.use_jax: return np.exp2(x)
    else: return jnp.exp2(x)

def log(x,/): 
    if not rp.use_jax: return np.log(x)
    else: return jnp.log(x)

def log10(x,/): 
    if not rp.use_jax: return np.log10(x)
    else: return jnp.log10(x)

def log2(x,/): 
    if not rp.use_jax: return np.log2(x)
    else: return jnp.log2(x)

def log1p(x,/): 
    if not rp.use_jax: return np.log1p(x)
    else: return jnp.log1p(x)

def logaddexp(x1,x2,/):
    if not rp.use_jax: return np.logaddexp(x1,x2)
    else: return jnp.logaddexp(x1,x2)

def logaddexp2(x1,x2,/):
    if not rp.use_jax: return np.logaddexp2(x1,x2)
    else: return jnp.logaddexp2(x1,x2)

def i0(x):
    if not rp.use_jax: return np.i0(x)
    else: return jnp.i0(x)

def sinc(x,/):
    if not rp.use_jax: return np.sinc(x)
    else: return jnp.sinc(x)

def signbitc(x,/):
    if not rp.use_jax: return np.signbit(x)
    else: return jnp.signbit(x)

def copysign(x1,x2,/):
    if not rp.use_jax: return np.copysign(x1,x2)
    else: return jnp.copysign(x1,x2)

def frexp(x,/):
    if not rp.use_jax: return np.frexp(x)
    else: return jnp.frexp(x)

def ldexp(x1,x2,/):
    if not rp.use_jax: return np.ldexp(x1,x2)
    else: return jnp.ldexp(x1,x2)

def nextafter(x1,x2,/):
    if not rp.use_jax: return np.nextafter(x1,x2)
    else: return jnp.nextafter(x1,x2)

def spacing(x,/):
    if not rp.use_jax: return np.spacing(x)
    else: return jnp.spacing(x)

def lcm(x1,x2):
    if not rp.use_jax: return np.lcm(x1,x2)
    else: return jnp.lcm(x1,x2)

def gcd(x1,x2):
    if not rp.use_jax: return np.gcd(x1,x2)
    else: return jnp.gcd(x1,x2)

def add(x1,x2,/):
    if not rp.use_jax: return np.add(x1,x2)
    else: return jnp.add(x1,x2)

def reciprocal(x,/):
    if not rp.use_jax: return np.reciprocal(x)
    else: return jnp.reciprocal(x)

def positive(x,/):
    if not rp.use_jax: return np.positive(x)
    else: return jnp.positive(x)
                               
def negative(x,/):
    if not rp.use_jax: return np.negative(x)
    else: return jnp.negative(x)

def multiply(x1,x2,/):
    if not rp.use_jax: return np.multiply(x1,x2)
    else: return jnp.multiply(x1,x2)

def divide(x1,x2,/):
    if not rp.use_jax: return np.divide(x1,x2)
    else: return jnp.divide(x1,x2)

def power(x1,x2,/):
    if not rp.use_jax: return np.power(x1,x2)
    else: return jnp.power(x1,x2)

def pow(x1,x2,/):
    if not rp.use_jax: return np.pow(x1,x2)
    else: return jnp.pow(x1,x2)

def subtract(x1,x2,/):
    if not rp.use_jax: return np.subtract(x1,x2)
    else: return jnp.subtract(x1,x2)

def true_divide(x1,x2,/):
    if not rp.use_jax: return np.true_divide(x1,x2)
    else: return jnp.true_divide(x1,x2)

def floor_divide(x1,x2,/):
    if not rp.use_jax: return np.floor_divide(x1,x2)
    else: return jnp.floor_divide(x1,x2)

def float_power(x1,x2,/):
    if not rp.use_jax: return np.float_power(x1,x2)
    else: return jnp.float_power(x1,x2)

def fmod(x1,x2,/):
    if not rp.use_jax: return np.fmod(x1,x2)
    else: return jnp.fmod(x1,x2)

def mod(x1,x2,/):
    if not rp.use_jax: return np.mod(x1,x2)
    else: return jnp.mod(x1,x2)

def modf(x,/,out=None):
    if not rp.use_jax: return np.modf(x,out=out)
    else: return jnp.modf(x,out=None)

def remainder(x1,x2,/):
    if not rp.use_jax: return np.remainder(x1,x2)
    else: return jnp.remainder(x1,x2)

def divmod(x1,x2,/):
    if not rp.use_jax: return np.divmod(x1,x2)
    else: return jnp.divmod(x1,x2)

def angle(z, deg=False):
    if not rp.use_jax: return np.angle(z,deg=deg)
    else: return jnp.angle(z,deg=deg)

def real(val,/):
    if not rp.use_jax: return np.real(val)
    else: return jnp.real(val)

def imag(val,/):
    if not rp.use_jax: return np.imag(val)
    else: return jnp.imag(val)

def conj(x,/):
    if not rp.use_jax: return np.conj(x)
    else: return jnp.conj(x)

def conjugate(x,/):
    if not rp.use_jax: return np.conjugate(x)
    else: return jnp.conjugate(x)
    
def maximum(x,y,/):
    if not rp.use_jax: return np.maximum(x,y)
    else: return jnp.maximum(x,y)

def max(a, axis=None, out=None, keepdims=False, initial=None, where=None):
    if not rp.use_jax: return np.max(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)
    else: return jnp.max(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)

def amax(a, axis=None, out=None, keepdims=False, initial=None, where=None):
    if not rp.use_jax: return np.amax(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)
    else: return jnp.amax(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)

def fmax(x1,x2):
    if not rp.use_jax: return np.fmax(x1,x2)
    else: return jnp.fmax(x1,x2)

def nanmax(a, axis=None, out=None, keepdims=False, initial=None, where=None):
    if not rp.use_jax: return np.nanmax(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)
    else: return jnp.nanmax(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)

def minimum(x,y,/):
    if not rp.use_jax: return np.minimum(x,y)
    else: return jnp.minimum(x,y)

def min(a, axis=None, out=None, keepdims=False, initial=None, where=None):
    if not rp.use_jax: return np.min(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)
    else: return jnp.min(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)

def amin(a, axis=None, out=None, keepdims=False, initial=None, where=None):
    if not rp.use_jax: return np.amin(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)
    else: return jnp.amin(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)

def fmin(x1,x2):
    if not rp.use_jax: return np.fmin(x1,x2)
    else: return jnp.fmin(x1,x2)

def nanmin(a, axis=None, out=None, keepdims=False, initial=None, where=None):
    if not rp.use_jax: return np.nanmin(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)
    else: return jnp.nanmin(a,axis=axis,out=out,keepdims=keepdims,initial=initial,where=where)

def convolve(a, v, mode='full', *, precision=None, preferred_element_type=None):
    if not rp.use_jax: return np.convolve(a,v,mode=mode)
    else: return jnp.convolve(a,v,mode=mode,precision=precision,preferred_element_type=preferred_element_type)

def clip(arr=None, /, min=None, max=None,):
    if not rp.use_jax: return np.clip(a,a_min=min,a_max=max)
    else: return jnp.clip(arr=arr, min=min, max=max)

def sqrt(x,/):
    if  not rp.use_jax:  return np.sqrt(x)
    else: return jnp.sqrt(x)

def cbrt(x,/):
    if  not rp.use_jax:  return np.cbrt(x)
    else: return jnp.cbrt(x)

def square(x,/):
    if  not rp.use_jax:  return np.square(x)
    else: return jnp.square(x)
    
def absolute(x,/):
    if  not rp.use_jax:  return np.absolute(x)
    else: return jnp.absolute(x)

def fab(x,/):
    if  not rp.use_jax:  return np.fabs(x)
    else: return jnp.fabs(x)

def sign(x,/):
    if  not rp.use_jax:  return np.sign(x)
    else: return jnp.sign(x)

def heaviside(x1,x2,/):
    if not rp.use_jax: return np.heaviside(x1,x2)
    else: return jnp.heaviside(x1,x2)

def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    if  not rp.use_jax:  return np.nan_to_num(x,copy=copy,nan=nan,posinf=posinf,neginf=neginf)
    else: return jnp.nan_to_num(x,copy=copy,nan=nan,posinf=posinf,neginf=neginf)

def real_if_close(): raise NotImplementedError # There is no JAX functionality

def interp(x, xp, fp, left=None, right=None, period=None):
    if not rp.use_jax: return np.interp(x,xp,fp,left=left,right=right,period=period)
    else: return jnp.interp(x,xp,fp,left=left,right=right,period=period)

def bitwise_count(x, /):
    if not rp.use_jax: return np.bitwise_count(x)
    else: return jnp.bitwise_count(x)

def vdot(a, b, *, precision=None, preferred_element_type=None):
    if not rp.use_jax: return np.vdot(a,b)
    else: return jnp.vdot(a,b,precision=precision,preferred_element_type=preferred_element_type)

def vecdot(a, b, *, precision=None, preferred_element_type=None):
    if not rp.use_jax: return np.vecdot(a,b)
    else: return jnp.vecdot(a,b,precision=precision,preferred_element_type=preferred_element_type)

def inner(a, b, *, precision=None, preferred_element_type=None):
    if not rp.use_jax: return np.inner(a,b)
    else: return jnp.inner(a,b,precision=precision,preferred_element_type=preferred_element_type)

def outer(a, b, out=None):
    if not rp.use_jax: return np.outer(a,b,out=out)
    else: return jnp.outer(a,b,out=out)

def matmul(a, b, *, precision=None, preferred_element_type=None):
    if not rp.use_jax: return np.inner(a,b)
    else: return jnp.inner(a,b,precision=precision,preferred_element_type=preferred_element_type)

def tensordot(a, b, axes=2, *, precision=None, preferred_element_type=None):
    if not rp.use_jax: return np.tensordot(a,b,axes=axes)
    else: return jnp.inner(a,b,axes=axes,precision=precision,preferred_element_type=preferred_element_type)

def einsum(subscript: str, /, *operands, out=None, optimize: str | bool | list[tuple[int, ...]] = 'optimal', precision =None, preferred_element_type =None):
    if not rp.use_jax: return np.einsum(subscript,*operands,out=out,order='K',casting='safe',optimize=optimize)
    else: return jnp.einsum(subscript,*operands,out=out,optimize=optimize,precision=precision,preferred_element_type=preferred_element_type)

def einsum_path(subscripts,/,*operands,optimize="greedy"):
    if not rp.use_jax: return np.einsum_path(subscripts,*operands,optimize=optimize)
    else: return jnp.einsum_path(subscripts,*operands,optimize=optimize)

def kron(a,b):
    if not rp.use_jax: return np.kron(a,b)
    else: return jnp.kron(a,b)

def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if not rp.use_jax: return np.trace(a, offset=offset,axis1=axis1,axis2=axis2,dtype=dtype,out=out)
    else: return jnp.kron(a, offset=offset,axis1=axis1,axis2=axis2,dtype=dtype,out=out)

def diagonal(a, offset=0, axis1=0, axis2=1):
    if not rp.use_jax: return np.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
    else: return jnp.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)

def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII', *, max_header_size=10000):
    if not rp.use_jax: return np.load(file, mmap_mode=mmap_mode, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding, max_header_size=max_header_size)
    else: return jnp.load(file, mmap_mode=mmap_mode, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding, max_header_size=max_header_size)

def save(file, arr, allow_pickle=True):
    if not rp.use_jax: return np.save(file, arr, allow_pickle=allow_pickle)
    else: return jnp.save(file, arr, allow_pickle=allow_pickle)

def savez(file, *args, **kwds):
    if not rp.use_jax: return np.savez(file,*args,**kwds)
    else: return jnp.savez(file,*args,**kwds)

def savez_compressed(): raise NotImplementedError

def loadtxt(): raise NotImplementedError

def savetxt(): raise NotImplementedError

def genfromtxt(): raise NotImplementedError

def fromregex(): raise NotImplementedError

def fromstring(string, dtype=float, count=-1, *, sep): 
    if not rp.use_jax: return np.fromstring(string, dtype=dtype, count=-1, sep=sep)
    else: return jnp.fromstring(string, dtype=dtype, count=-1, sep=sep)

def array2string(): raise NotImplementedError

def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    if not rp.use_jax: return np.array_repr(arr, max_line_width=max_line_width, precision=precision, suppress_small=suppress_small)
    else: return jnp.array_repr(arr, max_line_width=max_line_width, precision=precision, suppress_small=suppress_small)

def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    if not rp.use_jax: return np.array_str(a, max_line_width=max_line_width, precision=precision, suppress_small=suppress_small)
    else: return jnp.array_str(a, max_line_width=max_line_width, precision=precision, suppress_small=suppress_small)

def format_float_positional(): raise NotImplementedError

def format_float_scientific(): raise NotImplementedError

def memmap(): raise NotImplementedError

def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, \
                     formatter=None, sign=None, floatmode=None, *, legacy=None, override_repr=None):
    if not rp.use_jax: return np.set_printoptions(precision=precision, threshold=threshold, edgeitems=edgeitems, linewidth=linewidth, \
                                                  suppress=suppress, nanstr=nanstr, infstr=infstr, formatter=formatter, sign=sign, floatmode=floatmode, legacy=legacy)
    else: return jnp.set_printoptions(precision=precision, threshold=threshold, edgeitems=edgeitems, linewidth=linewidth, \
                                                  suppress=suppress, nanstr=nanstr, infstr=infstr, formatter=formatter, sign=sign, floatmode=floatmode, legacy=legacy, override_repr=override_repr)

def get_printoptions(): 
    if not rp.use_jax: return np.get_printoptions()
    else: return jnp.get_printoptions()

def printoptions(*args, **kwargs):
    if not rp.use_jax: np.printoptions(*args, **kwargs)
    else: jnp.printoptions(*args, **kwargs)

def binary_repr(): raise NotImplementedError

def base_repr(): raise NotImplementedError

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    if not rp.use_jax: return np.apply_along_axis(func1d, axis=axis, arr=arr, *args, **kwargs)
    else: return jnp.apply_along_axis(func1d, axis=axis, arr=arr, *args, **kwargs)

def apply_over_axes(func, a, axes):
    if not rp.use_jax: return np.apply_over_axes(func, a, axes)
    else: return jnp.apply_over_axes(func, a, axes)

def vectorize(pyfunc, *, excluded=frozenset({}), signature=None):
    if not rp.use_jax: return np.vectorize(pyfunc=pyfunc, otypes=None, doc=None, excluded=excluded, cache=False,signature=signature)
    else: return jnp.vectorize(pyfunc=pyfunc,excluded=excluded,signature=signature)

def frompyfunc(func, /, nin, nout, *, identity=None):
    if not rp.use_jax: return np.frompyfunc(func,nin,nout, identity=identity)
    else: return jnp.frompyfunc(func, nin, nout, identity=identity)

def piecewise(x, condlist, funclist, *args, **kw):
    if not rp.use_jax: return np.piecewise(x, condlist, funclist, *args, **kw)
    else: return jnp.piecewise(x, condlist, funclist, *args, **kw)

def empty(shape, dtype=float, *, device=None):
    if not rp.use_jax: return np.empty(shape,dtype=dtype,device=device)
    else: return jnp.empty(shape,dtype=dtype,device=device)

def empty_like(prototype, dtype=None, shape=None, *, device=None):
    if not rp.use_jax: return np.empty_like(prototype,dtype=dtype,shape=shape,device=device)
    else: return jnp.empty_like(prototype, dtype=dtype, shape=shape, device=device)

def eye(N, M=None, k=0, dtype=float, *, device=None):
    if not rp.use_jax: return np.eye(N,M=M,k=k,dtype=dtype,device=device)
    else: return jnp.eye(N,M=M,k=k,dtype=dtype,device=device)

def identity(n, dtype=None):
    if not rp.use_jax: return np.identity(n,dtype=dtype)
    else: return jnp.identity(n,dtype=dtype)

def ones(shape, dtype=None, *, device=None): 
    if not rp.use_jax: return np.ones(shape,dtype=dtype,device=device)
    else: return jnp.ones(shape, dtype=dtype, device=device)
 
def ones_like(a, dtype=None, shape=None, *, device=None):
    if not rp.use_jax: return np.ones_like(a, dtype=dtype, shape=shape, device=device)
    else: return jnp.ones_like(a, dtype=dtype, shape=shape, device=device)

def zeros(shape, dtype=None, *, device=None):
    if not rp.use_jax: return np.zeros(shape,dtype=dtype)
    else: return jnp.zeros(shape,dtype=dtype,device=device)

def zeros_like(a, dtype=None, shape=None, *, device=None):
    if not rp.use_jax: return np.zeros_like(a, dtype=dtype, shape=shape)
    else: return jnp.zeros_like(a, dtype=dtype, shape=shape, device=device)

def full(shape, fill_value, dtype=None, *, device=None):
    if not rp.use_jax: return np.full(shape, fill_value, dtype=dtype, device=device)
    else: return jnp.full(shape, fill_value, dtype=dtype, device=None)

def full_like(a, fill_value, dtype=None, shape=None, *, device=None):
    if not rp.use_jax: return np.full_like(a, fill_value, dtype=dtype, shape=shape, device=device)
    else: return jnp.full_like(a, fill_value, dtype=dtype, shape=shape, device=device)

def array(object, dtype=None, copy=True, order='K', ndmin=0, *, device=None):
    if not rp.use_jax: return np.array(object, dtype=dtype, copy=copy, order=order, ndmin=ndmin)
    else: return jnp.array(object, dtype=dtype, copy=copy, order=order, ndmin=ndim, device=device)

def asarray(a, dtype=None, order=None, *, copy=None, device=None):
    if not rp.use_jax: return np.asanyarray(a, dtype=dtype, order=order, device=device, copy=copy)
    else: return jnp.asarray(a, dtype=dtype, order=order, copy=copy, device=device)
	
def asanyarray(): raise NotImplementedError

def ascontiguousarray(): raise NotImplementedError

def asmatrix(): raise NotImplementedError

def astype(x, dtype, /, *, copy=False, device=None)[source]:
	if not rp.use_jax: return np.astype(x, dtype, copy=copy, device=device)
	else: return jnp.astype(x, dtype, copy=copy, device=device)

def copy(a, order='K'):
	if not rp.use_jax: return np.copy(a, order=order)
	else: return jnp.copy(a, order=order)

def frombuffer(buffer, dtype=float, count=-1, offset=0):
	if not rp.use_jax: return np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
	else: return jnp.frombuffer(buffer, dtype=dtype, count=count, offset=offset)

def from_dlpack(x, /, *, device=None, copy=None):
	if not rp.use_jax: return np.from_dlpack(x, device=device, copy=copy)
	else: return jnp.from_dlpack(x, device=device, copy=copy)

def fromfile(): raise NotImplementedError

def fromfunction(function, shape, *, dtype=float', **kwargs):
	if not rp.use_jax: return np.fromfunction(function, shape, dtype=dtype, like=None, **kwargs
	else: return jnp.fromfunction(function, shape,dtype=dtype **kwargs)

def fromiter(): raise NotImplementedError

def arange(start, stop=None, step=None, dtype=None, *, device=None):
	if not rp.use_jax:
         if stop is not None:
            return np.arange(start, stop=stop, step=step, dtype=dtype, device=device)
        else:
            return np.arange(stop=start, step=step, dtype=dtype, device=device)
	else: return jnp.arange(start, stop=stop, step=step, dtype=dtype, *, device=device)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None):
	if not rp.use_jax: return np.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis, device=device)
	else: return jnp.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis, device=device)

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
	if not rp.use_jax: return np.logspace(start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype, axis=axis)
	else: return jnp.logspace(start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype, axis=axis)

def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)):
	if not rp.use_jax: return np.geomspace(start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis)
	else: return jnp.geomspace(start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis)

def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
	if not rp.use_jax: return np.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
	else: return jnp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)

def mgrid(): raise NotImplementedError

def ogrid(): raise NotImplementedError

def diag(v, k=0):
	if not rp.use_jax: return np.diag(v, k=k)
	else: return jnp.diag(v, k=k)

def diagflat(v, k=0):
	if not rp.use_jax: return np.diagflat(v, k=k)
	else: return jnp.diagflatv, k=k)

def tri(N, M=None, k=0, dtype=float):
	if not rp.use_jax: return np.tri(N, M=M, k=k, dtype=dtype)
	else: return jnp.tri(N, M=M, k=k, dtype=dtype)

def tril(m, k=0):
	if not rp.use_jax: return np.tril(m, k=k)
	else: return jnp.tril(m, k=k)

def triu(m, k=0):
	if not rp.use_jax: return np.triu(m, k=k)
	else: return jnp.triu(m, k=k)

def vander(x, N=None, increasing=False):
	if not rp.use_jax: return np.vander(x, N=N, increasing=increasing)
	else: return jnp.vander(x, N=N, increasing=increasing)

def bmat(): raise NotImplementedError

# def bitwise_and(*args,**kwargs):
# 	if not rp.use_jax: return np.bitwise_and(*args,**kwargs)
# 	else: return jnp.bitwise_and(*args,**kwargs)

# def bitwise_or(*args,**kwargs):
# 	if not rp.use_jax: return np.bitwise_or(*args,**kwargs)
# 	else: return jnp.bitwise_or(*args,**kwargs)

# def bitwise_xor(*args,**kwargs):
# 	if not rp.use_jax: return np.bitwise_xor(*args,**kwargs)
# 	else: return jnp.bitwise_xor(*args,**kwargs)

# def invert(*args,**kwargs):
# 	if not rp.use_jax: return np.invert(*args,**kwargs)
# 	else: return jnp.invert(*args,**kwargs)

# def bitwise_invert(*args,**kwargs):
# 	if not rp.use_jax: return np.bitwise_invert(*args,**kwargs)
# 	else: return jnp.bitwise_invert(*args,**kwargs)

# def left_shift(*args,**kwargs):
# 	if not rp.use_jax: return np.left_shift(*args,**kwargs)
# 	else: return jnp.left_shift(*args,**kwargs)

# def bitwise_left_shift(*args,**kwargs):
# 	if not rp.use_jax: return np.bitwise_left_shift(*args,**kwargs)
# 	else: return jnp.bitwise_left_shift(*args,**kwargs)

# def right_shift(*args,**kwargs):
# 	if not rp.use_jax: return np.right_shift(*args,**kwargs)
# 	else: return jnp.right_shift(*args,**kwargs)

# def bitwise_right_shift(*args,**kwargs):
# 	if not rp.use_jax: return np.bitwise_right_shift(*args,**kwargs)
# 	else: return jnp.bitwise_right_shift(*args,**kwargs)

# def packbits(*args,**kwargs):
# 	if not rp.use_jax: return np.packbits(*args,**kwargs)
# 	else: return jnp.packbits(*args,**kwargs)

# def unpackbits(*args,**kwargs):
# 	if not rp.use_jax: return np.unpackbits(*args,**kwargs)
# 	else: return jnp.unpackbits(*args,**kwargs)

# def binary_repr(*args,**kwargs):
# 	if not rp.use_jax: return np.binary_repr(*args,**kwargs)
# 	else: return jnp.binary_repr(*args,**kwargs)

# def c_(*args,**kwargs):
# 	if not rp.use_jax: return np.c_(*args,**kwargs)
# 	else: return jnp.c_(*args,**kwargs)

# def r_(*args,**kwargs):
# 	if not rp.use_jax: return np.r_(*args,**kwargs)
# 	else: return jnp.r_(*args,**kwargs)

# def s_(*args,**kwargs):
# 	if not rp.use_jax: return np.s_(*args,**kwargs)
# 	else: return jnp.s_(*args,**kwargs)

# def nonzero(*args,**kwargs):
# 	if not rp.use_jax: return np.nonzero(*args,**kwargs)
# 	else: return jnp.nonzero(*args,**kwargs)

# def where(*args,**kwargs):
# 	if not rp.use_jax: return np.where(*args,**kwargs)
# 	else: return jnp.where(*args,**kwargs)

# def indices(*args,**kwargs):
# 	if not rp.use_jax: return np.indices(*args,**kwargs)
# 	else: return jnp.indices(*args,**kwargs)

# def ix_(*args,**kwargs):
# 	if not rp.use_jax: return np.ix_(*args,**kwargs)
# 	else: return jnp.ix_(*args,**kwargs)

# def ravel_multi_index(*args,**kwargs):
# 	if not rp.use_jax: return np.ravel_multi_index(*args,**kwargs)
# 	else: return jnp.ravel_multi_index(*args,**kwargs)

# def unravel_index(*args,**kwargs):
# 	if not rp.use_jax: return np.unravel_index(*args,**kwargs)
# 	else: return jnp.unravel_index(*args,**kwargs)

# def diag_indices(*args,**kwargs):
# 	if not rp.use_jax: return np.diag_indices(*args,**kwargs)
# 	else: return jnp.diag_indices(*args,**kwargs)

# def diag_indices_from(*args,**kwargs):
# 	if not rp.use_jax: return np.diag_indices_from(*args,**kwargs)
# 	else: return jnp.diag_indices_from(*args,**kwargs)

# def mask_indices(*args,**kwargs):
# 	if not rp.use_jax: return np.mask_indices(*args,**kwargs)
# 	else: return jnp.mask_indices(*args,**kwargs)

# def tril_indices(*args,**kwargs):
# 	if not rp.use_jax: return np.tril_indices(*args,**kwargs)
# 	else: return jnp.tril_indices(*args,**kwargs)

# def tril_indices_from(*args,**kwargs):
# 	if not rp.use_jax: return np.tril_indices_from(*args,**kwargs)
# 	else: return jnp.tril_indices_from(*args,**kwargs)

# def triu_indices(*args,**kwargs):
# 	if not rp.use_jax: return np.triu_indices(*args,**kwargs)
# 	else: return jnp.triu_indices(*args,**kwargs)

# def triu_indices_from(*args,**kwargs):
# 	if not rp.use_jax: return np.triu_indices_from(*args,**kwargs)
# 	else: return jnp.triu_indices_from(*args,**kwargs)

# def take(*args,**kwargs):
# 	if not rp.use_jax: return np.take(*args,**kwargs)
# 	else: return jnp.take(*args,**kwargs)

# def take_along_axis(*args,**kwargs):
# 	if not rp.use_jax: return np.take_along_axis(*args,**kwargs)
# 	else: return jnp.take_along_axis(*args,**kwargs)

# def choose(*args,**kwargs):
# 	if not rp.use_jax: return np.choose(*args,**kwargs)
# 	else: return jnp.choose(*args,**kwargs)

# def compress(*args,**kwargs):
# 	if not rp.use_jax: return np.compress(*args,**kwargs)
# 	else: return jnp.compress(*args,**kwargs)

# def diag(*args,**kwargs):
# 	if not rp.use_jax: return np.diag(*args,**kwargs)
# 	else: return jnp.diag(*args,**kwargs)

# def select(*args,**kwargs):
# 	if not rp.use_jax: return np.select(*args,**kwargs)
# 	else: return jnp.select(*args,**kwargs)

# def place(*args,**kwargs):
# 	if not rp.use_jax: return np.place(*args,**kwargs)
# 	else: return jnp.place(*args,**kwargs)

# def put(*args,**kwargs):
# 	if not rp.use_jax: return np.put(*args,**kwargs)
# 	else: return jnp.put(*args,**kwargs)

# def put_along_axis(*args,**kwargs):
# 	if not rp.use_jax: return np.put_along_axis(*args,**kwargs)
# 	else: return jnp.put_along_axis(*args,**kwargs)

# def putmask(*args,**kwargs):
# 	if not rp.use_jax: return np.putmask(*args,**kwargs)
# 	else: return jnp.putmask(*args,**kwargs)

# def fill_diagonal(*args,**kwargs):
# 	if not rp.use_jax: return np.fill_diagonal(*args,**kwargs)
# 	else: return jnp.fill_diagonal(*args,**kwargs)

# def nditer(*args,**kwargs):
# 	if not rp.use_jax: return np.nditer(*args,**kwargs)
# 	else: return jnp.nditer(*args,**kwargs)

# def ndenumerate(*args,**kwargs):
# 	if not rp.use_jax: return np.ndenumerate(*args,**kwargs)
# 	else: return jnp.ndenumerate(*args,**kwargs)

# def ndindex(*args,**kwargs):
# 	if not rp.use_jax: return np.ndindex(*args,**kwargs)
# 	else: return jnp.ndindex(*args,**kwargs)

# def nested_iters(*args,**kwargs):
# 	if not rp.use_jax: return np.nested_iters(*args,**kwargs)
# 	else: return jnp.nested_iters(*args,**kwargs)

# def flatiter(*args,**kwargs):
# 	if not rp.use_jax: return np.flatiter(*args,**kwargs)
# 	else: return jnp.flatiter(*args,**kwargs)

# def iterable(*args,**kwargs):
# 	if not rp.use_jax: return np.iterable(*args,**kwargs)
# 	else: return jnp.iterable(*args,**kwargs)

# def unique(*args,**kwargs):
# 	if not rp.use_jax: return np.unique(*args,**kwargs)
# 	else: return jnp.unique(*args,**kwargs)

# def unique_all(*args,**kwargs):
# 	if not rp.use_jax: return np.unique_all(*args,**kwargs)
# 	else: return jnp.unique_all(*args,**kwargs)

# def unique_counts(*args,**kwargs):
# 	if not rp.use_jax: return np.unique_counts(*args,**kwargs)
# 	else: return jnp.unique_counts(*args,**kwargs)

# def unique_inverse(*args,**kwargs):
# 	if not rp.use_jax: return np.unique_inverse(*args,**kwargs)
# 	else: return jnp.unique_inverse(*args,**kwargs)

# def unique_values(*args,**kwargs):
# 	if not rp.use_jax: return np.unique_values(*args,**kwargs)
# 	else: return jnp.unique_values(*args,**kwargs)

# def in1d(*args,**kwargs):
# 	if not rp.use_jax: return np.in1d(*args,**kwargs)
# 	else: return jnp.in1d(*args,**kwargs)

# def intersect1d(*args,**kwargs):
# 	if not rp.use_jax: return np.intersect1d(*args,**kwargs)
# 	else: return jnp.intersect1d(*args,**kwargs)

# def isin(*args,**kwargs):
# 	if not rp.use_jax: return np.isin(*args,**kwargs)
# 	else: return jnp.isin(*args,**kwargs)

# def setdiff1d(*args,**kwargs):
# 	if not rp.use_jax: return np.setdiff1d(*args,**kwargs)
# 	else: return jnp.setdiff1d(*args,**kwargs)

# def setxor1d(*args,**kwargs):
# 	if not rp.use_jax: return np.setxor1d(*args,**kwargs)
# 	else: return jnp.setxor1d(*args,**kwargs)

# def union1d(*args,**kwargs):
# 	if not rp.use_jax: return np.union1d(*args,**kwargs)
# 	else: return jnp.union1d(*args,**kwargs)

# def sort(*args,**kwargs):
# 	if not rp.use_jax: return np.sort(*args,**kwargs)
# 	else: return jnp.sort(*args,**kwargs)

# def lexsort(*args,**kwargs):
# 	if not rp.use_jax: return np.lexsort(*args,**kwargs)
# 	else: return jnp.lexsort(*args,**kwargs)

# def argsort(*args,**kwargs):
# 	if not rp.use_jax: return np.argsort(*args,**kwargs)
# 	else: return jnp.argsort(*args,**kwargs)

# def sort_complex(*args,**kwargs):
# 	if not rp.use_jax: return np.sort_complex(*args,**kwargs)
# 	else: return jnp.sort_complex(*args,**kwargs)

# def partition(*args,**kwargs):
# 	if not rp.use_jax: return np.partition(*args,**kwargs)
# 	else: return jnp.partition(*args,**kwargs)

# def argpartition(*args,**kwargs):
# 	if not rp.use_jax: return np.argpartition(*args,**kwargs)
# 	else: return jnp.argpartition(*args,**kwargs)

# def argmax(*args,**kwargs):
# 	if not rp.use_jax: return np.argmax(*args,**kwargs)
# 	else: return jnp.argmax(*args,**kwargs)

# def nanargmax(*args,**kwargs):
# 	if not rp.use_jax: return np.nanargmax(*args,**kwargs)
# 	else: return jnp.nanargmax(*args,**kwargs)

# def argmin(*args,**kwargs):
# 	if not rp.use_jax: return np.argmin(*args,**kwargs)
# 	else: return jnp.argmin(*args,**kwargs)

# def nanargmin(*args,**kwargs):
# 	if not rp.use_jax: return np.nanargmin(*args,**kwargs)
# 	else: return jnp.nanargmin(*args,**kwargs)

# def argwhere(*args,**kwargs):
# 	if not rp.use_jax: return np.argwhere(*args,**kwargs)
# 	else: return jnp.argwhere(*args,**kwargs)

# def nonzero(*args,**kwargs):
# 	if not rp.use_jax: return np.nonzero(*args,**kwargs)
# 	else: return jnp.nonzero(*args,**kwargs)

# def flatnonzero(*args,**kwargs):
# 	if not rp.use_jax: return np.flatnonzero(*args,**kwargs)
# 	else: return jnp.flatnonzero(*args,**kwargs)

# def where(*args,**kwargs):
# 	if not rp.use_jax: return np.where(*args,**kwargs)
# 	else: return jnp.where(*args,**kwargs)

# def searchsorted(*args,**kwargs):
# 	if not rp.use_jax: return np.searchsorted(*args,**kwargs)
# 	else: return jnp.searchsorted(*args,**kwargs)

# def extract(*args,**kwargs):
# 	if not rp.use_jax: return np.extract(*args,**kwargs)
# 	else: return jnp.extract(*args,**kwargs)

# def count_nonzero(*args,**kwargs):
# 	if not rp.use_jax: return np.count_nonzero(*args,**kwargs)
# 	else: return jnp.count_nonzero(*args,**kwargs)

# def ptp(*args,**kwargs):
# 	if not rp.use_jax: return np.ptp(*args,**kwargs)
# 	else: return jnp.ptp(*args,**kwargs)

# def percentile(*args,**kwargs):
# 	if not rp.use_jax: return np.percentile(*args,**kwargs)
# 	else: return jnp.percentile(*args,**kwargs)

# def nanpercentile(*args,**kwargs):
# 	if not rp.use_jax: return np.nanpercentile(*args,**kwargs)
# 	else: return jnp.nanpercentile(*args,**kwargs)

# def quantile(*args,**kwargs):
# 	if not rp.use_jax: return np.quantile(*args,**kwargs)
# 	else: return jnp.quantile(*args,**kwargs)

# def nanquantile(*args,**kwargs):
# 	if not rp.use_jax: return np.nanquantile(*args,**kwargs)
# 	else: return jnp.nanquantile(*args,**kwargs)

# def median(*args,**kwargs):
# 	if not rp.use_jax: return np.median(*args,**kwargs)
# 	else: return jnp.median(*args,**kwargs)

# def average(*args,**kwargs):
# 	if not rp.use_jax: return np.average(*args,**kwargs)
# 	else: return jnp.average(*args,**kwargs)

# def mean(*args,**kwargs):
# 	if not rp.use_jax: return np.mean(*args,**kwargs)
# 	else: return jnp.mean(*args,**kwargs)

# def std(*args,**kwargs):
# 	if not rp.use_jax: return np.std(*args,**kwargs)
# 	else: return jnp.std(*args,**kwargs)

# def var(*args,**kwargs):
# 	if not rp.use_jax: return np.var(*args,**kwargs)
# 	else: return jnp.var(*args,**kwargs)

# def nanmedian(*args,**kwargs):
# 	if not rp.use_jax: return np.nanmedian(*args,**kwargs)
# 	else: return jnp.nanmedian(*args,**kwargs)

# def nanmean(*args,**kwargs):
# 	if not rp.use_jax: return np.nanmean(*args,**kwargs)
# 	else: return jnp.nanmean(*args,**kwargs)

# def nanstd(*args,**kwargs):
# 	if not rp.use_jax: return np.nanstd(*args,**kwargs)
# 	else: return jnp.nanstd(*args,**kwargs)

# def nanvar(*args,**kwargs):
# 	if not rp.use_jax: return np.nanvar(*args,**kwargs)
# 	else: return jnp.nanvar(*args,**kwargs)

# def corrcoef(*args,**kwargs):
# 	if not rp.use_jax: return np.corrcoef(*args,**kwargs)
# 	else: return jnp.corrcoef(*args,**kwargs)

# def correlate(*args,**kwargs):
# 	if not rp.use_jax: return np.correlate(*args,**kwargs)
# 	else: return jnp.correlate(*args,**kwargs)

# def cov(*args,**kwargs):
# 	if not rp.use_jax: return np.cov(*args,**kwargs)
# 	else: return jnp.cov(*args,**kwargs)

# def histogram(*args,**kwargs):
# 	if not rp.use_jax: return np.histogram(*args,**kwargs)
# 	else: return jnp.histogram(*args,**kwargs)

# def histogram2d(*args,**kwargs):
# 	if not rp.use_jax: return np.histogram2d(*args,**kwargs)
# 	else: return jnp.histogram2d(*args,**kwargs)

# def histogramdd(*args,**kwargs):
# 	if not rp.use_jax: return np.histogramdd(*args,**kwargs)
# 	else: return jnp.histogramdd(*args,**kwargs)

# def bincount(*args,**kwargs):
# 	if not rp.use_jax: return np.bincount(*args,**kwargs)
# 	else: return jnp.bincount(*args,**kwargs)

# def histogram_bin_edges(*args,**kwargs):
# 	if not rp.use_jax: return np.histogram_bin_edges(*args,**kwargs)
# 	else: return jnp.histogram_bin_edges(*args,**kwargs)

# def digitize(*args,**kwargs):
# 	if not rp.use_jax: return np.digitize(*args,**kwargs)
# 	else: return jnp.digitize(*args,**kwargs)

# def copyto(*args,**kwargs):
# 	if not rp.use_jax: return np.copyto(*args,**kwargs)
# 	else: return jnp.copyto(*args,**kwargs)

# def ndim(*args,**kwargs):
# 	if not rp.use_jax: return np.ndim(*args,**kwargs)
# 	else: return jnp.ndim(*args,**kwargs)

# def shape(*args,**kwargs):
# 	if not rp.use_jax: return np.shape(*args,**kwargs)
# 	else: return jnp.shape(*args,**kwargs)

# def size(*args,**kwargs):
# 	if not rp.use_jax: return np.size(*args,**kwargs)
# 	else: return jnp.size(*args,**kwargs)

# def reshape(*args,**kwargs):
# 	if not rp.use_jax: return np.reshape(*args,**kwargs)
# 	else: return jnp.reshape(*args,**kwargs)

# def ravel(*args,**kwargs):
# 	if not rp.use_jax: return np.ravel(*args,**kwargs)
# 	else: return jnp.ravel(*args,**kwargs)

# def moveaxis(*args,**kwargs):
# 	if not rp.use_jax: return np.moveaxis(*args,**kwargs)
# 	else: return jnp.moveaxis(*args,**kwargs)

# def rollaxis(*args,**kwargs):
# 	if not rp.use_jax: return np.rollaxis(*args,**kwargs)
# 	else: return jnp.rollaxis(*args,**kwargs)

# def swapaxes(*args,**kwargs):
# 	if not rp.use_jax: return np.swapaxes(*args,**kwargs)
# 	else: return jnp.swapaxes(*args,**kwargs)

# def transpose(*args,**kwargs):
# 	if not rp.use_jax: return np.transpose(*args,**kwargs)
# 	else: return jnp.transpose(*args,**kwargs)

# def permute_dims(*args,**kwargs):
# 	if not rp.use_jax: return np.permute_dims(*args,**kwargs)
# 	else: return jnp.permute_dims(*args,**kwargs)

# def matrix_transpose(*args,**kwargs):
# 	if not rp.use_jax: return np.matrix_transpose(*args,**kwargs)
# 	else: return jnp.matrix_transpose(*args,**kwargs)

# def atleast_1d(*args,**kwargs):
# 	if not rp.use_jax: return np.atleast_1d(*args,**kwargs)
# 	else: return jnp.atleast_1d(*args,**kwargs)

# def atleast_2d(*args,**kwargs):
# 	if not rp.use_jax: return np.atleast_2d(*args,**kwargs)
# 	else: return jnp.atleast_2d(*args,**kwargs)

# def atleast_3d(*args,**kwargs):
# 	if not rp.use_jax: return np.atleast_3d(*args,**kwargs)
# 	else: return jnp.atleast_3d(*args,**kwargs)

# def broadcast(*args,**kwargs):
# 	if not rp.use_jax: return np.broadcast(*args,**kwargs)
# 	else: return jnp.broadcast(*args,**kwargs)

# def broadcast_to(*args,**kwargs):
# 	if not rp.use_jax: return np.broadcast_to(*args,**kwargs)
# 	else: return jnp.broadcast_to(*args,**kwargs)

# def broadcast_arrays(*args,**kwargs):
# 	if not rp.use_jax: return np.broadcast_arrays(*args,**kwargs)
# 	else: return jnp.broadcast_arrays(*args,**kwargs)

# def expand_dims(*args,**kwargs):
# 	if not rp.use_jax: return np.expand_dims(*args,**kwargs)
# 	else: return jnp.expand_dims(*args,**kwargs)

# def squeeze(*args,**kwargs):
# 	if not rp.use_jax: return np.squeeze(*args,**kwargs)
# 	else: return jnp.squeeze(*args,**kwargs)

# def asanyarray(*args,**kwargs):
# 	if not rp.use_jax: return np.asanyarray(*args,**kwargs)
# 	else: return jnp.asanyarray(*args,**kwargs)

# def asmatrix(*args,**kwargs):
# 	if not rp.use_jax: return np.asmatrix(*args,**kwargs)
# 	else: return jnp.asmatrix(*args,**kwargs)

# def asfortranarray(*args,**kwargs):
# 	if not rp.use_jax: return np.asfortranarray(*args,**kwargs)
# 	else: return jnp.asfortranarray(*args,**kwargs)

# def ascontiguousarray(*args,**kwargs):
# 	if not rp.use_jax: return np.ascontiguousarray(*args,**kwargs)
# 	else: return jnp.ascontiguousarray(*args,**kwargs)

# def asarray_chkfinite(*args,**kwargs):
# 	if not rp.use_jax: return np.asarray_chkfinite(*args,**kwargs)
# 	else: return jnp.asarray_chkfinite(*args,**kwargs)

# def require(*args,**kwargs):
# 	if not rp.use_jax: return np.require(*args,**kwargs)
# 	else: return jnp.require(*args,**kwargs)

# def concatenate(*args,**kwargs):
# 	if not rp.use_jax: return np.concatenate(*args,**kwargs)
# 	else: return jnp.concatenate(*args,**kwargs)

# def concat(*args,**kwargs):
# 	if not rp.use_jax: return np.concat(*args,**kwargs)
# 	else: return jnp.concat(*args,**kwargs)

# def stack(*args,**kwargs):
# 	if not rp.use_jax: return np.stack(*args,**kwargs)
# 	else: return jnp.stack(*args,**kwargs)

# def block(*args,**kwargs):
# 	if not rp.use_jax: return np.block(*args,**kwargs)
# 	else: return jnp.block(*args,**kwargs)

# def vstack(*args,**kwargs):
# 	if not rp.use_jax: return np.vstack(*args,**kwargs)
# 	else: return jnp.vstack(*args,**kwargs)

# def hstack(*args,**kwargs):
# 	if not rp.use_jax: return np.hstack(*args,**kwargs)
# 	else: return jnp.hstack(*args,**kwargs)

# def dstack(*args,**kwargs):
# 	if not rp.use_jax: return np.dstack(*args,**kwargs)
# 	else: return jnp.dstack(*args,**kwargs)

# def column_stack(*args,**kwargs):
# 	if not rp.use_jax: return np.column_stack(*args,**kwargs)
# 	else: return jnp.column_stack(*args,**kwargs)

# def split(*args,**kwargs):
# 	if not rp.use_jax: return np.split(*args,**kwargs)
# 	else: return jnp.split(*args,**kwargs)

# def array_split(*args,**kwargs):
# 	if not rp.use_jax: return np.array_split(*args,**kwargs)
# 	else: return jnp.array_split(*args,**kwargs)

# def dsplit(*args,**kwargs):
# 	if not rp.use_jax: return np.dsplit(*args,**kwargs)
# 	else: return jnp.dsplit(*args,**kwargs)

# def hsplit(*args,**kwargs):
# 	if not rp.use_jax: return np.hsplit(*args,**kwargs)
# 	else: return jnp.hsplit(*args,**kwargs)

# def vsplit(*args,**kwargs):
# 	if not rp.use_jax: return np.vsplit(*args,**kwargs)
# 	else: return jnp.vsplit(*args,**kwargs)

# def tile(*args,**kwargs):
# 	if not rp.use_jax: return np.tile(*args,**kwargs)
# 	else: return jnp.tile(*args,**kwargs)

# def repeat(*args,**kwargs):
# 	if not rp.use_jax: return np.repeat(*args,**kwargs)
# 	else: return jnp.repeat(*args,**kwargs)

# def delete(*args,**kwargs):
# 	if not rp.use_jax: return np.delete(*args,**kwargs)
# 	else: return jnp.delete(*args,**kwargs)

# def insert(*args,**kwargs):
# 	if not rp.use_jax: return np.insert(*args,**kwargs)
# 	else: return jnp.insert(*args,**kwargs)

# def append(*args,**kwargs):
# 	if not rp.use_jax: return np.append(*args,**kwargs)
# 	else: return jnp.append(*args,**kwargs)

# def resize(*args,**kwargs):
# 	if not rp.use_jax: return np.resize(*args,**kwargs)
# 	else: return jnp.resize(*args,**kwargs)

# def trim_zeros(*args,**kwargs):
# 	if not rp.use_jax: return np.trim_zeros(*args,**kwargs)
# 	else: return jnp.trim_zeros(*args,**kwargs)

# def unique(*args,**kwargs):
# 	if not rp.use_jax: return np.unique(*args,**kwargs)
# 	else: return jnp.unique(*args,**kwargs)

# def pad(*args,**kwargs):
# 	if not rp.use_jax: return np.pad(*args,**kwargs)
# 	else: return jnp.pad(*args,**kwargs)

# def flip(*args,**kwargs):
# 	if not rp.use_jax: return np.flip(*args,**kwargs)
# 	else: return jnp.flip(*args,**kwargs)

# def fliplr(*args,**kwargs):
# 	if not rp.use_jax: return np.fliplr(*args,**kwargs)
# 	else: return jnp.fliplr(*args,**kwargs)

# def flipud(*args,**kwargs):
# 	if not rp.use_jax: return np.flipud(*args,**kwargs)
# 	else: return jnp.flipud(*args,**kwargs)

# def reshape(*args,**kwargs):
# 	if not rp.use_jax: return np.reshape(*args,**kwargs)
# 	else: return jnp.reshape(*args,**kwargs)

# def roll(*args,**kwargs):
# 	if not rp.use_jax: return np.roll(*args,**kwargs)
# 	else: return jnp.roll(*args,**kwargs)

# def rot90(*args,**kwargs):
# 	if not rp.use_jax: return np.rot90(*args,**kwargs)
# 	else: return jnp.rot90(*args,**kwargs)


# def 	lib.npyio.NpzFile	(): raise NotImplementedError
# def 	rec.array	(): raise NotImplementedError
# def 	rec.fromarrays	(): raise NotImplementedError
# def 	rec.fromrecords	(): raise NotImplementedError
# def 	rec.fromstring	(): raise NotImplementedError
# def 	rec.fromfile	(): raise NotImplementedError
# def 	char.array	(): raise NotImplementedError
# def 	char.asarray	(): raise NotImplementedError
# def 	ndarray.sort	(): raise NotImplementedError
# def 	ndarray.flat	(): raise NotImplementedError
# def 	ndarray.flatten	(): raise NotImplementedError
# def 	ndarray.T	(): raise NotImplementedError
# def 	ndarray.tofile	(): raise NotImplementedError
# def 	ndarray.tolist	(): raise NotImplementedError
# def 	lib.format.open_memmap	(): raise NotImplementedError
# def 	lib.npyio.DataSource	(): raise NotImplementedError
# def 	lib.form	(): raise NotImplementedError