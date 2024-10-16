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
    if not rp.use_jax: return np.array(object, dtype=dtype, copy=copy, order=order, ndmin=ndim, device=device)
    else: return jnp.array(object, dtype=dtype, copy=copy, order=order, ndmin=ndim, device=device)

def 	asarray(a, dtype=None, order=None, *, copy=None, device=None): raise NotImplementedError
def 	asanyarray	(): raise NotImplementedError
def 	ascontiguousarray	(): raise NotImplementedError
def 	asmatrix	(): raise NotImplementedError
def 	astype	(): raise NotImplementedError
def 	copy	(): raise NotImplementedError
def 	frombuffer	(): raise NotImplementedError
def 	from_dlpack	(): raise NotImplementedError
def 	fromfile	(): raise NotImplementedError
def 	fromfunction	(): raise NotImplementedError
def 	fromiter	(): raise NotImplementedError
def 	fromstring	(): raise NotImplementedError
def 	loadtxt	(): raise NotImplementedError
# def 	rec.array	(): raise NotImplementedError
# def 	rec.fromarrays	(): raise NotImplementedError
# def 	rec.fromrecords	(): raise NotImplementedError
# def 	rec.fromstring	(): raise NotImplementedError
# def 	rec.fromfile	(): raise NotImplementedError
# def 	char.array	(): raise NotImplementedError
# def 	char.asarray	(): raise NotImplementedError
def 	arange	(): raise NotImplementedError
def 	linspace	(): raise NotImplementedError
def 	logspace	(): raise NotImplementedError
def 	geomspace	(): raise NotImplementedError
def 	meshgrid	(): raise NotImplementedError
def 	mgrid	(): raise NotImplementedError
def 	ogrid	(): raise NotImplementedError
def 	diag	(): raise NotImplementedError
def 	diagflat	(): raise NotImplementedError
def 	tri	(): raise NotImplementedError
def 	tril	(): raise NotImplementedError
def 	triu	(): raise NotImplementedError
def 	vander	(): raise NotImplementedError
def 	bmat	(): raise NotImplementedError
def 	bitwise_and	(): raise NotImplementedError
def 	bitwise_or	(): raise NotImplementedError
def 	bitwise_xor	(): raise NotImplementedError
def 	invert	(): raise NotImplementedError
def 	bitwise_invert	(): raise NotImplementedError
def 	left_shift	(): raise NotImplementedError
def 	bitwise_left_shift	(): raise NotImplementedError
def 	right_shift	(): raise NotImplementedError
def 	bitwise_right_shift	(): raise NotImplementedError
def 	packbits	(): raise NotImplementedError
def 	unpackbits	(): raise NotImplementedError
def 	binary_repr	(): raise NotImplementedError
def 	c_	(): raise NotImplementedError
def 	r_	(): raise NotImplementedError
def 	s_	(): raise NotImplementedError
def 	nonzero	(): raise NotImplementedError
def 	where	(): raise NotImplementedError
def 	indices	(): raise NotImplementedError
def 	ix_	(): raise NotImplementedError
def 	ravel_multi_index	(): raise NotImplementedError
def 	unravel_index	(): raise NotImplementedError
def 	diag_indices	(): raise NotImplementedError
def 	diag_indices_from	(): raise NotImplementedError
def 	mask_indices	(): raise NotImplementedError
def 	tril_indices	(): raise NotImplementedError
def 	tril_indices_from	(): raise NotImplementedError
def 	triu_indices	(): raise NotImplementedError
def 	triu_indices_from	(): raise NotImplementedError
def 	take	(): raise NotImplementedError
def 	take_along_axis	(): raise NotImplementedError
def 	choose	(): raise NotImplementedError
def 	compress	(): raise NotImplementedError
def 	diag	(): raise NotImplementedError
def 	select	(): raise NotImplementedError
def 	place	(): raise NotImplementedError
def 	put	(): raise NotImplementedError
def 	put_along_axis	(): raise NotImplementedError
def 	putmask	(): raise NotImplementedError
def 	fill_diagonal	(): raise NotImplementedError
def 	nditer	(): raise NotImplementedError
def 	ndenumerate	(): raise NotImplementedError
def 	ndindex	(): raise NotImplementedError
def 	nested_iters	(): raise NotImplementedError
def 	flatiter	(): raise NotImplementedError
def 	iterable	(): raise NotImplementedError
def 	unique	(): raise NotImplementedError
def 	unique_all	(): raise NotImplementedError
def 	unique_counts	(): raise NotImplementedError
def 	unique_inverse	(): raise NotImplementedError
def 	unique_values	(): raise NotImplementedError
def 	in1d	(): raise NotImplementedError
def 	intersect1d	(): raise NotImplementedError
def 	isin	(): raise NotImplementedError
def 	setdiff1d	(): raise NotImplementedError
def 	setxor1d	(): raise NotImplementedError
def 	union1d	(): raise NotImplementedError
def 	sort	(): raise NotImplementedError
def 	lexsort	(): raise NotImplementedError
def 	argsort	(): raise NotImplementedError
# def 	ndarray.sort	(): raise NotImplementedError
def 	sort_complex	(): raise NotImplementedError
def 	partition	(): raise NotImplementedError
def 	argpartition	(): raise NotImplementedError
def 	argmax	(): raise NotImplementedError
def 	nanargmax	(): raise NotImplementedError
def 	argmin	(): raise NotImplementedError
def 	nanargmin	(): raise NotImplementedError
def 	argwhere	(): raise NotImplementedError
def 	nonzero	(): raise NotImplementedError
def 	flatnonzero	(): raise NotImplementedError
def 	where	(): raise NotImplementedError
def 	searchsorted	(): raise NotImplementedError
def 	extract	(): raise NotImplementedError
def 	count_nonzero	(): raise NotImplementedError
def 	ptp	(): raise NotImplementedError
def 	percentile	(): raise NotImplementedError
def 	nanpercentile	(): raise NotImplementedError
def 	quantile	(): raise NotImplementedError
def 	nanquantile	(): raise NotImplementedError
def 	median	(): raise NotImplementedError
def 	average	(): raise NotImplementedError
def 	mean	(): raise NotImplementedError
def 	std	(): raise NotImplementedError
def 	var	(): raise NotImplementedError
def 	nanmedian	(): raise NotImplementedError
def 	nanmean	(): raise NotImplementedError
def 	nanstd	(): raise NotImplementedError
def 	nanvar	(): raise NotImplementedError
def 	corrcoef	(): raise NotImplementedError
def 	correlate	(): raise NotImplementedError
def 	cov	(): raise NotImplementedError
def 	histogram	(): raise NotImplementedError
def 	histogram2d	(): raise NotImplementedError
def 	histogramdd	(): raise NotImplementedError
def 	bincount	(): raise NotImplementedError
def 	histogram_bin_edges	(): raise NotImplementedError
def 	digitize	(): raise NotImplementedError
def 	copyto	(): raise NotImplementedError
def 	ndim	(): raise NotImplementedError
def 	shape	(): raise NotImplementedError
def 	size	(): raise NotImplementedError
def 	reshape	(): raise NotImplementedError
def 	ravel	(): raise NotImplementedError
# def 	ndarray.flat	(): raise NotImplementedError
# def 	ndarray.flatten	(): raise NotImplementedError
def 	moveaxis	(): raise NotImplementedError
def 	rollaxis	(): raise NotImplementedError
def 	swapaxes	(): raise NotImplementedError
def 	ndarray.T	(): raise NotImplementedError
def 	transpose	(): raise NotImplementedError
def 	permute_dims	(): raise NotImplementedError
def 	matrix_transpose	(): raise NotImplementedError
def 	atleast_1d	(): raise NotImplementedError
def 	atleast_2d	(): raise NotImplementedError
def 	atleast_3d	(): raise NotImplementedError
def 	broadcast	(): raise NotImplementedError
def 	broadcast_to	(): raise NotImplementedError
def 	broadcast_arrays	(): raise NotImplementedError
def 	expand_dims	(): raise NotImplementedError
def 	squeeze	(): raise NotImplementedError
def 	asarray	(): raise NotImplementedError
def 	asanyarray	(): raise NotImplementedError
def 	asmatrix	(): raise NotImplementedError
def 	asfortranarray	(): raise NotImplementedError
def 	ascontiguousarray	(): raise NotImplementedError
def 	asarray_chkfinite	(): raise NotImplementedError
def 	require	(): raise NotImplementedError
def 	concatenate	(): raise NotImplementedError
def 	concat	(): raise NotImplementedError
def 	stack	(): raise NotImplementedError
def 	block	(): raise NotImplementedError
def 	vstack	(): raise NotImplementedError
def 	hstack	(): raise NotImplementedError
def 	dstack	(): raise NotImplementedError
def 	column_stack	(): raise NotImplementedError
def 	split	(): raise NotImplementedError
def 	array_split	(): raise NotImplementedError
def 	dsplit	(): raise NotImplementedError
def 	hsplit	(): raise NotImplementedError
def 	vsplit	(): raise NotImplementedError
def 	tile	(): raise NotImplementedError
def 	repeat	(): raise NotImplementedError
def 	delete	(): raise NotImplementedError
def 	insert	(): raise NotImplementedError
def 	append	(): raise NotImplementedError
def 	resize	(): raise NotImplementedError
def 	trim_zeros	(): raise NotImplementedError
def 	unique	(): raise NotImplementedError
def 	pad	(): raise NotImplementedError
def 	flip	(): raise NotImplementedError
def 	fliplr	(): raise NotImplementedError
def 	flipud	(): raise NotImplementedError
def 	reshape	(): raise NotImplementedError
def 	roll	(): raise NotImplementedError
def 	rot90	(): raise NotImplementedError

# def 	lib.npyio.NpzFile	(): raise NotImplementedError
# def 	ndarray.tofile	(): raise NotImplementedError
# def 	ndarray.tolist	(): raise NotImplementedError
# def 	lib.format.open_memmap	(): raise NotImplementedError
# def 	lib.npyio.DataSource	(): raise NotImplementedError
# def 	lib.form	(): raise NotImplementedError