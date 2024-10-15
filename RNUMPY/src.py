# src.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Aug 2024 E. Botero
# Modified: 

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORTS
# ----------------------------------------------------------------------------------------------------------------------  


import RNUMPY as rp

j   = rp.jax_handle
np  = rp.numpy_handle
jnp = j.numpy

# ----------------------------------------------------------------------------------------------------------------------
#  debug print function
# ----------------------------------------------------------------------------------------------------------------------  

def debugprint(fmt, *args, ordered=False, **kwargs):
    if rp.use_jax: j.debug.print(fmt,*args,ordered=ordered,**kwargs)
    else: print(fmt.format(*args, **kwargs))

# ----------------------------------------------------------------------------------------------------------------------
#  .at functions
# ----------------------------------------------------------------------------------------------------------------------  

def set(a,indices,b):
    if rp.use_jax: return a.at[indices].set(b)
    else: a[indices] = b; return a 

def get(a,indices):
    if rp.use_jax: return a.at[indices].get()
    else: return a[indices]

def pequals(a,indices,b):
    if rp.use_jax: return a.at[indices].add(b)
    else: a[indices] += b; return a

def sequals(a,indices,b):
    if rp.use_jax: return a.at[indices].minus(b)
    else: a[indices] -= b; return a

def mequals(a,indices,b):
    if rp.use_jax: return a.at[indices].multiply(b)
    else: a[indices] *= b; return a

def dequals(a,indices,b):
    if rp.use_jax: return a.at[indices].divide(b)
    else: a[indices] /= b; return a


# ----------------------------------------------------------------------------------------------------------------------
#  NUMPY Functions
# ----------------------------------------------------------------------------------------------------------------------  


def dot(x,y):
    if rp.use_jax: return jnp.dot(x,y)
    else: return np.dot(x,y)

def sin(x,/):
    if rp.use_jax: return jnp.sin(x)
    else: return np.sin(x)

def cos	(x): 
    if rp.use_jax: return jnp.cos(x)
    else: return np.cos(x)

def tan	(x):
    if rp.use_jax: return jnp.tan(x)
    else: return np.tan(x)
    
def 	arcsin	(): raise NotImplementedError
def 	asin	(): raise NotImplementedError
def 	arccos	(): raise NotImplementedError
def 	acos	(): raise NotImplementedError
def 	arctan	(): raise NotImplementedError
def 	atan	(): raise NotImplementedError
def 	hypot	(): raise NotImplementedError
def 	arctan2	(): raise NotImplementedError
def 	atan2	(): raise NotImplementedError
def 	degrees	(): raise NotImplementedError
def 	radians	(): raise NotImplementedError
def 	unwrap	(): raise NotImplementedError
def 	deg2rad	(): raise NotImplementedError
def 	rad2deg	(): raise NotImplementedError
def 	sinh	(): raise NotImplementedError
def 	cosh	(): raise NotImplementedError
def 	tanh	(): raise NotImplementedError
def 	arcsinh	(): raise NotImplementedError
def 	asinh	(): raise NotImplementedError
def 	arccosh	(): raise NotImplementedError
def 	acosh	(): raise NotImplementedError
def 	arctanh	(): raise NotImplementedError
def 	atanh	(): raise NotImplementedError
def 	round	(): raise NotImplementedError
def 	around	(): raise NotImplementedError
def 	rint	(): raise NotImplementedError
def 	fix	(): raise NotImplementedError
def 	floor	(): raise NotImplementedError
def 	ceil	(): raise NotImplementedError
def 	trunc	(): raise NotImplementedError
def 	prod	(): raise NotImplementedError
def 	sum	(): raise NotImplementedError
def 	nanprod	(): raise NotImplementedError
def 	nansum	(): raise NotImplementedError
def 	cumprod	(): raise NotImplementedError
def 	cumsum	(): raise NotImplementedError
def 	nancumprod	(): raise NotImplementedError
def 	nancumsum	(): raise NotImplementedError
def 	diff	(): raise NotImplementedError
def 	ediff1d	(): raise NotImplementedError
def 	gradient	(): raise NotImplementedError
def 	cross	(): raise NotImplementedError
def 	exp	(): raise NotImplementedError
def 	expm1	(): raise NotImplementedError
def 	exp2	(): raise NotImplementedError
def 	log	(): raise NotImplementedError
def 	log10	(): raise NotImplementedError
def 	log2	(): raise NotImplementedError
def 	log1p	(): raise NotImplementedError
def 	logaddexp	(): raise NotImplementedError
def 	logaddexp2	(): raise NotImplementedError
def 	i0	(): raise NotImplementedError
def 	sinc	(): raise NotImplementedError
def 	signbit	(): raise NotImplementedError
def 	copysign	(): raise NotImplementedError
def 	frexp	(): raise NotImplementedError
def 	ldexp	(): raise NotImplementedError
def 	nextafter	(): raise NotImplementedError
def 	spacing	(): raise NotImplementedError
def 	lcm	(): raise NotImplementedError
def 	gcd	(): raise NotImplementedError
def 	add	(): raise NotImplementedError
def 	reciprocal	(): raise NotImplementedError
def 	positive	(): raise NotImplementedError
def 	negative	(): raise NotImplementedError
def 	multiply	(): raise NotImplementedError
def 	divide	(): raise NotImplementedError
def 	power	(): raise NotImplementedError
def 	pow	(): raise NotImplementedError
def 	subtract	(): raise NotImplementedError
def 	true_divide	(): raise NotImplementedError
def 	floor_divide	(): raise NotImplementedError
def 	float_power	(): raise NotImplementedError
def 	fmod	(): raise NotImplementedError
def 	mod	(): raise NotImplementedError
def 	modf	(): raise NotImplementedError
def 	remainder	(): raise NotImplementedError
def 	divmod	(): raise NotImplementedError
def 	angle	(): raise NotImplementedError
def 	real	(): raise NotImplementedError
def 	imag	(): raise NotImplementedError
def 	conj	(): raise NotImplementedError
def 	conjugate	(): raise NotImplementedError
def 	maximum	(): raise NotImplementedError
def 	max	(): raise NotImplementedError
def 	amax	(): raise NotImplementedError
def 	fmax	(): raise NotImplementedError
def 	nanmax	(): raise NotImplementedError
def 	minimum	(): raise NotImplementedError
def 	min	(): raise NotImplementedError
def 	amin	(): raise NotImplementedError
def 	fmin	(): raise NotImplementedError
def 	nanmin	(): raise NotImplementedError
def 	convolve	(): raise NotImplementedError
def 	clip	(): raise NotImplementedError
def 	sqrt	(x):
    if rp.use_jax: return jnp.sqrt(x)
    else: return np.sqrt(x)
def 	cbrt	(): raise NotImplementedError
def 	square	(): raise NotImplementedError
def 	absolute	(): raise NotImplementedError
def 	fabs	(): raise NotImplementedError
def 	sign	(): raise NotImplementedError
def 	heaviside	(): raise NotImplementedError
def 	nan_to_num	(): raise NotImplementedError
def 	real_if_close	(): raise NotImplementedError
def 	interp	(): raise NotImplementedError
def 	bitwise_count	(): raise NotImplementedError
def 	vdot	(): raise NotImplementedError
def 	vecdot	(): raise NotImplementedError
def 	inner	(): raise NotImplementedError
def 	outer	(): raise NotImplementedError
def 	matmul	(): raise NotImplementedError
def 	tensordot	(): raise NotImplementedError
def 	einsum	(): raise NotImplementedError
def 	einsum_path	(): raise NotImplementedError
def 	kron	(): raise NotImplementedError
def 	trace	(): raise NotImplementedError
def 	diagonal	(): raise NotImplementedError
def 	load	(): raise NotImplementedError
def 	save	(): raise NotImplementedError
def 	savez	(): raise NotImplementedError
def 	savez_compressed	(): raise NotImplementedError
# def 	lib.npyio.NpzFile	(): raise NotImplementedError
def 	loadtxt	(): raise NotImplementedError
def 	savetxt	(): raise NotImplementedError
def 	genfromtxt	(): raise NotImplementedError
def 	fromregex	(): raise NotImplementedError
def 	fromstring	(): raise NotImplementedError
# def 	ndarray.tofile	(): raise NotImplementedError
# def 	ndarray.tolist	(): raise NotImplementedError
def 	array2string	(): raise NotImplementedError
def 	array_repr	(): raise NotImplementedError
def 	array_str	(): raise NotImplementedError
def 	format_float_positional	(): raise NotImplementedError
def 	format_float_scientific	(): raise NotImplementedError
def 	memmap	(): raise NotImplementedError
# def 	lib.format.open_memmap	(): raise NotImplementedError
def 	set_printoptions	(): raise NotImplementedError
def 	get_printoptions	(): raise NotImplementedError
def 	printoptions	(): raise NotImplementedError
def 	binary_repr	(): raise NotImplementedError
def 	base_repr	(): raise NotImplementedError
# def 	lib.npyio.DataSource	(): raise NotImplementedError
# def 	lib.form	(): raise NotImplementedError
def 	apply_along_axis	(): raise NotImplementedError
def 	apply_over_axes	(): raise NotImplementedError
def 	vectorize	(): raise NotImplementedError
def 	frompyfunc	(): raise NotImplementedError
def 	piecewise	(): raise NotImplementedError
def 	empty	(): raise NotImplementedError
def 	empty_like	(): raise NotImplementedError
def 	eye	(): raise NotImplementedError
def 	identity	(): raise NotImplementedError
def 	ones	(): raise NotImplementedError
def 	ones_like	(): raise NotImplementedError
def 	zeros	(): raise NotImplementedError
def 	zeros_like	(): raise NotImplementedError
def 	full	(): raise NotImplementedError
def 	full_like	(): raise NotImplementedError
def 	array	(): raise NotImplementedError
def 	asarray	(): raise NotImplementedError
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