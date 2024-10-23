# src.py
# (c) Copyright 2024 Aerospace Research Community LLC

# Created:  Aug 2024 E. Botero
# Modified: 

import RNUMPY as rp

j   = rp.jax_handle
np  = rp.numpy_handle
jnp = j.numpy

jl = jnp.linalg
nl = np.linalg

def multi_dot(arrays, *, precision=None):
    if not rp.use_jax: return nl.multi_dot(arrays, out=precision)
    else: return jl.multi_dot(arrays, precision=precision)
    
def cross(x1, x2, /, *, axis=-1):
    if not rp.use_jax: return nl.cross(x1, x2, axis=axis)
    else: return jl.cross(x1, x2, axis=axis)
    
def cholesky(a, *, upper=False):
    if not rp.use_jax: return nl.cholesky(a, upper=upper)
    else: return jl.cholesky(a, upper=upper)
    
def outer(x1, x2, /):
    if not rp.use_jax: return nl.outer(x1, x2)
    else: return jl.outer(x1, x2)
    
def qr(a, mode='reduced'):
    if not rp.use_jax: return nl.qr(a, mode=mode)
    else: return jl.qr(a, mode=mode)
    
def svd(a, full_matrices=True, *, compute_uv=True, hermitian=False, subset_by_index=None):
    if not rp.use_jax: return nl.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    else: return jl.svd(a, full_matrices=full_matrices, compute_uv=compute_uv,
                        hermitian=hermitian, subset_by_index=subset_by_index)
    
def svdvals(x, /):
    if not rp.use_jax: return nl.svdvals(x)
    else: return jl.svdvals(x)
    
def eig(a):
    if not rp.use_jax: return nl.eig(a)
    else: return jl.eig(a)
    
def eigh(a, UPLO=None, symmetrize_input=True):
    if not rp.use_jax: return nl.eigh(a, UPLO=UPLO)
    else: return jl.eigh(a, UPLO=UPLO, symmetrize_input=symmetrize_input)
    
def eigvals(a):
    if not rp.use_jax: return nl.eigvals(a)
    else: return jl.eigvals(a)
    
def eigvalsh(a, UPLO='L'):
    if not rp.use_jax: return nl.eigvalsh(a, UPLO=UPLO)
    else: return jl.eigvalsh(a, UPLO=UPLO)
    
def norm(x, ord=None, axis=None, keepdims=False):
    if not rp.use_jax: return nl.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    else: return jl.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    
def matrix_norm(x, /, *, keepdims=False, ord='fro'):
    if not rp.use_jax: return nl.matrix_norm(x, keepdims=keepdims, ord=ord)
    else: return jl.matrix_norm(x, keepdims=keepdims, ord=ord)
    
def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    if not rp.use_jax: return nl.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    else: return jl.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    
def cond(x, p=None):
    if not rp.use_jax: return nl.cond(x, p=p)
    else: return jl.cond(x, p=p)
    
def det(a):
    if not rp.use_jax: return nl.det(a)
    else: return jl.det(a)
    
def matrix_rank(M, rtol=None, *, tol=None):
    if not rp.use_jax: return nl.matrix_rank(M, tol=tol, hermitian=False, rtol=rtol)
    else: return jl.matrix_rank(M, rtol=rtol, tol=tol)
    
def slogdet(a, *, method=None):
    if not rp.use_jax: return nl.slogdet(a)
    else: return jl.slogdet(a, method=method)
    
def matrix_power(a, n):
    if not rp.use_jax: return nl.matrix_power(a, n)
    else: return jl.matrix_power(a, n)
    
def tensordot(x1, x2, /, *, axes=2, precision=None, preferred_element_type=None):
    if not rp.use_jax: return nl.tensordot(x1, x2, axes=axes)
    else: return jl.tensordot(x1, x2, axes=axes, precision=precision, preferred_element_type=preferred_element_type)
    
def matmul(x1, x2, /, *, precision=None, preferred_element_type=None):
    if not rp.use_jax: return nl.matmul(x1, x2)
    else: return jl.matmul(x1, x2, precision=precision, preferred_element_type=preferred_element_type)
    
def trace(x, /, *, offset=0, dtype=None):
    if not rp.use_jax: return nl.trace(x, offset=offset, dtype=dtype)
    else: return jl.trace(x, offset=offset, dtype=dtype)
    
def solve(a, b):
    if not rp.use_jax: return nl.solve(a, b)
    else: return jl.solve(a, b)
    
def tensorsolve(a, b, axes=None):
    if not rp.use_jax: return nl.tensorsolve(a, b, axes=axes)
    else: return jl.tensorsolve(a, b, axes=axes)
    
def lstsq(a, b, rcond=None, *, numpy_resid=False):
    if not rp.use_jax: return nl.lstsq(a, b, rcond=rcond)
    else: return jl.lstsq(a, b, rcond=rcond, numpy_resid=numpy_resid)
    
def inv(a):
    if not rp.use_jax: return nl.inv(a)
    else: return jl.inv(a)
    
def pinv(a, rtol=None, hermitian=False, *, rcond=None):
    if not rp.use_jax: return nl.pinv(a, rcond=rcond, hermitian=hermitian, rtol=rtol)
    else: return jl.pinv(a, rtol=rtol, hermitian=hermitian, rcond=rcond)
    
def tensorinv(a, ind=2):
    if not rp.use_jax: return nl.tensorinv(a, ind=ind)
    else: return jl.tensorinv(a, ind=ind)
    
def diagonal(x, /, *, offset=0):
    if not rp.use_jax: return nl.diagonal(x, offset=offset)
    else: return jl.diagonal(x, offset=offset)
    
def matrix_transpose(x, /):
    if not rp.use_jax: return nl.matrix_transpose(x)
    else: return jl.matrix_transpose(x)
    
def LinAlgError(): raise NotImplementedError
    
def vecdot(x1, x2, /, *, axis=-1, precision=None, preferred_element_type=None):
    if not rp.use_jax: return nl.vecdot(x1, x2, axis=axis)
    else: return jl.vecdot(x1, x2, axis=axis, precision=precision, preferred_element_type=preferred_element_type)
    