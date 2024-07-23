# cython: language_level=3str
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython
cimport numpy as cnp
from libc.math cimport erf, erfc, exp

DEF SQRT2 = 1.4142135623730951
DEF INVSQRT2 = 0.7071067811865475
DEF SQRT2_INVSQRTPI = 0.7978845608028654
DEF INVSQRTPI = 0.5641895835477563
DEF INVSQRT2PI = 0.3989422804014327

# Occupation function
@cython.ufunc
cdef cython.floating fermi_occ(cython.floating x):
    return 1.0 / (1.0 + exp(x))

@cython.ufunc
cdef cython.floating gauss_occ(cython.floating x):
    return 0.5 * erfc(x)

@cython.ufunc
cdef cython.floating cold_occ(cython.floating x):
    cdef cython.floating arg = x + INVSQRT2
    return -0.5 * erf(arg) + INVSQRT2PI * exp(-arg * arg) + 0.5

# Occupation derivative
@cython.ufunc
cdef cython.floating fermi_docc(cython.floating x):
    return 1.0 / (2.0 + exp(x) + exp(-x))

@cython.ufunc
cdef cython.floating gauss_docc(cython.floating x):
    return INVSQRTPI * exp(-x * x)

@cython.ufunc
cdef cython.floating cold_docc(cython.floating x):
    cdef cython.floating arg = x + INVSQRT2
    return exp(-arg * arg) * (SQRT2_INVSQRTPI * arg + INVSQRTPI)

# Occupation second derivative
@cython.ufunc
cdef cython.floating fermi_ddocc(cython.floating x):
    cdef cython.floating expx = exp(x)
    cdef cython.floating expmx = exp(-x)
    cdef cython.floating arg = (2.0 + expmx + expx)
    return -(expx - expmx) / (arg * arg)

@cython.ufunc
cdef cython.floating gauss_ddocc(cython.floating x):
    return -2.0 * INVSQRTPI * x * exp(-x * x)

@cython.ufunc
cdef cython.floating cold_ddocc(cython.floating x):
    cdef cnp.float64_t arg = x + INVSQRT2
    cdef cnp.float64_t arg2 = arg * arg
    return -exp(-arg2) * (2.0 * SQRT2 * arg2 + 2.0 * arg - SQRT2) * INVSQRTPI
