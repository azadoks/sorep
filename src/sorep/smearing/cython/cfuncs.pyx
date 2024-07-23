# cython: language_level=3str
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as cnp
from libc.math cimport erf, erfc, exp

DEF SQRT2 = 1.4142135623730951
DEF INVSQRT2 = 0.7071067811865475
DEF SQRT2_INVSQRTPI = 0.7978845608028654
DEF INVSQRTPI = 0.5641895835477563
DEF INVSQRT2PI = 0.3989422804014327

# Occupation function
cdef inline cnp.float64_t fermi_occ_c(cnp.float64_t x):
    return 1.0 / (1.0 + exp(x))

cdef inline cnp.float64_t gauss_occ_c(cnp.float64_t x):
    return 0.5 * erfc(x)

cdef inline cnp.float64_t cold_occ_c(cnp.float64_t x):
    cdef cnp.float64_t arg = x + INVSQRT2
    return -0.5 * erf(arg) + INVSQRT2PI * exp(-arg * arg) + 0.5

# Occupation derivative
cdef inline cnp.float64_t fermi_docc_c(cnp.float64_t x):
    return 1.0 / (2.0 + exp(x) + exp(-x))

cdef inline cnp.float64_t gauss_docc_c(cnp.float64_t x):
    return INVSQRTPI * exp(-x * x)

cdef inline cnp.float64_t cold_docc_c(cnp.float64_t x):
    cdef cnp.float64_t arg = x + INVSQRT2
    return exp(-arg * arg) * (SQRT2_INVSQRTPI * arg + INVSQRTPI)

# Occupation second derivative
cdef inline cnp.float64_t fermi_ddocc_c(cnp.float64_t x):
    cdef cnp.float64_t expx = exp(x)
    cdef cnp.float64_t expmx = exp(-x)
    cdef cnp.float64_t arg = (2.0 + expmx + expx)
    return -(expx - expmx) / (arg * arg)

cdef inline cnp.float64_t gauss_ddocc_c(cnp.float64_t x):
    return -2.0 * INVSQRTPI * x * exp(-x * x)

cdef inline cnp.float64_t cold_ddocc_c(cnp.float64_t x):
    cdef cnp.float64_t arg = x + INVSQRT2
    cdef cnp.float64_t arg2 = arg * arg
    return -exp(-arg2) * (2.0 * SQRT2 * arg2 + 2.0 * arg - SQRT2) * INVSQRTPI
