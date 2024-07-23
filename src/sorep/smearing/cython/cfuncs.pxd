# cython: language_level=3str
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as cnp

ctypedef cnp.float64_t (*occ_p) (cnp.float64_t)

cdef cnp.float64_t fermi_occ_c(cnp.float64_t)
cdef cnp.float64_t gauss_occ_c(cnp.float64_t)
cdef cnp.float64_t cold_occ_c(cnp.float64_t)

cdef cnp.float64_t fermi_docc_c(cnp.float64_t)
cdef cnp.float64_t gauss_docc_c(cnp.float64_t)
cdef cnp.float64_t cold_docc_c(cnp.float64_t)

cdef cnp.float64_t fermi_ddocc_c(cnp.float64_t)
cdef cnp.float64_t gauss_ddocc_c(cnp.float64_t)
cdef cnp.float64_t cold_ddocc_c(cnp.float64_t)
