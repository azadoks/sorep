# cython: language_level=3str
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython


cdef cython.floating fermi_occ(cython.floating)
cdef cython.floating gauss_occ(cython.floating)
cdef cython.floating cold_occ(cython.floating)

cdef cython.floating fermi_docc(cython.floating)
cdef cython.floating gauss_docc(cython.floating)
cdef cython.floating cold_docc(cython.floating)

cdef cython.floating fermi_ddocc(cython.floating)
cdef cython.floating gauss_ddocc(cython.floating)
cdef cython.floating cold_ddocc(cython.floating)
