# cython: language_level=3str
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import typing as ty

import numpy as np
import numpy.typing as npt

cimport cython
cimport numpy as cnp

from ..smearing.cython cimport (
    cold_ddocc_c,
    cold_docc_c,
    cold_occ_c,
    fermi_ddocc_c,
    fermi_docc_c,
    fermi_occ_c,
    gauss_ddocc_c,
    gauss_docc_c,
    gauss_occ_c,
    occ_p,
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cnp.float64_t _compute_n_electrons_impl(
    cnp.float64_t[:, :, ::1] eigenvalues,
    cnp.float64_t[::1] weights,
    cnp.float64_t fermi_energy,
    occ_p function,
    cnp.float64_t smearing_width,
    cnp.float64_t zlim,
    cnp.int64_t derivative_order
):
    cdef Py_ssize_t n_spins = eigenvalues.shape[0]
    cdef Py_ssize_t n_kpoints = eigenvalues.shape[1]
    cdef Py_ssize_t n_bands = eigenvalues.shape[2]
    cdef Py_ssize_t i_spin, i_kpoint, i_band
    cdef cnp.float64_t inv_width = 1 / smearing_width
    cdef cnp.float64_t z  # Z-value temporary variable (Enk - E) / σ
    cdef cnp.float64_t n = 0.0
    cdef cnp.float64_t n_k = 0.0

    for i_spin in range(n_spins):
        for i_kpoint in range(n_kpoints):
            for i_band in range(n_bands):
                z = (eigenvalues[i_spin, i_kpoint, i_band] - fermi_energy) * inv_width
                if abs(z) < zlim:
                    n_k += function(z)
            n += n_k * weights[i_kpoint]
            n_k = 0.0

    if n_spins == 1:
        n *= 2.0

    if derivative_order == 0:
        return n
    elif derivative_order == 1:
        return n * inv_width
    else:  # derivative_order == 2
        return n * inv_width * inv_width


def _compute_n_electrons_wrap(
    bands: npt.ArrayLike,
    weights: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str,int],
    smearing_width: float,
    max_exponent: float,
    derivative_order: int
) -> float:
    bands = np.ascontiguousarray(bands, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)

    if weights.shape[0] != bands.shape[1]:
        raise ValueError("The number of k-points in weights and bands must be the same")
    if derivative_order < 0 or derivative_order > 2:
        raise ValueError("Derivative order must be 0, 1, or 2")

    smearing_type = str(smearing_type).lower()
    cdef occ_p occ  # Function pointer to occupation function
    if smearing_type in ("fd", "f-d", "fermi", "fermi-dirac", "-99"):
        if derivative_order == 0:
            occ = fermi_occ_c
            zlim = max_exponent
        elif derivative_order == 1:
            occ = fermi_docc_c
            zlim = max_exponent
        else:  # derivative_order == 2
            occ = fermi_ddocc_c
            zlim = max_exponent
    elif smearing_type in ("gauss", "gaussian", "0"):
        if derivative_order == 0:
            occ = gauss_occ_c
            zlim = max_exponent
        elif derivative_order == 1:
            occ = gauss_docc_c
            zlim = np.sqrt(max_exponent)
        else:  # derivative_order == 2
            occ = gauss_ddocc_c
            zlim = np.sqrt(max_exponent)
    elif smearing_type in ("mv", "m-v", "marzari-vanderbilt", "cold", "-1"):
        if derivative_order == 0:
            occ = cold_occ_c
            zlim = max_exponent
        elif derivative_order == 1:
            occ = cold_docc_c
            zlim = np.sqrt(max_exponent)
        else:  # derivative_order == 2
            occ = cold_ddocc_c
            zlim = np.sqrt(max_exponent)
    else:
        raise ValueError(f"Unknown smearing type '{smearing_type}'")

    return _compute_n_electrons_impl(bands, weights, fermi_energy, occ, smearing_width, zlim, derivative_order)

# The occupation function has the widest spread and requires a higher max_exponent
def compute_n_electrons(
    bands: npt.ArrayLike,
    weights: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str,int],
    smearing_width: float,
    max_exponent: float = 512.0
) -> float:
    return _compute_n_electrons_wrap(bands, weights, fermi_energy, smearing_type, smearing_width, max_exponent, 0)


def compute_n_electrons_derivative(
    bands: npt.ArrayLike,
    weights: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str,int],
    smearing_width: float,
    max_exponent: float = 36.0
) -> float:
    return _compute_n_electrons_wrap(bands, weights, fermi_energy, smearing_type, smearing_width, max_exponent, 1)


def compute_n_electrons_2nd_derivative(
    bands: npt.ArrayLike,
    weights: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str,int],
    smearing_width: float,
    max_exponent: float = 36.0
) -> float:
    return _compute_n_electrons_wrap(bands, weights, fermi_energy, smearing_type, smearing_width, max_exponent, 2)
