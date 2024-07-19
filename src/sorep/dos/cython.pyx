# cython: language_level=3str
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import typing as ty

import numpy as np

cimport cython
cimport numpy as cnp
from libc.math cimport exp, sqrt

DEF INVSQRT2 = 0.7071067811865475
DEF SQRT2_INVSQRTPI = 0.7978845608028654
DEF INVSQRTPI = 0.5641895835477563

cdef double fermi_docc(double x):
    return 1.0 / (2.0 + exp(x) + exp(-x))

cdef double gauss_docc(double x):
    return INVSQRTPI * exp(-x * x)

cdef double cold_docc(double x):
    cdef double arg = x + INVSQRT2
    return exp(-arg * arg) * (SQRT2_INVSQRTPI * arg + INVSQRTPI)

ctypedef double (*docc_p)(double)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def smeared_dos(
    energies: cnp.float64_t[::1],
    bands: cnp.float64_t[:, :, ::1],
    weights: cnp.float64_t[::1],
    smearing_type: ty.Union[str,int],
    smearing_width: float,
    max_exponent: float = 36.0
) -> cnp.float64_t[:,::1]:
    """Compute a smeared density of states.

    Args:
        energies (npt.ArrayLike): energies at which to sample the DOS
        bands (npt.ArrayLike): (n_spins, n_kpoints, n_bands) array of eigenvalues
        weights (npt.ArrayLike): (n_kpoints,) array of k-point weights
        smearing_type (ty.Union[str,int]): type of smearing (see `smearing_from_name`)
        smearing_width (float): smearing width
        max_exponent (float, optional): maximum argument to `exp` in the smearing function. Defaults to 36.0 (Z=6.0).

    Returns:
        npt.NDArray: (n_spins, n_energies) array containing the DOS for each spin channel
    """
    if weights.shape[0] != bands.shape[1]:
        raise ValueError("The number of k-points in weights and bands must be the same")

    smearing_type = str(smearing_type).lower()
    cdef docc_p docc  # Function pointer to occupation derivative
    cdef double zlim  # Limit on the Z-value for the smearing function
    if smearing_type in ("fd", "f-d", "fermi", "fermi-dirac", "-99"):
        docc = fermi_docc
        zlim = max_exponent
    elif smearing_type in ("gauss", "gaussian", "0"):
        docc = gauss_docc
        zlim = sqrt(max_exponent)
    elif smearing_type in ("mv", "m-v", "marzari-vanderbilt", "cold", "-1"):
        docc = cold_docc
        zlim = sqrt(max_exponent)
    else:
        raise ValueError(f"Unknown smearing type '{smearing_type}'")

    cdef Py_ssize_t n_spins = bands.shape[0]
    cdef Py_ssize_t n_kpoints = bands.shape[1]
    cdef Py_ssize_t n_bands = bands.shape[2]
    cdef Py_ssize_t n_energies = energies.shape[0]
    dos = np.zeros((n_spins, n_energies), dtype=np.float64)
    cdef cnp.float64_t[:, ::1] dos_view = dos  # C view of the dos numpy array
    cdef double inv_width = 1 / smearing_width
    cdef double z  # Z-value temporary variable (E - Enk) / σ
    cdef Py_ssize_t i_spin, i_energy, i_kpoint, i_band

    for i_spin in range(n_spins):
        for i_energy in range(n_energies):
            for i_kpoint in range(n_kpoints):
                for i_band in range(n_bands):
                    # z = (x - μ) / σ
                    z = (bands[i_spin, i_kpoint, i_band] - energies[i_energy]) * inv_width
                    if abs(z) < zlim:
                        dos_view[i_spin, i_energy] += weights[i_kpoint] * docc(z)
            dos_view[i_spin, i_energy] *= inv_width

    return dos
