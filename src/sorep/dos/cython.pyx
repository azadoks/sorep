# cython: language_level=3str
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import typing as ty

import numpy as np
import numpy.typing as npt

cimport cython
cimport numpy as cnp

from ..smearing.cython cimport cold_docc_c, fermi_docc_c, gauss_docc_c, occ_p


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _smeared_dos(
    cnp.float64_t[:, ::1] dos_view,
    cnp.float64_t[::1] energies,
    cnp.float64_t[:, :, ::1] eigenvalues,
    cnp.float64_t[::1] weights,
    occ_p docc,
    cnp.float64_t smearing_width,
    cnp.float64_t zlim
):
    cdef Py_ssize_t n_spins = eigenvalues.shape[0]
    cdef Py_ssize_t n_kpoints = eigenvalues.shape[1]
    cdef Py_ssize_t n_bands = eigenvalues.shape[2]
    cdef Py_ssize_t n_energies = energies.shape[0]
    cdef Py_ssize_t i_spin, i_kpoint, i_band, i_energy
    cdef double inv_width = 1 / smearing_width
    cdef double z  # Z-value temporary variable (E - Enk) / σ

    for i_spin in range(n_spins):
        for i_energy in range(n_energies):
            for i_kpoint in range(n_kpoints):
                for i_band in range(n_bands):
                    # z = (x - μ) / σ
                    z = (eigenvalues[i_spin, i_kpoint, i_band] - energies[i_energy]) * inv_width
                    if abs(z) < zlim:
                        dos_view[i_spin, i_energy] += weights[i_kpoint] * docc(z)
            dos_view[i_spin, i_energy] *= inv_width

    return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def smeared_dos(
    energies: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    smearing_type: ty.Union[str,int],
    smearing_width: float,
    max_exponent: float = 36.0
) -> cnp.float64_t[:,::1]:
    """Compute a smeared density of states.

    Args:
        energies (npt.ArrayLike): energies at which to sample the DOS
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) array of eigenvalues
        weights (npt.ArrayLike): (n_kpoints,) array of k-point weights
        smearing_type (ty.Union[str,int]): type of smearing (see `smearing_from_name`)
        smearing_width (float): smearing width
        max_exponent (float, optional): maximum argument to `exp` in the smearing function. Defaults to 36.0 (Z=6.0).

    Returns:
        npt.NDArray: (n_spins, n_energies) array containing the DOS for each spin channel
    """
    energies = np.ascontiguousarray(energies, dtype=np.float64)
    eigenvalues = np.ascontiguousarray(eigenvalues, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)

    if weights.shape[0] != eigenvalues.shape[1]:
        raise ValueError("The number of k-points in weights and eigenvalues must be the same")

    smearing_type = str(smearing_type).lower()
    cdef occ_p docc  # Function pointer to occupation derivative
    if smearing_type in ("fd", "f-d", "fermi", "fermi-dirac", "-99"):
        docc = fermi_docc_c
        zlim = max_exponent
    elif smearing_type in ("gauss", "gaussian", "0"):
        docc = gauss_docc_c
        zlim = np.sqrt(max_exponent)
    elif smearing_type in ("mv", "m-v", "marzari-vanderbilt", "cold", "-1"):
        docc = cold_docc_c
        zlim = np.sqrt(max_exponent)
    else:
        raise ValueError(f"Unknown smearing type '{smearing_type}'")

    dos = np.zeros((eigenvalues.shape[0], energies.shape[0]), dtype=np.float64)
    cdef cnp.float64_t[:, ::1] dos_view = dos  # C view of the dos numpy array
    _smeared_dos(dos_view, energies, eigenvalues, weights, docc, smearing_width, zlim)

    return dos
