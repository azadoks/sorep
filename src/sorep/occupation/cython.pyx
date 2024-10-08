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
cdef cnp.float64_t _compute_n_electrons_delta(
    cnp.float64_t[:, :, ::1] eigenvalues,
    cnp.float64_t[::1] weights,
    cnp.float64_t fermi_energy
):
    cdef Py_ssize_t n_spins = eigenvalues.shape[0]
    cdef Py_ssize_t n_kpoints = eigenvalues.shape[1]
    cdef Py_ssize_t n_bands = eigenvalues.shape[2]
    cdef Py_ssize_t i_spin, i_kpoint, i_band
    cdef cnp.float64_t n = 0.0
    cdef cnp.float64_t n_k = 0.0

    for i_spin in range(n_spins):
        for i_kpoint in range(n_kpoints):
            for i_band in range(n_bands):
                if eigenvalues[i_spin, i_kpoint, i_band] < fermi_energy:
                    n_k += 1.0
            n += n_k * weights[i_kpoint]
            n_k = 0.0

    if n_spins == 1:
        n *= 2.0

    return n

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
                elif z < -zlim:
                    n_k += 1.0
            n += n_k * weights[i_kpoint]
            n_k = 0.0

    if n_spins == 1:
        n *= 2.0

    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cnp.float64_t _compute_n_electrons_derivative_impl(
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

    if derivative_order == 1:
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

    smearing_type = str(smearing_type).lower()
    if smearing_type in ("fd", "f-d", "fermi", "fermi-dirac", "-99"):
        smearing_type = "fermi"
    elif smearing_type in ("gauss", "gaussian", "0"):
        smearing_type = "gauss"
    elif smearing_type in ("mv", "m-v", "marzari-vanderbilt", "cold", "-1"):
        smearing_type = "cold"
    elif smearing_type in ("None", "delta", "fixed"):
        smearing_type = "delta"
    else:
        raise ValueError(f"Unknown smearing type '{smearing_type}'")

    if derivative_order < 0 or derivative_order > 2:
        raise ValueError("Derivative order must be 0, 1, or 2")

    cdef occ_p occ  # Function pointer to occupation function
    if smearing_type == "fermi":
        if derivative_order == 0:
            occ = fermi_occ_c
        elif derivative_order == 1:
            occ = fermi_docc_c
        elif derivative_order == 2:
            occ = fermi_ddocc_c
    elif smearing_type == "gauss":
        if derivative_order == 0:
            occ = gauss_occ_c
        elif derivative_order == 1:
            occ = gauss_docc_c
        elif derivative_order == 2:
            occ = gauss_ddocc_c
    elif smearing_type == "cold":
        if derivative_order == 0:
            occ = cold_occ_c
        elif derivative_order == 1:
            occ = cold_docc_c
        elif derivative_order == 2:
            occ = cold_ddocc_c
    elif smearing_type == "delta":
        assert smearing_width == 0.0
        if derivative_order == 0:
            return _compute_n_electrons_delta(bands, weights, fermi_energy)
        else:
            return 0.0

    if derivative_order == 0:
        zlim = max_exponent
        return _compute_n_electrons_impl(bands, weights, fermi_energy, occ, smearing_width, zlim)
    else:
        zlim = np.sqrt(max_exponent)
        return _compute_n_electrons_derivative_impl(
            bands, weights, fermi_energy, occ, smearing_width, zlim, derivative_order
        )

# The occupation function has the widest spread and requires a higher max_exponent
def compute_n_electrons(
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str,int],
    smearing_width: float,
    max_exponent: float = np.inf
) -> float:
    """Compute the number of electrons (total occupation).

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        fermi_energy (float): Fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.
        max_exponent (float, optional): maximum argument to `exp` in the smearing function. Defaults to inf.

    Returns:
        float: (fractional) number of electrons.
    """
    return _compute_n_electrons_wrap(eigenvalues, weights, fermi_energy, smearing_type, smearing_width, max_exponent, 0)


def compute_n_electrons_derivative(
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str,int],
    smearing_width: float,
    max_exponent: float = np.inf
) -> float:
    """Compute the derivative of the number of electrons (total occupation) w.r.t. the Fermi energy.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        fermi_energy (float): Fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.
        max_exponent (float, optional): maximum argument to `exp` in the smearing function. Defaults to inf.

    Returns:
        float: d[Nelec]/d[E_fermi].
    """
    return _compute_n_electrons_wrap(eigenvalues, weights, fermi_energy, smearing_type, smearing_width, max_exponent, 1)


def compute_n_electrons_2nd_derivative(
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    fermi_energy: float,
    smearing_type: ty.Union[str,int],
    smearing_width: float,
    max_exponent: float = np.inf
) -> float:
    """Compute the derivative of the number of electrons (total occupation) w.r.t. the Fermi energy.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        fermi_energy (float): Fermi energy.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.
        max_exponent (float, optional): maximum argument to `exp` in the smearing function. Defaults to inf.

    Returns:
        float: d[Nelec]/d[E_fermi].
    """
    return _compute_n_electrons_wrap(eigenvalues, weights, fermi_energy, smearing_type, smearing_width, max_exponent, 2)
