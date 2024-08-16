"""Fermi level solvers."""

import logging
import typing as ty

import numpy as np
import numpy.typing as npt
import scipy as sp

from .occupation import compute_n_electrons, compute_n_electrons_2nd_derivative, compute_n_electrons_derivative
from .smearing import Cold, Delta, FermiDirac, Gaussian, smearing_from_name

LOGGER = logging.getLogger(__name__)

__all__ = (
    "find_fermi_energy",
    "find_fermi_energy_zero_temp",
    "find_fermi_energy_bisection",
    "find_fermi_energy_advanced",
)


def find_fermi_energy(  # pylint: disable=too-many-arguments
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: int,
    n_electrons_tol: float = 1e-6,
    n_electrons_kwargs: ty.Optional[dict] = None,
    dn_electrons_kwargs: ty.Optional[dict] = None,
    ddn_electrons_kwargs: ty.Optional[dict] = None,
    newton_kwargs: ty.Optional[ty.Dict[str, ty.Any]] = None,
) -> ty.Dict[str, ty.Any]:
    """Find the Fermi level by bisection, two-stage algorithm, or at zero temperature depending on the smearing type.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.
        n_electrons (int): target number of electrons.
        n_electrons_tol (float, optional): tolerance for the number of electrons. Defaults to 1e-6.
        n_electrons_kwargs (ty.Optional[dict]): Keyword arguments to pass to `compute_n_electrons`.
        dn_electrons_kwargs (ty.Optional[dict]): Keyword arguments to pass to `compute_n_electrons_derivative`.
        ddn_electrons_kwargs (ty.Optional[dict]): Keyword arguments to pass to `compute_n_electrons_2nd_derivative`.
        newton_kwargs (ty.Optional[ty.Dict]): keyword arguments for `scipy.optimize.root_scalar`.

    Raises:
        ValueError: if the smearing class returned by `smearing_from_name` is unknown.

    Returns:
        float: Fermi energy.
    """
    n_electrons_kwargs = n_electrons_kwargs or {}
    smearing_cls = smearing_from_name(smearing_type)
    if smearing_cls is Delta:
        fermi_result = find_fermi_energy_zero_temp(eigenvalues, weights, n_electrons)
    elif smearing_cls in (Gaussian, FermiDirac):
        fermi_result = find_fermi_energy_bisection(
            eigenvalues,
            weights,
            smearing_type,
            smearing_width,
            n_electrons,
            n_electrons_tol,
            n_electrons_kwargs=n_electrons_kwargs,
        )
    elif smearing_cls is Cold:
        fermi_result = find_fermi_energy_advanced(
            eigenvalues,
            weights,
            smearing_type,
            smearing_width,
            n_electrons,
            n_electrons_tol,
            n_electrons_kwargs=n_electrons_kwargs,
            dn_electrons_kwargs=dn_electrons_kwargs,
            ddn_electrons_kwargs=ddn_electrons_kwargs,
            newton_kwargs=newton_kwargs,
        )
    else:
        raise ValueError(f"Unknown smearing class: {smearing_cls}")
    return fermi_result


def find_fermi_energy_zero_temp(
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    n_electrons: int,
) -> float:
    """Find the Fermi level at zero electronic temperature (no smearing) by fully occupying bands.
    If the number of electrons is obtained by fully occupying every band, the Fermi level is the maximum band energy.
    If unfilled conduction bands exist, the Fermi level is the midpoint between the valence band maximum and conduction
    band minimum.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        n_electrons (int): target number of electrons.

    Raises:
        ValueError: if the number of electrons cannot be obtained without partial occupations.

    Returns:
        float: Fermi energy.
    """
    total_weight = weights.sum()

    if n_electrons % (eigenvalues.shape[0] * eigenvalues.shape[2] * total_weight) != 0:
        raise ValueError(f"{n_electrons} electrons cannot be obtained with no partial occupations.")

    return NotImplementedError("Zero temperature Fermi level not implemented.")


def find_fermi_energy_bisection(  # pylint: disable=too-many-arguments
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: int,
    n_electrons_tol: float = 1e-6,
    n_electrons_kwargs: ty.Optional[dict] = None,
) -> ty.Dict[str, ty.Any]:
    """Find the Fermi level by bisection.

    Adapated from DFTK.jl/src/occupation.jl.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.
        n_electrons (int): target number of electrons.
        n_electrons_tol (float, optional): tolerance for the number of electrons. Defaults to 1e-6.
        n_electrons_kwargs (ty.Optional[dict]): Keyword arguments to pass to `compute_n_electrons`.

    Returns:
        float: Fermi energy.
    """
    n_electrons_kwargs = n_electrons_kwargs or {}

    def objective(ef):
        ne = compute_n_electrons(eigenvalues, weights, ef, smearing_type, smearing_width, **n_electrons_kwargs)
        return ne - n_electrons

    # Get rough bounds for the Fermi level
    e_min = eigenvalues.min() - 10 * smearing_width
    e_max = eigenvalues.max() + 10 * smearing_width

    obj_upper = objective(e_max)
    if (obj_upper < 0) and np.isclose(obj_upper, 0.0, atol=1e-4):
        # Fully occupying all bands gives the correct number of electrons.
        # I use a fixed tolerance here because the two-stage algorithm provides n_electrons_tol=np.inf,
        # and the bisection will fail if we let this go.
        # Note that this is not the same as the Fermi level being the maximum band energy because of smearing
        # However, we'll return the maximum band energy so that the result doesn't depend on the smearing width
        # or our choice of guess bounds (bands.max() + 10 * smearing with)
        return {"fermi_energy": eigenvalues.max(), "flag": "max_eigenvalue"}

    fermi_energy = sp.optimize.bisect(objective, e_min, e_max)
    if np.abs(objective(fermi_energy)) > n_electrons_tol:
        raise RuntimeError(f"Failed to find Fermi energy with bisection: {fermi_energy}")

    if fermi_energy >= eigenvalues.max():
        return {"fermi_energy": eigenvalues.max(), "flag": "max_eigenvalue"}

    return {"fermi_energy": fermi_energy, "flag": "bisection"}


class _RootScalarObjective:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        eigenvalues: npt.ArrayLike,
        weights: npt.ArrayLike,
        smearing_type: str,
        smearing_width: float,
        n_electrons: float,
        n_electrons_kwargs: ty.Optional[dict] = None,
        dn_electrons_kwargs: ty.Optional[dict] = None,
        ddn_electrons_kwargs: ty.Optional[dict] = None,
        fprime: bool = True,
        fprime2: bool = True,
    ):
        if fprime2 and not fprime:
            raise ValueError("fprime2 requires fprime to be True.")
        self.eigenvalues: np.ndarray = np.ascontiguousarray(eigenvalues)
        self.weights: np.ndarray = np.ascontiguousarray(weights)
        self.smearing_type: str = str(smearing_type)
        self.smearing_width: float = float(smearing_width)
        self.n_electrons: float = float(n_electrons)
        self.n_electrons_kwargs: dict = n_electrons_kwargs or {}
        self.dn_electrons_kwargs: dict = dn_electrons_kwargs or {}
        self.ddn_electrons_kwargs: dict = ddn_electrons_kwargs or {}
        self.fprime: bool = fprime
        self.fprime2: bool = fprime2

    def __call__(self, ef: float) -> ty.Union[float, ty.Tuple[float, float], ty.Tuple[float, float, float]]:
        ne = self._compute_n_electrons(ef)
        f = (ne - self.n_electrons) ** 2
        if not self.fprime and not self.fprime2:
            return f
        dne = self._compute_n_electrons_derivative(ef)
        fprime = 2 * (ne - self.n_electrons) * dne
        if not self.fprime2:
            return f, fprime
        ddne = self._compute_n_electrons_2nd_derivative(ef)
        fprime2 = 2 * ((ne - self.n_electrons) * ddne + dne**2)
        return f, fprime, fprime2

    def _compute_n_electrons(self, ef: float) -> float:
        return compute_n_electrons(
            self.eigenvalues, self.weights, ef, self.smearing_type, self.smearing_width, **self.n_electrons_kwargs
        )

    def _compute_n_electrons_derivative(self, ef: float) -> float:
        return compute_n_electrons_derivative(
            self.eigenvalues, self.weights, ef, self.smearing_type, self.smearing_width, **self.dn_electrons_kwargs
        )

    def _compute_n_electrons_2nd_derivative(self, ef: float) -> float:
        return compute_n_electrons_2nd_derivative(
            self.eigenvalues, self.weights, ef, self.smearing_type, self.smearing_width, **self.ddn_electrons_kwargs
        )


def find_fermi_energy_advanced(  # pylint: disable=too-many-arguments
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: float,
    n_electrons_tol: float = 1e-6,
    newton_kwargs: ty.Optional[ty.Dict[str, ty.Any]] = None,
    n_electrons_kwargs: ty.Optional[dict] = None,
    dn_electrons_kwargs: ty.Optional[dict] = None,
    ddn_electrons_kwargs: ty.Optional[dict] = None,
) -> ty.Dict[str, ty.Any]:
    """Find the Fermi level using a multi-stage algorithm as described in
    F.J. dos Santos and N. Marzari, PRB 107, 195122 (2023).

        # Gaussian bisection
        eF = bisection(enk, wk, delta[gauss], sigma, Ne)
        if |Ne - Ne[cold](eF)| < tol:
            return eF
        # Cold Newton refinement
        eF = newton(enk, wk, delta[cold], sigma, Ne)
        if |Ne - Ne[cold](eF)| < tol:
            return eF
        # Cold bisection fallback
        eF = bisection(enk, wk, sigma, delta[cold], Ne)
        if |Ne - Ne[cold](eF)| < tol:
            warn
        return eF

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width
        n_electrons (float): target number of electrons
        n_electrons_tol (float): tolerance on the number of electrons
        newton_kwargs (ty.Optional[ty.Dict]): keyword arguments for `scipy.optimize.root_scalar`
        n_electrons_kwargs (ty.Optional[dict]): Keyword arguments to pass to `compute_n_electrons`.
        dn_electrons_kwargs (ty.Optional[dict]): Keyword arguments to pass to `compute_n_electrons_derivative`.
        ddn_electrons_kwargs (ty.Optional[dict]): Keyword arguments to pass to `compute_n_electrons_2nd_derivative`.

    Returns:
        float: Fermi energy.
    """
    objective = _RootScalarObjective(
        eigenvalues,
        weights,
        smearing_type,
        smearing_width,
        n_electrons,
        n_electrons_kwargs,
        dn_electrons_kwargs,
        ddn_electrons_kwargs,
    )

    # Start with bisection and Gaussian smearing
    gauss_bisection_fermi = find_fermi_energy_bisection(
        eigenvalues, weights, "gauss", smearing_width, n_electrons, np.inf, n_electrons_kwargs
    )["fermi_energy"]

    if np.sqrt(objective(gauss_bisection_fermi)[0]) <= n_electrons_tol:
        return {"fermi_energy": gauss_bisection_fermi, "flag": "gauss_bisection"}

    # If the bisection Fermi level is the maximum band energy, return it.
    # As noted in `find_fermi_energy_bisection`, this won't actually give the correct number of electrons
    # at non-zero smearing width, but it's the result that makes sense in this context.
    if gauss_bisection_fermi >= eigenvalues.max():
        return {"fermi_energy": eigenvalues.max(), "flag": "max_eigenvalue"}

    newton_kwargs = newton_kwargs or {}
    root_result = sp.optimize.root_scalar(
        f=objective, fprime=True, fprime2=True, x0=gauss_bisection_fermi, method="halley", **newton_kwargs
    )
    newton_fermi = root_result.root

    if np.sqrt(objective(newton_fermi)[0]) <= n_electrons_tol:
        return {"fermi_energy": newton_fermi, "flag": "newton"}

    # Fall back to bisection with cold smearing
    cold_bisection_fermi = find_fermi_energy_bisection(
        eigenvalues, weights, smearing_type, smearing_width, n_electrons, n_electrons_kwargs=n_electrons_kwargs
    )["fermi_energy"]

    if np.sqrt(objective(cold_bisection_fermi)[0]) > n_electrons_tol:
        LOGGER.warning("Fallback to cold bisection failed to converge.")

    return {"fermi_energy": cold_bisection_fermi, "flag": "cold_bisection"}
