"""Fermi level solvers."""

import typing as ty

import numpy as np
import numpy.typing as npt
import scipy as sp

# from .occupation import compute_n_electrons, compute_n_electrons_derivative, compute_n_electrons_2nd_derivative
from .occupation import compute_n_electrons_2nd_derivative_python as compute_n_electrons_2nd_derivative
from .occupation import compute_n_electrons_derivative_python as compute_n_electrons_derivative
from .occupation import compute_n_electrons_python as compute_n_electrons

__all__ = (
    "find_fermi_energy",
    "find_fermi_energy_zero_temp",
    "find_fermi_energy_bisection",
    "find_fermi_energy_two_stage",
)

from .smearing import Cold, Delta, FermiDirac, Gaussian, smearing_from_name


def find_fermi_energy(  # pylint: disable=too-many-arguments
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: int,
    n_electrons_tol: float = 1e-6,
) -> float:
    """Find the Fermi level by bisection, two-stage algorithm, or at zero temperature depending on the smearing type.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.
        n_electrons (int): target number of electrons.
        n_electrons_tol (float, optional): tolerance for the number of electrons. Defaults to 1e-6.

    Raises:
        ValueError: if the smearing class returned by `smearing_from_name` is unknown.

    Returns:
        float: Fermi energy.
    """
    smearing_cls = smearing_from_name(smearing_type)
    if smearing_cls is Delta:
        fermi_energy = find_fermi_energy_zero_temp(eigenvalues, weights, n_electrons)
    elif smearing_cls in (Gaussian, FermiDirac):
        fermi_energy = find_fermi_energy_bisection(
            eigenvalues, weights, smearing_type, smearing_width, n_electrons, n_electrons_tol
        )
    elif smearing_cls is Cold:
        fermi_energy = find_fermi_energy_two_stage(
            eigenvalues, weights, smearing_type, smearing_width, n_electrons, n_electrons_tol
        )
    else:
        raise ValueError(f"Unknown smearing class: {smearing_cls}")
    return fermi_energy


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
) -> float:
    """Find the Fermi level by bisection.

    Adapated from DFTK.jl/src/occupation.jl.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width.
        n_electrons (int): target number of electrons.
        n_electrons_tol (float, optional): tolerance for the number of electrons. Defaults to 1e-6.

    Returns:
        float: Fermi energy.
    """

    def objective(ef):
        ne = compute_n_electrons(eigenvalues, weights, ef, smearing_type, smearing_width)
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
        return eigenvalues.max()

    fermi_energy = sp.optimize.bisect(objective, e_min, e_max)
    if np.abs(objective(fermi_energy)) > n_electrons_tol:
        raise RuntimeError(f"Failed to find Fermi energy with bisection: {fermi_energy}")

    if fermi_energy >= eigenvalues.max():
        return eigenvalues.max()

    return fermi_energy


class _TwoStageObjective:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        eigenvalues: npt.ArrayLike,
        weights: npt.ArrayLike,
        smearing_type: str,
        smearing_width: float,
        n_electrons: float,
    ):
        self.eigenvalues: np.ndarray = np.ascontiguousarray(eigenvalues)
        self.weights: np.ndarray = np.ascontiguousarray(weights)
        self.smearing_type: str = str(smearing_type)
        self.smearing_width: float = float(smearing_width)
        self.n_electrons: float = float(n_electrons)
        self._ne: float = 0.0
        self._dne: float = 0.0

    def objective(self, ef: float) -> float:
        """Objective function for Newton minimization.

        Args:
            ef (float): Fermi energy

        Returns:
            float: (n_electrons(ef) - n_electrons_target)^2
        """
        self._ne = compute_n_electrons(self.eigenvalues, self.weights, ef, self.smearing_type, self.smearing_width)
        return (self._ne - self.n_electrons) ** 2

    def objective_deriv(self, ef: float) -> float:
        """Derivative of the objective function for Newton minimization.

        Args:
            ef (float): Fermi energy

        Returns:
            float: d((n_electrons(ef) - n_electrons_target)^2)/d(ef)
        """
        self._dne = compute_n_electrons_derivative(
            self.eigenvalues, self.weights, ef, self.smearing_type, self.smearing_width
        )
        return 2 * (self._ne - self.n_electrons) * self._dne

    def objective_2nd_deriv(self, ef: float) -> float:
        """Second derivative of the objective function for Newton minimization.

        Args:
            ef (float): Fermi energy

        Returns:
            float: d^2((n_electrons(ef) - n_electrons_target)^2)/d(ef)^2
        """
        ddne = compute_n_electrons_2nd_derivative(
            self.eigenvalues, self.weights, ef, self.smearing_type, self.smearing_width
        )
        return 2 * ((self._ne - self.n_electrons) * ddne + self._dne**2)


def find_fermi_energy_two_stage(  # pylint: disable=too-many-arguments
    eigenvalues: npt.ArrayLike,
    weights: npt.ArrayLike,
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: int,
    n_electrons_tol: float = 1e-6,
    newton_kwargs: ty.Optional[ty.Dict[str, ty.Any]] = None,
) -> float:
    """Find the Fermi level using a two-stage algorithm which starts from a bisection with Gaussian
    smearing and follows up with a Newton refinement with the requested smearing.

    Adapated from DFTK.jl/src/occupation.jl.

    Args:
        eigenvalues (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.ArrayLike): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width
        n_electrons (int): target number of electrons

    Returns:
        float: Fermi energy.
    """
    # Start with bisection and Gaussian smearing
    bisection_fermi = find_fermi_energy_bisection(eigenvalues, weights, "gauss", smearing_width, n_electrons, np.inf)

    # If the bisection Fermi level is the maximum band energy, return it.
    # As noted in `find_fermi_energy_bisection`, this won't actually give the correct number of electrons
    # at non-zero smearing width, but it's the result that makes sense in this context.
    if bisection_fermi >= eigenvalues.max():
        return eigenvalues.max()

    # Refine with Newton and the requested smearing (probably cold)
    two_stage_fermi = bisection_fermi

    obj = _TwoStageObjective(eigenvalues, weights, smearing_type, smearing_width, n_electrons)
    objective = obj.objective
    objective_deriv = obj.objective_deriv
    objective_2nd_deriv = obj.objective_2nd_deriv

    try:
        if newton_kwargs is None:
            newton_kwargs = {}
        newton_fermi = sp.optimize.newton(
            func=objective, fprime=objective_deriv, fprime2=objective_2nd_deriv, x0=bisection_fermi, **newton_kwargs
        )
        if np.abs(objective(newton_fermi)) <= n_electrons_tol:
            two_stage_fermi = newton_fermi
    except RuntimeError:
        pass

    return two_stage_fermi
