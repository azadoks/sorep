"""Electronic smearing functions."""

from abc import ABC, abstractmethod
import typing as ty

import numpy as np
import numpy.typing as npt
import scipy as sp

__all__ = (
    "Smearing",
    "Delta",
    "Gaussian",
    "FermiDirac",
    "Cold",
    "smearing_from_name",
)

MAX_EXP_ARG = 200.0  # Maximum argument for exponential functions, taken from QE


class Smearing(ABC):
    """Abstract base class for smearing functions."""

    def __init__(self, center: float, width: float):
        self.center = center
        self.width = width

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(center={self.center}, width={self.width})"

    def __str__(self) -> str:
        return self.__repr__()

    def _scale(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        return (self.center - bands) / self.width

    @abstractmethod
    def occupation(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        """Compute the occupation function (cumulative distribution function)."""

    @abstractmethod
    def occupation_derivative(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        """Compute the derivative of the occupation function (probability density function)."""

    @abstractmethod
    def occupation_2nd_derivative(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        """Compute the second derivative (curvature) of the occupation function."""


class Delta(Smearing):
    """No smearing, i.e. a true step/delta function."""

    def occupation(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        return np.where(bands < self.center, 1.0, 0.0)

    def occupation_derivative(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        return np.where(bands == self.center, -np.inf, 0.0)

    #! This may not be correct, should be checked
    def occupation_2nd_derivative(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        return np.where(bands == self.center, -np.inf, 0.0)


class Gaussian(Smearing):
    """Gaussian smearing."""

    def occupation(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        return 0.5 * sp.special.erfc(-self._scale(bands))

    def occupation_derivative(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        x = self._scale(bands) ** 2
        return np.where(
            x > MAX_EXP_ARG,
            1.0 / np.sqrt(np.pi) * np.exp(-MAX_EXP_ARG),
            1.0 / np.sqrt(np.pi) * np.exp(-x),
        )

    #! This may not be correct, should be checked
    def occupation_2nd_derivative(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        x = self._scale(bands)
        z = self._scale(bands) ** 2
        return np.where(
            z > MAX_EXP_ARG,
            -2.0 * x / np.sqrt(np.pi) * np.exp(-MAX_EXP_ARG),
            -2.0 * x / np.sqrt(np.pi) * np.exp(-z),
        )


class FermiDirac(Smearing):
    """Fermi-Dirac smearing."""

    def occupation(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        x = self._scale(bands)
        return np.where(x < -MAX_EXP_ARG, 0.0, np.where(x > MAX_EXP_ARG, 1.0, 1.0 / (1.0 + np.exp(-x))))

    def occupation_derivative(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        x = self._scale(bands)
        return np.where(np.abs(x) > MAX_EXP_ARG, 0.0, 1.0 / (2 + np.exp(-x) + np.exp(x)))

    #! This may not be correct, should be checked
    def occupation_2nd_derivative(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        x = self._scale(bands)
        return np.where(np.abs(x) > MAX_EXP_ARG, 0.0, -(np.exp(x) - np.exp(-x)) / (2.0 + np.exp(-x) + np.exp(x)) ** 2)


class Cold(Smearing):
    """Marzari-Vanderbilt-DeVita-Payne (cold) smearing."""

    def occupation(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        x = self._scale(bands) - 1.0 / np.sqrt(2.0)
        z = np.minimum(x**2, MAX_EXP_ARG)
        return 0.5 * sp.special.erf(x) + 1 / np.sqrt(2.0 * np.pi) * np.exp(-z) + 0.5

    def occupation_derivative(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        x = self._scale(bands) - 1.0 / np.sqrt(2.0)
        z = np.minimum(x**2, MAX_EXP_ARG)
        return 1 / (2 * np.sqrt(np.pi)) * np.exp(-z) * (2.0 - np.sqrt(2.0) * x)

    #! This may not be correct, should be checked
    def occupation_2nd_derivative(self, bands: npt.ArrayLike) -> npt.ArrayLike:
        x = self._scale(bands)
        z = np.minimum((x - 1.0 / np.sqrt(2.0)) ** 2, MAX_EXP_ARG)
        return 1 / (2 * np.sqrt(np.pi)) * np.exp(-z) * (2.0 * np.sqrt(2.0) * x**2 - 6.0 * x + np.sqrt(2.0))


def smearing_from_name(name: ty.Optional[ty.Union[str, int]]) -> Smearing:
    """Retreive a smearing class by name.

    Supported names are:

        * NoSmearing: None
        * GaussianSmearing: 'gauss', 'gaussian', '0', 0
        * FermiDiracSmearing: 'fd', 'f-d', 'fermi-dirac', '-99', -99
        * ColdSmearing: 'mv', 'm-v', 'marzari-vanderbilt', 'cold', '-1', -1

    Args:
        name (ty.Optional[ty.Union[str,int]]): smearing function name.

    Raises:
        ValueError: for unknown smearing funtions.

    Returns:
        Smearing: smearing class.
    """
    if name is None:
        smearing = Delta
    name = str(name).lower()
    if name in ("mv", "m-v", "marzari-vanderbilt", "cold", "-1"):
        smearing = Cold
    elif name in ("gauss", "gaussian", "0"):
        smearing = Gaussian
    elif name in ("fd", "f-d", "fermi-dirac", "-99"):
        smearing = FermiDirac
    else:
        raise ValueError(f"Unknown smearing function name {name}")
    return smearing
