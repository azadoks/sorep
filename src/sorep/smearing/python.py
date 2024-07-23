"""Electronic smearing functions."""

from abc import ABC
import typing as ty

import numpy as np
import numpy.typing as npt

__all__ = (
    "Smearing",
    "Delta",
    "Gaussian",
    "FermiDirac",
    "Cold",
    "smearing_from_name",
)

from .cython import ufuncs


class Smearing(ABC):
    """Abstract base class for smearing functions."""

    center: float
    width: float
    _occ: ty.Callable
    _docc: ty.Callable
    _ddocc: ty.Callable

    def __init__(self, center: float, width: float):
        self.center = center
        self.width = width

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(center={self.center}, width={self.width})"

    def __str__(self) -> str:
        return self.__repr__()

    def _scale(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return (x - self.center) / self.width

    def _clip_occ(self, x: npt.ArrayLike, max_exponent: ty.Optional[float] = None) -> npt.ArrayLike:
        max_exponent = max_exponent or np.inf
        where_lt = x < -max_exponent
        where_gt = x > max_exponent
        if max_exponent is not None:
            x[where_lt] = 1.0
            x[where_gt] = 0.0
        return x, ~(where_lt | where_gt)

    def _clip_docc(self, x: npt.ArrayLike, max_exponent: ty.Optional[float] = None) -> npt.ArrayLike:
        max_exponent = max_exponent or np.inf
        where_gt = np.abs(x) > np.sqrt(max_exponent)
        x[where_gt] = 0.0
        return x, ~where_gt

    def occupation(self, x: npt.ArrayLike, max_exponent: ty.Optional[float] = 512.0) -> npt.ArrayLike:
        """Compute the occupation function (cumulative distribution function mirrored around x)."""
        z: npt.ArrayLike = self._scale(x)
        z, where = self._clip_occ(z, max_exponent)
        return self._occ(z, out=z, where=where)

    def occupation_derivative(self, x: npt.ArrayLike, max_exponent: ty.Optional[float] = 36.0) -> npt.ArrayLike:
        """Compute the negative derivative of the occupation function (probability density function)."""
        z: npt.ArrayLike = self._scale(x)
        z, where = self._clip_docc(z, max_exponent)
        return self._docc(z, out=z, where=where)

    def occupation_2nd_derivative(self, x: npt.ArrayLike, max_exponent: ty.Optional[float] = 36.0) -> npt.ArrayLike:
        """Compute the negative second derivative (curvature) of the occupation function."""
        z: npt.ArrayLike = self._scale(x)
        z, where = self._clip_docc(z, max_exponent)
        return self._ddocc(z, out=z, where=where)


class Delta(Smearing):
    """No smearing, i.e. a true step/delta function."""

    def occupation(self, x: npt.ArrayLike, max_exponent: ty.Optional[float] = None) -> npt.ArrayLike:
        x = self._scale(x)
        return np.where(x <= 0.0, 1.0, 0.0)

    def occupation_derivative(self, x: npt.ArrayLike, max_exponent: ty.Optional[float] = None) -> npt.ArrayLike:
        return np.zeros_like(x)

    def occupation_2nd_derivative(self, x: npt.ArrayLike, max_exponent: ty.Optional[float] = None) -> npt.ArrayLike:
        return np.zeros_like(x)


class FermiDirac(Smearing):
    """Fermi-Dirac smearing."""

    def __init__(self, center: float, width: float):
        super().__init__(center, width)
        self._occ = ufuncs.fermi_occ
        self._docc = ufuncs.fermi_docc
        self._ddocc = ufuncs.fermi_ddocc

    def _clip_docc(self, x: npt.ArrayLike, max_exponent: ty.Optional[float] = None) -> npt.ArrayLike:
        max_exponent = max_exponent or np.inf
        where_gt = np.abs(x) > max_exponent
        x[where_gt] = 0.0
        return x, ~where_gt


class Gaussian(Smearing):
    """Gaussian smearing."""

    def __init__(self, center: float, width: float):
        super().__init__(center, width)
        self._occ = ufuncs.gauss_occ
        self._docc = ufuncs.gauss_docc
        self._ddocc = ufuncs.gauss_ddocc


class Cold(Smearing):
    """Marzari-Vanderbilt-DeVita-Payne (cold) smearing."""

    def __init__(self, center: float, width: float):
        super().__init__(center, width)
        self._occ = ufuncs.cold_occ
        self._docc = ufuncs.cold_docc
        self._ddocc = ufuncs.cold_ddocc


def smearing_from_name(name: ty.Optional[ty.Union[str, int]]) -> Smearing:
    """Retreive a smearing class by name.

    Supported names are:

        * Delta: None, 'fixed'
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
    if isinstance(name, int):
        name = str(name)
    if isinstance(name, str):
        name = name.lower()
    if name is None or name == "fixed":
        smearing = Delta
    if name in ("mv", "m-v", "marzari-vanderbilt", "cold", "-1"):
        smearing = Cold
    elif name in ("gauss", "gaussian", "0"):
        smearing = Gaussian
    elif name in ("fd", "f-d", "fermi", "fermi-dirac", "-99"):
        smearing = FermiDirac
    else:
        raise ValueError(f"Unknown smearing function name {name}")
    return smearing
