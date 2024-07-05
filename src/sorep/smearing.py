"""Electronic smearing functions."""

from abc import ABC, abstractmethod
import typing as ty

import jax.typing as jty

# import jax.numpy as jnp
# import jax.scipy as jsp
import numpy as jnp
import scipy as jsp

__all__ = (
    "Smearing",
    "Delta",
    "Gaussian",
    "FermiDirac",
    "Cold",
    "smearing_from_name",
)

MAX_EXPONENT = 200.0


class Smearing(ABC):
    """Abstract base class for smearing functions."""

    def __init__(self, center: float, width: float):
        self.center = center
        self.width = width

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(center={self.center}, width={self.width})"

    def __str__(self) -> str:
        return self.__repr__()

    def _scale(self, x: jty.ArrayLike) -> jty.ArrayLike:
        return (x - self.center) / self.width

    @abstractmethod
    def occupation(self, x: jty.ArrayLike) -> jty.ArrayLike:
        """Compute the occupation function (cumulative distribution function mirrored around x)."""

    @abstractmethod
    def occupation_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
        """Compute the negative derivative of the occupation function (probability density function)."""

    @abstractmethod
    def occupation_2nd_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
        """Compute the negative second derivative (curvature) of the occupation function."""


class Delta(Smearing):
    """No smearing, i.e. a true step/delta function."""

    def occupation(self, x: jty.ArrayLike) -> jty.ArrayLike:
        x = self._scale(x)
        return jnp.where(x <= 0.0, 1.0, 0.0)

    def occupation_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
        return jnp.zeros_like(x)

    def occupation_2nd_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
        return jnp.zeros_like(x)


class FermiDirac(Smearing):
    """Fermi-Dirac smearing."""

    def occupation(self, x: jty.ArrayLike) -> jty.ArrayLike:
        x = self._scale(x)
        return jnp.where(x < -MAX_EXPONENT, 1.0, jnp.where(x > MAX_EXPONENT, 0.0, 1.0 / (1.0 + jnp.exp(x))))

    def occupation_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
        x = self._scale(x)
        dx = jnp.where(jnp.abs(x) > MAX_EXPONENT, 0.0, -1.0 / (2.0 + jnp.exp(x) + jnp.exp(-x)))  # avoid overflow
        return -dx  # pylint: disable=invalid-unary-operand-type

    def occupation_2nd_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
        x = self._scale(x)
        ddx = jnp.where(
            jnp.abs(x) > MAX_EXPONENT,  # avoid overflow
            0.0,
            (jnp.exp(x) - jnp.exp(-x)) / (2.0 + jnp.exp(-x) + jnp.exp(x)) ** 2,
        )
        return -ddx  # pylint: disable=invalid-unary-operand-type


class Gaussian(Smearing):
    """Gaussian smearing."""

    def occupation(self, x: jty.ArrayLike) -> jty.ArrayLike:
        x = self._scale(x)
        return jnp.where(x < -MAX_EXPONENT, 1.0, jnp.where(x > MAX_EXPONENT, 0.0, 0.5 * jsp.special.erfc(x)))

    def occupation_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
        x = self._scale(x)
        dx = jnp.where(
            jnp.abs(x) > jnp.sqrt(MAX_EXPONENT), 0.0, -1.0 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))  # avoid overflow
        )
        return -dx  # pylint: disable=invalid-unary-operand-type

    def occupation_2nd_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
        x = self._scale(x)
        ddx = jnp.where(
            jnp.abs(x) > jnp.sqrt(MAX_EXPONENT), 0.0, 2.0 * x / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))  # avoid overflow
        )
        return -ddx  # pylint: disable=invalid-unary-operand-type

    # def occupation(self, x: jty.ArrayLike) -> jty.ArrayLike:
    #     x = self._scale(x)
    #     return 0.5 * jsp.special.erfc(x)

    # def occupation_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
    #     x = self._scale(x)
    #     dx = -1.0 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))
    #     return -dx

    # def occupation_2nd_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
    #     x = self._scale(x)
    #     ddx = 2.0 * x / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))
    #     return -ddx


class Cold(Smearing):
    """Marzari-Vanderbilt-DeVita-Payne (cold) smearing."""

    def occupation(self, x: jty.ArrayLike) -> jty.ArrayLike:
        z = self._scale(x) + 1.0 / jnp.sqrt(2.0)
        return jnp.where(
            z < -jnp.sqrt(MAX_EXPONENT),  # avoid overflow
            1.0,
            jnp.where(
                z > jnp.sqrt(MAX_EXPONENT),  # avoid overflow
                0.0,
                -0.5 * jsp.special.erf(z) + 1 / jnp.sqrt(2.0 * jnp.pi) * jnp.exp(-(z**2)) + 0.5,
            ),
        )

    def occupation_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
        z = self._scale(x) + 1.0 / jnp.sqrt(2.0)
        dx = jnp.where(
            jnp.abs(z) > jnp.sqrt(MAX_EXPONENT),  # avoid overflow
            0.0,
            -jnp.exp(-(z**2)) * (jnp.sqrt(2) * z + 1) / jnp.sqrt(jnp.pi),
        )
        return -dx  # pylint: disable=invalid-unary-operand-type

    def occupation_2nd_derivative(self, x: jty.ArrayLike) -> jty.ArrayLike:
        z = self._scale(x) + 1.0 / jnp.sqrt(2.0)
        ddx = jnp.where(
            jnp.abs(z) > jnp.sqrt(MAX_EXPONENT),  # avoid overflow
            0.0,
            jnp.exp(-(z**2)) * (2.0 * jnp.sqrt(2.0) * z**2 + 2.0 * z - jnp.sqrt(2.0)) / jnp.sqrt(jnp.pi),
        )
        return -ddx  # pylint: disable=invalid-unary-operand-type


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
    elif name in ("fd", "f-d", "fermi-dirac", "-99"):
        smearing = FermiDirac
    else:
        raise ValueError(f"Unknown smearing function name {name}")
    return smearing
