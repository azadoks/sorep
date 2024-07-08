"""Convert reciprocal coordinates between fractional and Cartesian units."""

import numpy as np
import numpy.typing as npt

__all__ = ("recip_cart_to_frac", "recip_frac_to_cart")


def recip_cart_to_frac(cartesian_coords: npt.ArrayLike, cell: npt.ArrayLike) -> npt.ArrayLike:
    """Convert points in reciprocal Cartesian coordinates to reciprocal fractional coordinates.

    Args:
        cartesian_coords (npt.ArrayLike): Cartesian coordinates.
        cell (npt.ArrayLike): Direct lattice unit cell with rows as cell vectors.

    Returns:
        npt.ArrayLike: Fractional coordinates
    """
    reciprocal_cell = 2 * np.pi * np.linalg.inv(cell).T
    return np.dot(cartesian_coords, np.linalg.inv(reciprocal_cell))


def recip_frac_to_cart(fractional_coords: npt.ArrayLike, cell: npt.ArrayLike) -> npt.ArrayLike:
    """Convert points in reciprocal fractional coordinates to reciprocal Cartesian coordinates.

    Args:
        fractional_coords (npt.ArrayLike): Fractional coordinates.
        cell (npt.ArrayLike): Direct lattice unit cell with rows as cell vectors.

    Returns:
        npt.ArrayLike: Cartesian coordinates
    """
    reciprocal_cell = 2 * np.pi * np.linalg.inv(cell).T
    return np.dot(fractional_coords, reciprocal_cell)
