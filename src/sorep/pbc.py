"""Convert reciprocal coordinates between fractional and Cartesian units."""

from collections import Counter
import typing as ty

import numpy as np
import numpy.typing as npt
import spglib

__all__ = (
    "direct_cart_to_frac",
    "direct_frac_to_cart",
    "recip_cart_to_frac",
    "recip_frac_to_cart",
    "get_kgrid_shape_from_density",
    "build_irreducible_kpoints",
)


def direct_cart_to_frac(cartesian_coords: npt.ArrayLike, cell: npt.ArrayLike) -> npt.ArrayLike:
    """Convert points in direct Cartesian coordinates to direct fractional coordinates.

    Args:
        cartesian_coords (npt.ArrayLike): Cartesian coordinates.
        cell (npt.ArrayLike): Direct lattice unit cell with rows as cell vectors.

    Returns:
        npt.ArrayLike: Fractional coordinates.
    """
    return np.dot(cartesian_coords, np.linalg.inv(cell))


def direct_frac_to_cart(fractional_coords: npt.ArrayLike, cell: npt.ArrayLike) -> npt.ArrayLike:
    """Convert points in direct fractional coordinates to direct Cartesian coordinates.

    Args:
        fractional_coords (npt.ArrayLike): Fractional coordinates.
        cell (npt.ArrayLike): Direct lattice unit cell with rows as cell vectors.

    Returns:
        npt.ArrayLike: Cartesian coordinates.
    """
    return np.dot(fractional_coords, cell)


def recip_cart_to_frac(cartesian_coords: npt.ArrayLike, cell: npt.ArrayLike) -> npt.ArrayLike:
    """Convert points in reciprocal Cartesian coordinates to reciprocal fractional coordinates.

    Args:
        cartesian_coords (npt.ArrayLike): Cartesian coordinates.
        cell (npt.ArrayLike): Direct lattice unit cell with rows as cell vectors.

    Returns:
        npt.ArrayLike: Fractional coordinates.
    """
    reciprocal_cell = 2 * np.pi * np.linalg.inv(cell).T
    return np.dot(cartesian_coords, np.linalg.inv(reciprocal_cell))


def recip_frac_to_cart(fractional_coords: npt.ArrayLike, cell: npt.ArrayLike) -> npt.ArrayLike:
    """Convert points in reciprocal fractional coordinates to reciprocal Cartesian coordinates.

    Args:
        fractional_coords (npt.ArrayLike): Fractional coordinates.
        cell (npt.ArrayLike): Direct lattice unit cell with rows as cell vectors.

    Returns:
        npt.ArrayLike: Cartesian coordinates.
    """
    reciprocal_cell = 2 * np.pi * np.linalg.inv(cell).T
    return np.dot(fractional_coords, reciprocal_cell)


def get_kgrid_shape_from_density(cell: npt.ArrayLike, kdensity: float = 0.1) -> ty.Tuple[int, int, int]:
    """
    Calculate the number of k-points in each reciprocal axis from a k-point density.

    :param cell: Unit cell matrix in Angstrom.
    :param kdensity: Minimum k-point density in 1/Angstrom.
    :returns kgrid: Number of k-points in each reciprocal axis direction.
    """
    reciprocal_cell = 2 * np.pi * np.linalg.inv(cell).T
    reciprocal_cell_lengths = np.linalg.norm(reciprocal_cell, axis=1)
    kgrid_shape = np.round(reciprocal_cell_lengths / kdensity).astype(int)
    kgrid_shape = tuple(np.maximum(1, kgrid_shape))

    return kgrid_shape


def build_irreducible_kpoints(
    cell: npt.ArrayLike,
    frac_coords: npt.ArrayLike,
    atomic_numbers: npt.ArrayLike,
    kmesh: ty.Union[ty.Tuple[int, int, int], npt.ArrayLike],
    is_shifted: ty.Union[ty.Tuple[int, int, int], npt.ArrayLike] = (0, 0, 0),
) -> ty.Tuple[npt.NDArray, npt.NDArray]:
    """
    Generate an irreducible k-point grid.

    :param structure: ase.Atoms or pymatgen.core.structure.Structure
    :param kmesh: Number of k-points in the first, second, and third reciprocal lattice directions
    :param is_shifted: Grid origin shift in the reciprocal lattice directions
    :returns: (
        kpoints: Irreducible k-point grid in fractional coordinates.
        weights: Weight of each k-point (number of full-grid k-points which map to each
            irreducible k-point, normalized so that the sum is 1).
    )
    """
    ir_mesh = spglib.get_ir_reciprocal_mesh(kmesh, (cell, frac_coords, atomic_numbers), is_shifted)
    if ir_mesh is None:
        raise ValueError("spglib failed to generate an irreducible reciprocal mesh.")
    ir_map, kgrid = ir_mesh
    kgrid = kgrid / kmesh

    # Count number of full grid points for each IR grid point
    weights = np.array(list(Counter(ir_map).values()), dtype=np.double)
    weights /= weights.sum()  # Normalize the sum to 1
    # Take the IR k-points from the full k-grid
    kpoints = kgrid[np.unique(ir_map)]

    return kpoints, weights
