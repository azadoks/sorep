"""Compute operator spectra with PySCF."""

import typing as ty

from ase import Atoms
from importlib_resources import files
import numpy as np
import numpy.typing as npt
from pyscf.lib.linalg_helper import safe_eigh
from pyscf.pbc import gto
from pyscf.pbc.gto import Cell
from scipy.linalg import eigh

from ..band_structure import BandStructure
from ..constants import ANGSTROM_TO_BOHR, HARTREE_TO_EV
from ..data import gto_basis_sets
from ..pbc import (
    build_irreducible_kpoints,
    direct_cart_to_frac,
    get_kgrid_shape_from_density,
    recip_cart_to_frac,
    recip_frac_to_cart,
)

SUPPORTED_EIGEN_SOLVERS = ("eigh", "safe_eigh")
BASIS_SETS = {
    "ano-ml-os": files(gto_basis_sets) / "ano-ml-os.dat",
    "ano-ml-ae": files(gto_basis_sets) / "ano-ml-ae.dat",
}

__all__ = ("compute_one_electron_spectrum",)


def _load_basis(basis: str, atoms: Atoms, **kwargs) -> ty.Dict[str, ty.Any]:
    basis_path = BASIS_SETS.get(basis, basis)
    symbols = set(sym for sym in atoms.symbols)
    return {sym: gto.basis.load(str(basis_path), sym, **kwargs) for sym in symbols}


def _build_cell(
    atoms: Atoms,
    basis: str,
    basis_kwargs: ty.Optional[ty.Dict[str, ty.Any]] = None,
    exp_to_discard: float = 0.1,
    **kwargs,
) -> Cell:
    """Build a PySCF Cell object from an ASE Atoms object.

    Args:
        atoms (Atoms): Atomic structure.
        basis (str): GTO basis set name.
        basis_kwargs (ty.Optional[ty.Dict[str, ty.Any]], optional): Additional basis set arguments. Defaults to None.
        exp_to_discard (float, optional): Exponent threshold for basis set truncation. Defaults to 0.1.

    Returns:
        Cell: PySCF Cell object.
    """
    basis_kwargs = basis_kwargs or {}
    cell_array = atoms.cell.array * ANGSTROM_TO_BOHR
    atom = list(zip(atoms.symbols, atoms.positions * ANGSTROM_TO_BOHR))
    cell = Cell(
        a=cell_array,
        atom=atom,
        unit="Bohr",
        basis=_load_basis(basis, atoms, **basis_kwargs),
        exp_to_discard=exp_to_discard,
        **kwargs,
    )
    cell.build()
    return cell


def _diagonalize(operator_matrix: npt.ArrayLike, overlap_matrix: npt.ArrayLike, eigen_solver: str) -> npt.ArrayLike:
    if eigen_solver == "eigh":
        eigvals = eigh(
            operator_matrix,
            b=overlap_matrix,
            type=1,  # a @ v = w @ b @ v
            check_finite=True,
            eigvals_only=True,
            overwrite_a=True,  # we don't need the operator matrix any more
            overwrite_b=True,  # we don't need the overlap matrix any more
        )
        return eigvals
    if eigen_solver == "safe_eigh":
        eigvals, _, _ = safe_eigh(operator_matrix, overlap_matrix)
        return eigvals
    raise ValueError(f"Unknown eigen_solver {eigen_solver}.")


def _build_kpoints(
    cell: Cell,
    kdensity: ty.Optional[float] = None,
    kgrid_shape: ty.Optional[ty.Sequence[int]] = None,
    use_symmetries: bool = False,
) -> ty.Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, int]:
    if kgrid_shape is None:
        assert kdensity is not None, "Either `kdensity` or `kgrid_shape` must be specified."
        kgrid_shape: ty.Sequence[int] = get_kgrid_shape_from_density(cell=cell.a, kdensity=kdensity)
    if use_symmetries:
        kpoints_frac, weights = build_irreducible_kpoints(
            cell=cell.a,
            frac_coords=direct_cart_to_frac(cell.atom_coords(), cell.a),
            atomic_numbers=cell.atom_charges(),
            kmesh=kgrid_shape,
        )
        kpoints_cart = recip_frac_to_cart(kpoints_frac, cell.a)  # 1/Bohr
        n_kpoints = kpoints_frac.shape[0]
    else:
        kpoints_cart = cell.make_kpts(kgrid_shape)  # 1/Bohr
        kpoints_frac = recip_cart_to_frac(kpoints_cart, cell.a)
        n_kpoints = kpoints_cart.shape[0]
        weights = np.full((n_kpoints,), 1 / n_kpoints)

    return kpoints_cart, kpoints_frac, weights, n_kpoints


def compute_one_electron_spectrum(  # pylint: disable=too-many-arguments,too-many-locals
    atoms: Atoms,
    basis: str,
    operator: str = "int1e_kin",
    kdensity: ty.Optional[float] = None,
    kgrid_shape: ty.Optional[ty.Sequence[int]] = None,
    eigen_solver: str = "eigh",
    use_symmetries: bool = True,
) -> BandStructure:
    """Compute the eigenspectrum of the provided one-electron integral of a periodic system using PySCF.

    Args:
        atoms (Atoms): Atomic structure.
        basis (str): GTO basis set.
        kdensity (float, optional): k-point density in 1/Å. Defaults to 0.15.
        kgrid_shape (ty.Optional[ty.Sequence[int]], optional): Monkhorst-Pack k-point grid shape (e.g. [5, 5, 5]).
        Defaults to None.
        eigen_solver (str, optional): Eigensolver algorithm. Defaults to "eigh".
        use_symmetries (bool, optional): Use spglib to construct symmetry-irreducible k-points. Defaults to True.

    Raises:
        AssertionError: Atoms must have full 3D periodic boundary conditions.
        AssertionError: Only one-electron operators are supported.
        ValueError: At least one of k_dist or kgrid_shape must be specified.
        ValueError: Only one of k_dist or kgrid_shape must be specified.
        ValueError: Unsupported eigen_solver.

    Returns:
        BandStructure: Bandstructure object containing the eigenvalues, k-points, and k-point weights.
    """
    assert np.all(atoms.pbc), "Full 3D periodic boundary conditions are required."
    assert operator.startswith("int1e"), "Only one-electron operators are supported."
    if kdensity is None and kgrid_shape is None:
        raise ValueError("Either `k_dist` or `kgrid_shape` must be specified.")
    if kdensity is not None and kgrid_shape is not None:
        raise ValueError("Only one of `kdensity` or `kgrid_shape` must be specified.")
    if eigen_solver not in SUPPORTED_EIGEN_SOLVERS:
        raise ValueError(f"Unsupported `eigen_solver` {eigen_solver}.")
    # Convert kdensity to 1/Bohr
    if kdensity is not None:
        kdensity_bohr: float = kdensity / ANGSTROM_TO_BOHR
    else:
        kdensity_bohr = None
    cell: Cell = _build_cell(atoms, basis)
    kpoints_cart, kpoints_frac, weights, n_kpoints = _build_kpoints(cell, kdensity_bohr, kgrid_shape, use_symmetries)
    # Allocate the bands array
    n_bands = cell.nao_nr()
    bands = np.full((n_kpoints, n_bands), np.nan)
    # Calculate the operator and overlap matrices for each k-point
    operator_matrices = cell.pbc_intor("int1e_kin", kpts=kpoints_cart)
    overlap_matrices = cell.pbc_intor("int1e_ovlp", kpts=kpoints_cart)
    # Diagonalize the operator matrices
    for i, (operator_matrix, overlap_matrix) in enumerate(zip(operator_matrices, overlap_matrices)):
        eigvals = _diagonalize(operator_matrix, overlap_matrix, eigen_solver)
        bands[i, 0 : eigvals.shape[0]] = eigvals
    # Add spin axis, convert Ha -> eV
    bands = np.expand_dims(bands, axis=0) * HARTREE_TO_EV
    return BandStructure(eigenvalues=bands, kpoints=kpoints_frac, weights=weights, cell=atoms.cell.array)
