"""Material dataclass."""

from dataclasses import dataclass
import typing as ty

from ase import Atoms
import h5py

from .band_structure import BandStructure

__all__ = ()


@dataclass
class MaterialData:
    """Data class containing information about a material as a whole (structure, bands, and various metadata)."""

    atoms: Atoms
    bands: BandStructure
    metadata: dict

    @classmethod
    def from_hdf(cls, hdf: ty.Union[h5py.File, h5py.Group]) -> "MaterialData":
        """Load the material data from an HDF5 file or group containing the following structure:

        - `attrs`
            - Metadata attributes.
        - atoms
            - positions: Atomic positions.
            - numbers: Atomic numbers.
            - masses: Atomic masses (optional).
            - cell: Unit cell.
            - pbc: Periodic boundary conditions.
        - bands
            - eigenvalues: Eigenvalues.
            - kpoints: K-points.
            - weights: Weights.
            - cell: Cell.
            - occupations: Occupations.
            - labels: Labels.
            - label_numbers: Label numbers.
            - fermi_energy: Fermi energy.
            - n_electrons: Number of electrons.

        Args:
            hdf (ty.Union[h5py.File, h5py.Group]): HDF5 file or group containing the material data.
        """
        atoms = Atoms(
            positions=hdf["atoms/positions"][()],
            numbers=hdf["atoms/numbers"][()],
            masses=hdf["atoms/masses"][()] if "masses" in hdf["atoms"] else None,
            cell=hdf["atoms/cell"][()],
            pbc=hdf["atoms/pbc"][()],
        )
        return cls(
            atoms=atoms,
            bands=BandStructure.from_hdf(hdf["bands"]),
            metadata={
                "atoms": dict(hdf["atoms"].attrs),
                "bands": dict(hdf["bands"].attrs),
                "calculation": dict(hdf.attrs),
            },
        )
