import json
import os
import typing as ty
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy import ma


from .dos import smeared_dos
from .smearing import smearing_from_name

__all__ = ('BandStructure',)

# TODO: clarify weights and occupations w.r.t. number of spins in the documentation
# TODO: clarify units of the k-points in the documentation

@dataclass
class BandStructure:
    """Data class containing band structure information."""
    bands: npt.NDArray[np.float64]
    kpoints: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    occupations: ty.Optional[npt.NDArray[np.float64]] = None
    labels: ty.Optional[npt.NDArray[np.float64]] = None
    label_numbers: ty.Optional[npt.NDArray[np.float64]] = None
    fermi_energy: ty.Optional[float] = None
    n_electrons: ty.Optional[int] = None

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'n_spins={self.n_spins}'
                f', n_kpoints={self.n_kpoints}'
                f', n_bands={self.n_bands}'
                f', n_electrons={self.n_electrons}'
                f', fermi_energy={self.fermi_energy}'
                ')')

    @classmethod
    def from_npz_metadata(cls,
                          npz_path: os.PathLike,
                          json_path: ty.Optional[os.PathLike] = None) -> None:
        """Load a band structure from an NPZ file containing arrays and
        a JSON file containing metadata.

        The NPZ file should contain the following arrays:
        - bands: (n_spins, n_kpoints, n_bands) Eigenvalues/bands.
        - kpoints: (n_kpoints, 3) K-points.
        - weights: (n_kpoints,) K-point weights.
        - occupations: (n_spins, n_kpoints, n_bands) Occupations (optional).
        - labels: (n_labels,) Labels (optional).
        - label_numbers: (n_labels,) Label indices (optional).

        The JSON file should contain the following metadata:
        - fermi_energy: Fermi energy.
        - number_of_electrons: Number of electrons.

        Args:
            path (os.PathLike): Path to the NPZ file.
        """
        # Load metadata (Fermi energy, number of electrons, etc.)
        metadata = {}
        if json_path:
            with open(json_path, 'r', encoding='utf-8') as fp:
                metadata = json.load(fp)
        # Load arrays (eigenvalues/bands, k-points, etc.)
        with open(npz_path, 'rb') as fp:
            arrays = dict(np.load(fp))
        # Add a spin dimension if not present
        if arrays['bands'].ndim == 2:
            arrays['bands'] = np.expand_dims(arrays['bands'], 0)
        if arrays['occupations'].ndim == 2:
            arrays['occupations'] = np.expand_dims(arrays['occupations'], 0)
        # Ignore occupations if all zeros
        if np.all(np.isclose(arrays['occupations'], 0)):
            arrays.pop('occupations')
        return cls(**arrays,
                   fermi_energy=metadata.get('fermi_energy'),
                   n_electrons=metadata.get('number_of_electrons'))

    @property
    def n_spins(self) -> int:
        """Number of spin channels.

        Returns:
            int: Number of spin channels
        """
        return self.bands.shape[0]

    @property
    def n_kpoints(self):
        """Number of k-points.

        Returns:
            int: Number of k-points.
        """
        return self.bands.shape[1]

    @property
    def n_bands(self) -> int:
        """Number of bands.

        Returns:
            int: Number of bands.
        """
        return self.bands.shape[2]

    @property
    def max_occupation(self) -> int:
        """Maximum occupation (1 for non-spin-polarized, 2 for collinear spin-polarized).

        Returns:
            int: Maximum occupation.
        """
        if self.n_spins == 1:
            return 2
        elif self.n_spins == 2:
            return 1
        else:
            raise ValueError(
                f'Unknown maximum occupation for n_spins={self.n_spins}')

    def compute_occupations_from_fermi(self, fermi_energy: float,
                                       smearing_type: str,
                                       smearing_width: float) -> npt.NDArray:
        """Compute the occupations given a Fermi energy and smearing.

        Args:
            fermi_energy (float): Fermi energy.
            smearing_type (str): Smearing type (see `smearing_from_name`).
            smearing_width (float): Smearing width.

        Raises:
            ValueError: If number of electrons is unknown.

        Returns:
            npt.NDArray: Occupations array.
        """
        if self.n_electrons is None:
            raise ValueError(
                "Cannot compute occupations if number of electrons is unknown."
            )
        smearing = smearing_from_name(smearing_type)(center=fermi_energy,
                                                     width=smearing_width)
        return self.max_occupation * smearing.occupation(self.bands)

    def compute_smeared_dos(self, energies: npt.NDArray,
                            smearing_type: ty.Union[str, int],
                            smearing_width: float) -> npt.NDArray:
        """Compute a smeared density of states from the band structure (see `smeared_dos`).

        Args:
            energies (npt.NDArray): energies at which to sample the DOS
            smearing_type (ty.Union[str,int]): type of smearing (see `smearing_from_name`)
            smearing_width (float): smearing width

        Returns:
            npt.NDArray: (n_spins, n_energies) array containing the DOS for each spin channel
        """
        return smeared_dos(energies, self.bands, self.weights, smearing_type,
                           smearing_width)

    def is_metallic(self, tol: float = 1e-6) -> bool:
        """Check if the band structure is metallic.

        Args:
            tol (float, optional): Tolerance on differences in energy. Defaults to 1e-6.

        Returns:
            bool: True if any band has at least one eigenvalue above and below the Fermi energy +/- tol.
        """
        if self.fermi_energy is None:
            return None

        # Flatten the spin axis so that each row ia a band
        # First reorder the axes to (n_spins, n_bands, n_kpoints)
        # Then reshape the array to (n_spins * n_bands, n_kpoints)
        # The first n_bands rows of the resulting array correspond to spin 0, and the rest to spin 1
        bands = np.transpose(self.bands, (0, 2, 1)).reshape((self.n_spins * self.n_bands, self.n_kpoints))

        return bool(
            np.any(
                np.any(bands < self.fermi_energy - tol, axis=1)
                & np.any(bands > self.fermi_energy + tol, axis=1)))

    def is_insulating(self, tol: float = 1e-6) -> bool:
        """Check if the band structure is insulating.

        Args:
            tol (float, optional): Tolerance on differences in energy. Defaults to 1e-6.

        Returns:
            bool: True if no band straddles the Fermi energy within tol.
        """
        is_metallic = self.is_metallic(tol)
        if is_metallic is None:
            return None
        return not is_metallic

    @property
    def vbm(self) -> float:
        """The valence band maximum.

        Returns:
            float: valence band maximum.
        """
        if self.fermi_energy is None:
            return None
        # Mask the conduction bands and find the maximum of the rest of the bands,
        # i.e. the valence bands
        return np.max(ma.array(self.bands, mask=self.bands > self.fermi_energy))

    @property
    def cbm(self) -> float:
        """The conduction band minimum.

        Returns:
            float: conduction band minimum.
        """
        if self.fermi_energy is None:
            return None
        # See `vbm`
        return np.min(ma.array(self.bands, mask=self.bands < self.fermi_energy))
