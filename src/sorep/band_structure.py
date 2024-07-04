"""Band structure data class and related functions."""

import json
import os
import typing as ty

# import numpy as np
import jax.numpy as np
import numpy.typing as npt

from .dos import smeared_dos
from .smearing import smearing_from_name

__all__ = ()


class BandStructure:
    # pylint: disable=too-many-instance-attributes
    """Band structure data and useful operations."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        bands: npt.NDArray[np.float64],
        kpoints: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        occupations: ty.Optional[npt.NDArray[np.float64]] = None,
        labels: ty.Optional[npt.NDArray[np.float64]] = None,
        label_numbers: ty.Optional[npt.NDArray[np.float64]] = None,
        fermi_energy: ty.Optional[float] = None,
        n_electrons: ty.Optional[int] = None,
    ):
        """Initialize a band structure.

        Args:
            bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands.
            kpoints (npt.NDArray[np.float64]): (n_kpoints, 3) k-points.
            weights (npt.NDArray[np.float64]): (n_kpoints,) k-point weights.
            occupations (ty.Optional[npt.NDArray[np.float64]], optional): (n_spins, n_kpoints, n_bands)
                band occupations. Defaults to None.
            labels (ty.Optional[npt.NDArray[np.float64]], optional): (n_labels,) k-point labels. Defaults to None.
            label_numbers (ty.Optional[npt.NDArray[np.float64]], optional): (n_labels,) k-point label indices.
                Defaults to None.
            fermi_energy (ty.Optional[float], optional): Fermi energy. Defaults to None.
            n_electrons (ty.Optional[int], optional): number of electrons. Defaults to None.
        """
        # Check all the shapes
        assert bands.ndim == 3
        assert kpoints.ndim == 2
        assert kpoints.shape[1] == 3
        assert weights.ndim == 1
        assert kpoints.shape[0] == weights.shape[0] == bands.shape[1]
        if occupations is not None:
            assert occupations.ndim == 3
            assert occupations.shape == bands.shape
        if labels is not None:
            assert labels.ndim == 1
        if label_numbers is not None:
            assert label_numbers.ndim == 1
        if labels is not None and label_numbers is not None:
            assert labels.shape == label_numbers.shape
        # Check that the Fermi energy makes sense
        if fermi_energy is not None:
            assert np.min(bands) <= fermi_energy <= np.max(bands)
        # Check that the number of electrons makes sense
        if n_electrons is not None:
            assert n_electrons >= 0

        # Normalize the sum of the k-weights to 1
        total_weight = weights.sum()
        if not (np.isclose(total_weight, 1, atol=1e-8) or np.isclose(total_weight, 2, atol=1e-8)):
            raise ValueError(f"Total weight is {total_weight}, is expected to be 1 or 2.")
        weights /= total_weight

        self.bands = bands
        self.kpoints = kpoints
        self.weights = weights
        self.occupations = occupations
        self.labels = labels
        self.label_numbers = label_numbers
        self.fermi_energy = fermi_energy
        self.n_electrons = n_electrons

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_spins={self.n_spins}"
            f", n_kpoints={self.n_kpoints}"
            f", n_bands={self.n_bands}"
            f", n_electrons={self.n_electrons}"
            f", fermi_energy={self.fermi_energy}"
            ")"
        )

    @classmethod
    def from_npz_metadata(cls, npz_path: os.PathLike, json_path: ty.Optional[os.PathLike] = None) -> None:
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
            with open(json_path, "r", encoding="utf-8") as fp:
                metadata = json.load(fp)
        # Load arrays (eigenvalues/bands, k-points, etc.)
        with open(npz_path, "rb") as fp:
            arrays = dict(np.load(fp))
        # Add a spin dimension if not present
        if arrays["bands"].ndim == 2:
            arrays["bands"] = np.expand_dims(arrays["bands"], 0)
        if arrays["occupations"].ndim == 2:
            arrays["occupations"] = np.expand_dims(arrays["occupations"], 0)
        # Ignore occupations and Fermi energy if occupations all zeros
        if np.all(np.isclose(arrays["occupations"], 0)):
            arrays.pop("occupations")
            metadata.pop("fermi_energy", None)
        return cls(**arrays, fermi_energy=metadata.get("fermi_energy"), n_electrons=metadata.get("number_of_electrons"))

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
    def max_occupation(self) -> float:
        """Maximum occupation (1 for non-spin-polarized, 2 for collinear spin-polarized).

        Returns:
            float: Maximum occupation.
        """
        if self.n_spins == 1:
            max_occupation = 2.0
        elif self.n_spins == 2:
            max_occupation = 1.0
        else:
            raise ValueError(f"Unknown maximum occupation for n_spins={self.n_spins}")
        return max_occupation

    def compute_occupations_from_fermi(
        self, fermi_energy: float, smearing_type: str, smearing_width: float
    ) -> npt.NDArray:
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
            raise ValueError("Cannot compute occupations if number of electrons is unknown.")
        smearing = smearing_from_name(smearing_type)(center=fermi_energy, width=smearing_width)
        return self.max_occupation * smearing.occupation(self.bands)

    def compute_smeared_dos(
        self, energies: npt.NDArray, smearing_type: ty.Union[str, int], smearing_width: float
    ) -> npt.NDArray:
        """Compute a smeared density of states from the band structure (see `smeared_dos`).

        Args:
            energies (npt.NDArray): energies at which to sample the DOS
            smearing_type (ty.Union[str,int]): type of smearing (see `smearing_from_name`)
            smearing_width (float): smearing width

        Returns:
            npt.NDArray: (n_spins, n_energies) array containing the DOS for each spin channel
        """
        return smeared_dos(energies, self.bands, self.weights, smearing_type, smearing_width)

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
            np.any(np.any(bands < self.fermi_energy - tol, axis=1) & np.any(bands > self.fermi_energy + tol, axis=1))
        )

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
        return np.max(self.bands[self.bands < self.fermi_energy])

    @property
    def cbm(self) -> float:
        """The conduction band minimum.

        Returns:
            float: conduction band minimum.
        """
        if self.fermi_energy is None:
            return None
        # See `vbm`
        return np.min(self.bands[self.bands > self.fermi_energy])
