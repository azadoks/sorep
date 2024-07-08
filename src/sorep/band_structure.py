"""Band structure data class and related functions."""

import json
import os
import typing as ty

from ase.io import read
import numpy as np
import numpy.typing as npt

from . import fermi
from .band_segment import BandPathSegment
from .dos import smeared_dos
from .pbc import recip_cart_to_frac, recip_frac_to_cart

__all__ = ()


class BandStructure:
    # pylint: disable=too-many-instance-attributes,too-many-branches,too-many-statements
    """Band structure data and useful operations."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        bands: npt.ArrayLike,
        kpoints: npt.ArrayLike,
        weights: npt.ArrayLike,
        cell: npt.ArrayLike,
        occupations: ty.Optional[npt.ArrayLike] = None,
        labels: ty.Optional[npt.ArrayLike] = None,
        label_numbers: ty.Optional[npt.ArrayLike] = None,
        fermi_energy: ty.Optional[float] = None,
        n_electrons: ty.Optional[int] = None,
        kpoints_are_cartesian: bool = False,
    ):
        """Initialize a band structure.

        Args:
            bands (npt.ArrayLike): (n_spins, n_kpoints, n_bands) eigenvalues/bands.
            kpoints (npt.ArrayLike): (n_kpoints, 3) k-points.
            weights (npt.ArrayLike): (n_kpoints,) k-point weights.
            cell (npt.ArrayLike): (3, 3) unit cell with rows as the cell vectors.
            occupations (ty.Optional[npt.ArrayLike], optional): (n_spins, n_kpoints, n_bands)
                band occupations. Defaults to None.
            labels (ty.Optional[npt.ArrayLike], optional): (n_labels,) k-point labels. Defaults to None.
            label_numbers (ty.Optional[npt.ArrayLike], optional): (n_labels,) k-point label indices.
                Defaults to None.
            fermi_energy (ty.Optional[float], optional): Fermi energy. Defaults to None.
            n_electrons (ty.Optional[int], optional): number of electrons. Defaults to None.
        """
        # Convert to numpy arrays
        bands = np.ascontiguousarray(bands)
        kpoints = np.ascontiguousarray(kpoints)
        weights = np.ascontiguousarray(weights)
        cell = np.ascontiguousarray(cell)
        if occupations is not None:
            occupations = np.ascontiguousarray(occupations)

        # Check all the shapes
        if bands.ndim == 2:  # Add a spin dimension if not present
            bands = np.expand_dims(bands, 0)
        assert bands.ndim == 3

        assert kpoints.ndim == 2
        assert kpoints.shape[1] == 3

        assert weights.ndim == 1

        assert cell.ndim == 2
        assert cell.shape[0] == cell.shape[1] == 3

        assert kpoints.shape[0] == weights.shape[0] == bands.shape[1]

        if occupations is not None:
            if occupations.ndim == 2:  # Add a spin dimension if not present
                occupations = np.expand_dims(occupations, 0)
            assert occupations.shape == bands.shape

        if labels is not None:
            assert labels.ndim == 1

        if label_numbers is not None:
            assert label_numbers.ndim == 1

        if labels is not None and label_numbers is not None:
            assert labels.shape == label_numbers.shape

        # Check that the number of electrons makes sense
        if n_electrons is not None:
            assert n_electrons >= 0

        # Convert to fractional coordinates if necessary
        if kpoints_are_cartesian:
            kpoints = recip_cart_to_frac(kpoints, cell)

        # Normalize the sum of the k-weights to 1
        total_weight = weights.sum()
        if not (np.isclose(total_weight, 1, atol=1e-8) or np.isclose(total_weight, 2, atol=1e-8)):
            raise ValueError(f"Total weight is {total_weight}, is expected to be 1 or 2.")
        weights /= total_weight

        # Fix k-labels
        if labels is not None and label_numbers is not None:
            labels = list(labels)
            label_numbers = list(label_numbers)
            # If the first k-point does not have a label, give it a placeholder label, and
            # add its index to the indices so that it can start a segment.
            if label_numbers[0] != 0:
                label_numbers.insert(0, 0)
                labels.insert(0, "")
            # Ditto with the last k-point; it needs to end the last segment
            if label_numbers[-1] != kpoints.shape[0] - 1:
                label_numbers.append(kpoints.shape[0] - 1)
                labels.append("")
        else:
            # Set dummy labels and indices corresponding to the first and last k-points
            labels = ["", ""]
            label_numbers = [0, kpoints.shape[0] - 1]

        self.bands = bands
        self.kpoints = kpoints
        self.weights = weights
        self.occupations = occupations
        self.cell = cell
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
    def from_npz_xyz_metadata(
        cls, npz_path: os.PathLike, xyz_path: os.PathLike, json_path: ty.Optional[os.PathLike] = None
    ) -> None:
        """Load a band structure from an NPZ file containing arrays, an extended XYZ file containing a
        periodic structure, and a JSON file containing metadata.

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
        # Load arrays (eigenvalues/bands, k-points, etc.)
        with open(npz_path, "rb") as fp:
            arrays = dict(np.load(fp))
        # Load structure
        with open(xyz_path, "r", encoding="utf-8") as fp:
            atoms = read(fp, index=0)
        # Load metadata (Fermi energy, number of electrons, etc.)
        metadata = {}
        if json_path:
            with open(json_path, "r", encoding="utf-8") as fp:
                metadata = json.load(fp)
        # Ignore occupations and Fermi energy if occupations all zeros
        if np.all(np.isclose(arrays["occupations"], 0)):
            arrays.pop("occupations")
            metadata.pop("fermi_energy", None)
        return cls(
            **arrays,
            cell=atoms.cell.array,
            fermi_energy=metadata.get("fermi_energy"),
            n_electrons=metadata.get("number_of_electrons"),
        )

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

    @property
    def fractional_kpoints(self) -> npt.ArrayLike:
        """Fractional k-points.

        Returns:
            npt.ArrayLike: (n_kpoints, 3) fractional k-points.
        """
        return self.kpoints

    @property
    def cartesian_kpoints(self) -> npt.ArrayLike:
        """Cartesian k-points.

        Returns:
            npt.ArrayLike: (n_kpoints, 3) Cartesian k-points.
        """
        return recip_frac_to_cart(self.kpoints, self.cell)

    @property
    def linear_k(self) -> npt.ArrayLike:
        """Linearized k-points.

        Returns:
            npt.ArrayLike: (n_kpoints, ) vector of linearized k-points.
        """
        # Compute distances between adjacent k-points
        distances = np.linalg.norm(np.diff(self.cartesian_kpoints, axis=0), axis=1)
        # Set distance to zero when adjacent k-points are both labeled (likely a discontinuity)
        mask = np.array([i in self.label_numbers and i - 1 in self.label_numbers for i in range(1, self.n_kpoints)])
        distances[mask] = 0.0
        # Prepend 0 (the linear location of the first k-point)
        linear_k = np.concatenate([[0], np.cumsum(distances)])
        return linear_k

    @property
    def path_segments(self) -> ty.List[BandPathSegment]:
        """Segments of the band structure.

        Returns:
            list[BandPathSegment]: List of band path segments.
        """
        linear_k = self.linear_k
        # Construct the segments
        segments = []
        for i_from, i_to in zip(range(len(self.labels) - 1), range(1, len(self.labels))):
            ik_from = self.label_numbers[i_from]
            ik_to = self.label_numbers[i_to] + 1
            segment = BandPathSegment(
                bands=self.bands[:, ik_from:ik_to].copy(),
                linear_k=linear_k[ik_from:ik_to].copy(),
                fermi_energy=self.fermi_energy if self.fermi_energy else None,
                start_label=self.labels[i_from],
                stop_label=self.labels[i_to],
                start_index=ik_from,
                stop_index=ik_to,
            )
            segments.append(segment)

        return segments

    def compute_occupations(
        self, smearing_type: str, smearing_width: float, fermi_energy: ty.Optional[float] = None
    ) -> npt.ArrayLike:
        """Compute the occupations given a Fermi energy and smearing.

        Args:
            smearing_type (str): Smearing type (see `smearing_from_name`).
            smearing_width (float): Smearing width.
            fermi_energy (ty.Optional[float]): Fermi energy. Defaults to the stored Fermi energy.

        Returns:
            npt.ArrayLike: Occupations array.
        """
        fermi_energy = fermi_energy if fermi_energy is not None else self.fermi_energy
        return fermi.compute_occupations(self.bands, fermi_energy, smearing_type, smearing_width)

    def compute_n_electrons(
        self, smearing_type: str, smearing_width: float, fermi_energy: ty.Optional[float] = None
    ) -> float:
        """Compute the number of electrons from the provided Fermi energy.

        Args:
            smearing_type (str): Smearing type (see `smearing_from_name`).
            smearing_width (float): Smearing width.
            fermi_energy (ty.Optional[float]): Fermi energy. Defaults to the stored Fermi energy.

        Returns:
            float: Fermi energy.
        """
        fermi_energy = fermi_energy if fermi_energy is not None else self.fermi_energy
        return fermi.compute_n_electrons(self.bands, self.weights, fermi_energy, smearing_type, smearing_width)

    def find_fermi_energy(self, smearing_type: str, smearing_width: float, n_electrons_tol: float = 1e-6) -> float:
        """Find a Fermi energy that yields the correct number of electrons.

        Args:
            smearing_type (str): type of smearing (see `smearing_from_name`)
            smearing_width (float): smearing width
            n_electrons_tol (float, optional): tolerance on the number of electrons as a function of the found Fermi
                energy. Defaults to 1e-6.

        Raises:
            ValueError: if the number of electrons is unknown.

        Returns:
            float: Fermi energy
        """
        if self.n_electrons is None:
            raise ValueError("Cannot find the Fermi level if the number of electrons is unknown.")
        return fermi.find_fermi_energy(
            self.bands, self.weights, smearing_type, smearing_width, self.n_electrons, n_electrons_tol
        )

    def compute_smeared_dos(
        self, energies: npt.ArrayLike, smearing_type: ty.Union[str, int], smearing_width: float
    ) -> npt.ArrayLike:
        """Compute a smeared density of states from the band structure (see `smeared_dos`).

        Args:
            energies (npt.ArrayLike): energies at which to sample the DOS
            smearing_type (ty.Union[str,int]): type of smearing (see `smearing_from_name`)
            smearing_width (float): smearing width

        Returns:
            npt.ArrayLike: (n_spins, n_energies) array containing the DOS for each spin channel
        """
        return smeared_dos(energies, self.bands, self.weights, smearing_type, smearing_width)

    def is_metallic(self, tol: float = 1e-6) -> ty.Union[bool, None]:
        """Check if the band structure is metallic.

        Args:
            tol (float, optional): Tolerance on differences in energy. Defaults to 1e-6.

        Returns:
            bool: True if any band has at least one eigenvalue above and below the Fermi energy +/- tol.
        """
        if self.fermi_energy is None:
            return None
        if self.fermi_energy >= np.max(self.bands):
            # All bands are fully-occupied in this case; can't know if the material is a metal or insulator.
            return None

        # Flatten the spin axis so that each row ia a band
        # First reorder the axes to (n_spins, n_bands, n_kpoints)
        # Then reshape the array to (n_spins * n_bands, n_kpoints)
        # The first n_bands rows of the resulting array correspond to spin 0, and the rest to spin 1
        bands = np.transpose(self.bands, (0, 2, 1)).reshape((self.n_spins * self.n_bands, self.n_kpoints))

        return bool(
            np.any(np.any(bands < self.fermi_energy - tol, axis=1) & np.any(bands > self.fermi_energy + tol, axis=1))
        )

    def is_insulating(self, tol: float = 1e-6) -> ty.Optional[bool]:
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
    def vbm(self) -> ty.Optional[float]:
        """The valence band maximum.

        Returns:
            float: valence band maximum.
        """
        if self.fermi_energy is None:
            return None
        if self.fermi_energy >= np.max(self.bands):
            return np.max(self.bands)
        # Mask the conduction bands and find the maximum of the rest of the bands,
        # i.e. the valence bands
        return np.max(self.bands[self.bands <= self.fermi_energy])

    @property
    def vbm_index(self) -> ty.Optional[tuple[int, int, int]]:
        """Cartesian index of the valence band maximum.

        Returns:
            ty.Optional[tuple[int,int,int]]: cartesian index.
        """
        if self.fermi_energy is None:
            return None
        return np.where(np.isclose(self.bands, self.vbm))

    @property
    def cbm(self) -> ty.Optional[float]:
        """The conduction band minimum.

        Returns:
            float: conduction band minimum.
        """
        if self.fermi_energy is None:
            return None
        if self.fermi_energy >= np.max(self.bands):
            return np.max(self.bands)
        # See `vbm`
        return np.min(self.bands[self.bands >= self.fermi_energy])

    @property
    def cbm_index(self) -> ty.Optional[tuple[npt.ArrayLike]]:
        """Cartesian index of the conduction band minimum.

        Returns:
            ty.Optional[tuple[npt.ArrayLike]]: cartesian index.
        """
        if self.fermi_energy is None:
            return None
        return np.where(np.isclose(self.bands, self.cbm))

    @property
    def band_gap(self) -> ty.Optional[float]:
        """The band gap.

        Returns:
            float: band gap.
        """
        if self.vbm is None or self.cbm is None:
            return None
        gap = self.cbm - self.vbm
        if gap < 0:
            return None
        return gap
