# %%
import json
import os
import pathlib as pl
import typing as ty
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy as sp
from ase import Atoms
from ase.io.extxyz import read_extxyz
from tqdm import tqdm

import matplotlib.pyplot as plt
#!! N.B. weights sum to 1 with n_spins = 2, sum to 2 with n_spins = 1

# %%
SQRT2 = np.sqrt(2)
INVSQRTPI = 1 / np.sqrt(np.pi)
INVSQRT2 = 1 / np.sqrt(2)
INVSQRT2PI = 1 / np.sqrt(2 * np.pi)
MAX_EXP_ARG = 200.0

class Smearing(ABC):
    """Abstract base class for smearing functions."""
    def __init__(self, center: float, width: float):
        self.center = center
        self.width = width

    def __call__(self, bands: np.array) -> np.array:
        return self.occupation(bands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(center={self.center}, width={self.width})"

    def __str__(self) -> str:
        return self.__repr__()

    def _scale(self, bands: np.array) -> np.array:
        return (self.center - bands) / self.width

    @abstractmethod
    def occupation(self, bands: np.array) -> np.array:
        pass

    @abstractmethod
    def occupation_derivative(self, bands: np.array) -> np.array:
        pass


class NoSmearing(Smearing):
    def occupation(self, bands: np.array) -> np.array:
        return np.where(bands < self.center, 1.0, 0.0)

    def occupation_derivative(self, bands: np.array) -> np.array:
        return np.where(bands == self.center, -np.inf, 0.0)


class GaussianSmearing(Smearing):
    def occupation(self, bands: np.array) -> np.array:
        return 0.5 * sp.special.erfc(-self._scale(bands))

    def occupation_derivative(self, bands: np.array) -> np.array:
        x = self._scale(bands)**2
        return np.where(
            x > MAX_EXP_ARG,
            1.0 / np.sqrt(np.pi) * np.exp(-MAX_EXP_ARG),
            1.0 / np.sqrt(np.pi) * np.exp(-x),
        )


class FermiDiracSmearing(Smearing):
    def occupation(self, bands: np.array) -> np.array:
        x = self._scale(bands)
        return np.where(
            x < -MAX_EXP_ARG,
            0.0,
            np.where(
                x > MAX_EXP_ARG,
                1.0,
                1.0 / (1.0 + np.exp(-x))
            )
        )

    def occupation_derivative(self, bands: np.array) -> np.array:
        x = self._scale(bands)
        return np.where(
            np.abs(x) > MAX_EXP_ARG,
            0.0,
            1.0 / (2 + np.exp(-x) + np.exp(x))
        )


class ColdSmearing(Smearing):
    def occupation(self, bands: np.array) -> np.array:
        x = self._scale(bands)
        return 0.5 * sp.special.erf(x - 1.0 / np.sqrt(2.0)) + 1 / np.sqrt(2.0 * np.pi) * np.exp(-(x - 1.0 / np.sqrt(2.0))**2) + 0.5

    def occupation_derivative(self, bands: np.array) -> np.array:
        x = self._scale(bands) - INVSQRT2
        z = np.minimum(x**2, MAX_EXP_ARG)
        return 1 / (2 * np.sqrt(np.pi)) * np.exp(-z) * (2.0 - np.sqrt(2.0) * x)


def _smearing_from_name(name: ty.Optional[ty.Union[str,int]]) -> Smearing:
    if name is None:
        return NoSmearing
    name = str(name).lower()
    if name in ('mv', 'm-v', 'marzari-vanderbilt', 'cold', '-1'):
        return ColdSmearing
    elif name in ('gauss', 'gaussian', '0'):
        return GaussianSmearing
    elif name in ('fd', 'f-d', 'fermi-dirac', '-99'):
        return FermiDiracSmearing
    raise ValueError(f'Unknown smearing function name {name}')
# %%
def smeared_dos(
    energies: np.array,
    bands: np.array,
    weights: np.array,
    smearing_type: ty.Union[str,int],
    smearing_width: float
) -> np.array:
    smearing_cls = _smearing_from_name(smearing_type)
    smearing = smearing_cls(0.0, smearing_width)

    dos = np.zeros((energies.shape[0], bands.shape[0]))
    for (i, energy) in enumerate(energies):
        smearing.center = energy
        occ_deriv = smearing.occupation_derivative(bands)
        dos[i] = np.einsum('skn,k->s', occ_deriv, weights)

    return (dos / smearing_width).T
# %%
@dataclass
class BandStructure:
    """Data class containing band structure information."""
    bands: np.array
    kpoints: np.array
    weights: np.array
    occupations: ty.Optional[np.array] = None
    labels: ty.Optional[np.array] = None
    label_numbers: ty.Optional[np.array] = None
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
        """Maximum occupation.

        Returns:
            int: maximum occupation
        """
        if self.n_spins == 1:
            return 2
        elif self.n_spins == 2:
            return 1
        else:
            raise ValueError(f'Unknown maximum occupation for n_spins={self.n_spins}')


    def get_occupations_from_fermi(self, fermi_energy: float,
                                   smearing_type: str,
                                   smearing_width: float) -> None:
        if self.n_electrons is None:
            raise ValueError(
                "Cannot compute occupations if number of electrons is unknown."
            )
        smearing = _smearing_from_name(smearing_type)(center=fermi_energy,
                                                      width=smearing_width)
        return self.max_occupation * smearing(self.bands)

    def get_smeared_dos(self, energies: np.array, smearing_type:ty.Union[str,int], smearing_width: float) -> np.array:
        return smeared_dos(energies, self.bands, self.weights, smearing_type, smearing_width)


@dataclass
class MaterialData:
    """Data class containing information about a material as a whole (atomic structure, band structure, and various metadata)."""
    atoms: Atoms
    bands: BandStructure
    metadata: dict

    @classmethod
    def from_dir(cls, dir_path: os.PathLike) -> None:
        with open(dir_path / 'structure.xyz', 'r', encoding='utf-8') as fp:
            atoms = list(read_extxyz(fp))[0]
        bands = BandStructure.from_npz_metadata(dir_path / 'bands.npz',
                                                dir_path / 'metadata.json')
        with open(dir_path / 'metadata.json', 'r', encoding='utf-8') as fp:
            metadata = json.load(fp)
        return cls(atoms, bands, metadata)



# %%
passes = []
fails = []
for dir_path in tqdm(list(pl.Path('/home/azadoks/Source/git/sorep-npj/data/mc3d/').glob('*/scf/0/'))):
    material = MaterialData.from_dir(dir_path)
    our_occs = material.bands.get_occupations_from_fermi(
        material.metadata['fermi_energy'],
        material.metadata['smearing_type'],
        material.metadata['degauss']
    )
    if not np.all(
        np.isclose(
            our_occs,
            material.bands.occupations,
            rtol=1e-3
        )
    ):
        fails.append(dir_path)
    else:
        passes.append(dir_path)
# %%
