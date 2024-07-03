# %%
import json
import os
import pathlib as pl
import typing as ty
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.io.extxyz import read_extxyz
from tqdm import tqdm

import sorep
# %%
@dataclass
class MaterialData:
    """Data class containing information about a material as a whole (atomic structure, band structure, and various metadata)."""
    atoms: Atoms
    bands: sorep.BandStructure
    metadata: dict

    @classmethod
    def from_dir(cls, dir_path: os.PathLike) -> 'MaterialData':
        """Load the material data from a directory containing the following files:
        - structure.xyz: ASE-compatible XYZ file containing the atomic structure.
        - bands.npz: NPZ file containing the band structure.
        - metadata.json: JSON file containing metadata.

        Args:
            dir_path (os.PathLike): Path to the directory containing the material data.

        Returns:
            MaterialData: Material data object.
        """
        structure_path = dir_path / 'structure.xyz'
        bands_path = dir_path / 'bands.npz'
        metadata_path = dir_path / 'metadata.json'
        assert structure_path.exists(), f"structure.xyz file not found: {structure_path}"
        assert bands_path.exists(), f"bands.npz file not found: {bands_path}"
        assert metadata_path.exists(), f"metadata.json file not found: {metadata_path}"
        return cls.from_files(structure_path, bands_path, metadata_path)

    @classmethod
    def from_files(cls, structure_path: os.PathLike, bands_path: os.PathLike, metadata_path: os.PathLike) -> 'MaterialData':
        """Load the material data from files containing the following:
        - .xyz: ASE-compatible XYZ file containing the atomic structure.
        - .npz: NPZ file containing the band structure.
        - .json: JSON file containing metadata.

        Args:
            structure_path (os.PathLike): Path to the structure file.
            bands_path (os.PathLike): Path to the bands file.
            metadata_path (os.PathLike): Path to the metadata file.

        Returns:
            MaterialData: Material data object.
        """
        with open(structure_path, 'r', encoding='utf-8') as fp:
            atoms = list(read_extxyz(fp))[0]
        bands = sorep.BandStructure.from_npz_metadata(bands_path, metadata_path)
        with open(metadata_path, 'r', encoding='utf-8') as fp:
            metadata = json.load(fp)
        return cls(atoms, bands, metadata)

# %%
passes = []
fails = []
for dir_path in tqdm(
        list(
            pl.Path('/home/azadoks/Source/git/sorep-npj/data/mc3d/').glob(
                '*/scf/0/'))):
    material = MaterialData.from_dir(dir_path)
    our_occs = material.bands.get_occupations_from_fermi(
        material.metadata['fermi_energy'], material.metadata['smearing_type'],
        material.metadata['degauss'])
    if not np.all(np.isclose(our_occs, material.bands.occupations, rtol=1e-3)):
        fails.append(dir_path)
    else:
        passes.append(dir_path)
# %%
