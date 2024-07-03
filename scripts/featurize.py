# %%
import json
import os
import pathlib as pl
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.io.extxyz import read_extxyz
from tqdm import tqdm
import matplotlib.pyplot as plt

import sorep


# %%
@dataclass
class MaterialData:
    """Data class containing information about a material as a whole (atomic structure, band structure, and various metadata)."""
    atoms: Atoms
    bands: sorep.band_structure.BandStructure
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
        assert structure_path.exists(
        ), f"structure.xyz file not found: {structure_path}"
        assert bands_path.exists(), f"bands.npz file not found: {bands_path}"
        assert metadata_path.exists(
        ), f"metadata.json file not found: {metadata_path}"
        return cls.from_files(structure_path, bands_path, metadata_path)

    @classmethod
    def from_files(cls, structure_path: os.PathLike, bands_path: os.PathLike,
                   metadata_path: os.PathLike) -> 'MaterialData':
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
        bands = sorep.band_structure.BandStructure.from_npz_metadata(
            bands_path, metadata_path)
        with open(metadata_path, 'r', encoding='utf-8') as fp:
            metadata = json.load(fp)
        return cls(atoms, bands, metadata)


# %%
import typing as ty
import scipy as sp
import numpy.typing as npt


def find_fermi_energy_two_stage(
    bands: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    smearing_type: ty.Union[str, int],
    smearing_width: float,
    n_electrons: int,
    return_all: bool = False,
) -> float:
    """Find the Fermi level using a two-stage algorithm which starts from a bisection with Gaussian
    smearing and follows up with a Newton refinement with the requested smearing.

    Adapated from DFTK.jl/src/occupation.jl.

    Args:
        bands (npt.NDArray[np.float64]): (n_spins, n_kpoints, n_bands) eigenvalues/bands array.
        weights (npt.NDArray[np.float64]): (n_kpoints, ) k-point weights array.
        smearing_type (ty.Union[str, int]): type of smearing (see `smearing_from_name`).
        smearing_width (float): smearing width
        n_electrons (int): target number of electrons
    """
    # Start with bisection and Gaussian smearing
    bisection_fermi = sorep.fermi.find_fermi_energy_bisection(
        bands, weights, 'gauss', smearing_width, n_electrons)
    # Refine with Newton and the requested smearing (probably cold)
    newton_fermi = bisection_fermi
    newton_refined = True


    def objective(ef):
        ne = sorep.fermi._compute_n_electrons(
            bands,
            weights,
            ef,
            smearing_type,
            smearing_width,
        )
        return (ne - n_electrons)**2

    def objective_deriv(ef):
        ne = sorep.fermi._compute_n_electrons(
            bands,
            weights,
            ef,
            smearing_type,
            smearing_width,
        )
        dne = sorep.fermi._compute_n_electrons_derivative(
            bands, weights, ef, smearing_type, smearing_width)
        return 2 * (ne - n_electrons) * dne

    def objective_curv(ef):
        ne = sorep.fermi._compute_n_electrons(
            bands,
            weights,
            ef,
            smearing_type,
            smearing_width,
        )
        dne = sorep.fermi._compute_n_electrons_derivative(
            bands, weights, ef, smearing_type, smearing_width)
        cne = sorep.fermi._compute_n_electrons_curvature(
            bands, weights, ef, smearing_type, smearing_width)
        return 2 * ((ne - n_electrons) * cne + dne**2)

    try:
        newton_fermi = sp.optimize.newton(func=objective,
                                          fprime=objective_deriv,
                                          fprime2=objective_curv,
                                          x0=bisection_fermi,
                                          maxiter=50)
    except RuntimeError:
        newton_refined = False
    if return_all:
        return newton_fermi, newton_refined, bisection_fermi
    return newton_fermi


# %%
# passes = []
# fails = []
# for dir_path in tqdm(
#         list(
#             pl.Path('../data/').glob('*/scf/0/'))):
#     material = MaterialData.from_dir(dir_path)
#     our_occs = material.bands.compute_occupations_from_fermi(
#         material.metadata['fermi_energy'], material.metadata['smearing_type'],
#         material.metadata['degauss'])
#     if not np.all(np.isclose(our_occs, material.bands.occupations, rtol=1e-3)):
#         fails.append(dir_path)
#     else:
#         passes.append(dir_path)
# %%
newton_fermis = []
bisection_fermis = []
qe_fermis = []
n_spins = []
i = 0
for dir_path in tqdm(list(pl.Path('../data/').glob('*/scf/0/'))):
    material = MaterialData.from_dir(dir_path)
    # found_fermi = sorep.fermi.find_fermi_energy_bisection(
    found_fermi, newton_refined, bisection_fermi = find_fermi_energy_two_stage(
        material.bands.bands,
        material.bands.weights,
        material.metadata['smearing_type'],
        material.metadata['degauss'],
        material.metadata['number_of_electrons'],
        return_all=True
    )
    newton_fermis.append(found_fermi)
    bisection_fermis.append(bisection_fermi)
    qe_fermis.append(material.metadata['fermi_energy'])
    n_spins.append(material.bands.n_spins)
    i += 1
    if i == 1000:
        break
# %%
ne = sorep.fermi._compute_n_electrons(material.bands.bands,
                                      material.bands.weights, found_fermi,
                                      material.metadata['smearing_type'],
                                      material.metadata['degauss'])

x = np.linspace(found_fermi - 0.1, found_fermi + 0.1, 100)
y = [
    sorep.fermi._compute_n_electrons(material.bands.bands,
                                     material.bands.weights, ef,
                                     material.metadata['smearing_type'],
                                     material.metadata['degauss']) for ef in x
]

dne = sorep.fermi._compute_n_electrons_derivative(
    material.bands.bands, material.bands.weights, found_fermi,
    material.metadata['smearing_type'], material.metadata['degauss'])
cne = sorep.fermi._compute_n_electrons_curvature(
    material.bands.bands, material.bands.weights, found_fermi,
    material.metadata['smearing_type'], material.metadata['degauss'])
# %%
our_fermis = np.array(newton_fermis)
qe_fermis = np.array(qe_fermis)
n_spins = np.array(n_spins)

fig, ax = plt.subplots()
for n in (1, 2):
    ax.scatter(our_fermis[n_spins == n],
               qe_fermis[n_spins == n],
               alpha=0.5,
               label=f'{n} spins')
ax.set_xlim(-20, 40)
ax.set_ylim(-20, 40)
ax.legend()
# %%
