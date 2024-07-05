# %%
from dataclasses import dataclass
import json
import os
import pathlib as pl

# %%
import typing as ty

from ase import Atoms
from ase.io.extxyz import read_extxyz
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp
from tqdm import tqdm

import sorep


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
    bisection_fermi = sorep.fermi.find_fermi_energy_bisection(bands, weights, "gauss", smearing_width, n_electrons)
    # Refine with Newton and the requested smearing (probably cold)
    newton_fermi = bisection_fermi
    newton_refined = True

    def objective(ef):
        ne = sorep.fermi.compute_n_electrons(bands, weights, ef, smearing_type, smearing_width)
        return (ne - n_electrons) ** 2

    def objective_deriv(ef):
        ne = sorep.fermi.compute_n_electrons(bands, weights, ef, smearing_type, smearing_width)
        dne = sorep.fermi.compute_n_electrons_derivative(bands, weights, ef, smearing_type, smearing_width)
        return 2 * (ne - n_electrons) * dne

    def objective_curv(ef):
        ne = sorep.fermi.compute_n_electrons(bands, weights, ef, smearing_type, smearing_width)
        dne = sorep.fermi.compute_n_electrons_derivative(bands, weights, ef, smearing_type, smearing_width)
        cne = sorep.fermi.compute_n_electrons_2nd_derivative(bands, weights, ef, smearing_type, smearing_width)
        return 2 * ((ne - n_electrons) * cne + dne**2)

    try:
        newton_fermi = sp.optimize.newton(
            func=objective, fprime=objective_deriv, fprime2=objective_curv, x0=bisection_fermi, maxiter=50
        )
    except RuntimeError:
        newton_refined = False
    if np.abs(newton_fermi - bisection_fermi) > 0.5:
        newton_refined = False
        newton_fermi = bisection_fermi

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
newton_refineds = []
is_insulatings = []
i = 0
for dir_path in tqdm(list(pl.Path("../data/").glob("*/scf/0/"))):
    material = sorep.MaterialData.from_dir(dir_path)
    # found_fermi = sorep.fermi.find_fermi_energy_bisection(
    found_fermi, newton_refined, bisection_fermi = find_fermi_energy_two_stage(
        material.bands.bands,
        material.bands.weights,
        material.metadata["smearing_type"],
        material.metadata["degauss"],
        material.metadata["number_of_electrons"],
        return_all=True,
    )
    newton_fermis.append(found_fermi)
    bisection_fermis.append(bisection_fermi)
    qe_fermis.append(material.metadata["fermi_energy"])
    n_spins.append(material.bands.n_spins)
    newton_refineds.append(newton_refined)
    is_insulatings.append(material.bands.is_insulating())
    i += 1
# %%
our_fermis = np.array(newton_fermis)
qe_fermis = np.array(qe_fermis)
n_spins = np.array(n_spins)
newton_refineds = np.array(newton_refined)
is_insulatings = np.array(is_insulatings)

ins_filter = True
fig, ax = plt.subplots()
for ref in [True, False]:
    ax.scatter(
        our_fermis[(newton_refineds == ref) & (is_insulatings == ins_filter)],
        qe_fermis[(newton_refineds == ref) & (is_insulatings == ins_filter)],
        alpha=0.5,
        label=f"Newton refined: {ref}",
    )
ax.set_title(f"Insulating: {ins_filter}")
ax.legend()
# %%
ins_filter = True
fig, ax = plt.subplots()
for ref in [True, False]:
    ax.hist(
        our_fermis[(newton_refineds == ref) & (is_insulatings == ins_filter)]
        - qe_fermis[(newton_refineds == ref) & (is_insulatings == ins_filter)],
        alpha=0.5,
        bins=100,
        label=f"Newton refined: {ref}",
    )
ax.set_xlabel("Our Fermi - QE Fermi (eV)")
ax.set_yscale("symlog")
ax.set_title(f"Insulating: {ins_filter}")
ax.legend()
# %%
np.sum(np.abs(our_fermis - qe_fermis) > 0.1)
# %%
