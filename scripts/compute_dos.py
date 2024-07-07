# %%
import pathlib as pl
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import sorep
# %%
SMEARING_TYPE = 'gauss'
SMEARING_WIDTH = 0.05  # eV
N_ENERGIES = 512

def _featurize(dir_):
    material = sorep.MaterialData.from_dir(dir_)

    # material.bands.fermi_energy = material.bands.find_fermi_energy(
    #     material.metadata['smearing'],
    #     material.metadata['degauss'],
    #     n_electrons_tol=1e-4)
    # material.bands.occupations = material.bands.compute_occupations(
    #     material.metadata['smearing'], material.metadata['degauss'])

    energies_fermi_centered = np.linspace(material.bands.fermi_energy - 5, material.bands.fermi_energy + 5, N_ENERGIES)
    dos_fermi_centered = material.bands.compute_smeared_dos(
        energies=energies_fermi_centered,
        smearing_type=SMEARING_TYPE,
        smearing_width=SMEARING_WIDTH
    )

    energies_vbm_centered = np.linspace(material.bands.vbm - 2, material.bands.fermi_energy + 6, N_ENERGIES)
    dos_vbm_centered = material.bands.compute_smeared_dos(
        energies=energies_vbm_centered,
        smearing_type=SMEARING_TYPE,
        smearing_width=SMEARING_WIDTH
    )

    if material.bands.is_insulating() and (material.bands.band_gap > 6 * SMEARING_WIDTH):
        energies_fermi_scissor = np.concatenate([
            np.linspace(material.bands.vbm - 2, material.bands.vbm + 3 * SMEARING_WIDTH, N_ENERGIES // 2),
            np.linspace(material.bands.cbm - 3 * SMEARING_WIDTH, material.bands.cbm + 2, N_ENERGIES // 2)
        ])
    else:
        energies_fermi_scissor = np.linspace(material.bands.vbm - 2, material.bands.cbm + 2, N_ENERGIES)
    dos_fermi_scissor = material.bands.compute_smeared_dos(
        energies=energies_fermi_scissor,
        smearing_type=SMEARING_TYPE,
        smearing_width=SMEARING_WIDTH
    )

    np.savez_compressed(
        dir_ / f'dos_{SMEARING_TYPE}_{SMEARING_WIDTH}.npz',
        energies_fermi_centered=energies_fermi_centered,
        dos_fermi_centered=dos_fermi_centered,
        energies_vbm_centered=energies_vbm_centered,
        dos_vbm_centered=dos_vbm_centered,
        energies_fermi_scissor=energies_fermi_scissor,
        dos_fermi_scissor=dos_fermi_scissor,
        fermi_energy=material.bands.fermi_energy,
        vbm=material.bands.vbm,
        cbm=material.bands.cbm
    )

# %%
def main():
    dirs = list(pl.Path('../data/mc3d/').glob('*/zero_shot')) + list(pl.Path('../data/mc3d/').glob('*/scf'))
    pbar = tqdm(dirs, desc='Compute SOREP features')
    with Pool(processes=12, maxtasksperchild=1) as p:
        p.map(_featurize, pbar)

# %%
if __name__ == '__main__':
    main()
