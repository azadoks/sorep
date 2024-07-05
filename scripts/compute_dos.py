# %%
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sorep

# %%
scf_dirs = pl.Path('../data/mc3d/known_materials/').glob('*/scf/*')
zero_shot_dirs = pl.Path('../data/mc3d/known_materials/').glob('*/zero_shot/*')
# %%
SMEARING_TYPE = 'gauss'
SMEARING_WIDTH = 0.05  # eV
E_MIN = -5
E_MAX = 5

for (name, dirs) in {
        'scf': list(scf_dirs),
        'zero-shot': zero_shot_dirs,
}.items():
    print('#' * 10 + ' ' + name.upper() + ' ' + '#' * 10)
    for dir_ in tqdm(dirs):
        material = sorep.MaterialData.from_dir(dir_)
        material.bands.fermi_energy = material.bands.find_fermi_energy(
            material.metadata['smearing_type'],
            material.metadata['degauss'],
            n_electrons_tol=1e-4)
        material.bands.occupations = material.bands.compute_occupations(
            material.metadata['smearing_type'], material.metadata['degauss'])

        energies_fermi_centered = np.linspace(material.bands.fermi_energy - 5, material.bands.fermi_energy + 5, 512)
        dos_fermi_centered = material.bands.compute_smeared_dos(
            energies=energies_fermi_centered,
            smearing_type=SMEARING_TYPE,
            smearing_width=SMEARING_WIDTH
        )

        energies_vbm_centered = np.linspace(material.bands.vbm - 2, material.bands.fermi_energy + 6, 512)
        dos_vbm_centered = material.bands.compute_smeared_dos(
            energies=energies_vbm_centered,
            smearing_type=SMEARING_TYPE,
            smearing_width=SMEARING_WIDTH
        )

        energies_fermi_scissor = np.concatenate([
            np.linspace(material.bands.vbm - 2, material.bands.vbm + 3 * SMEARING_WIDTH, 256),
            np.linspace(material.bands.cbm - 3 * SMEARING_WIDTH, material.bands.cbm + 2, 256)
        ])
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
fig, ax = plt.subplots()
ax.axvline(material.bands.fermi_energy, c='k', ls='--', label='Fermi energy')
ax.plot(energies_fermi_centered, dos_fermi_centered.sum(axis=0), label='Fermi centered')
ax.plot(energies_vbm_centered, dos_vbm_centered.sum(axis=0), label='VBM centered')
# ax.plot(dos_fermi_scissor.sum(axis=0), label='Fermi scissor')

ax.legend()
# %%
