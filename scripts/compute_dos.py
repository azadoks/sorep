# %%
from multiprocessing import Pool
import pathlib as pl

import numpy as np
from tqdm import tqdm

import sorep

# %%
SMEARING_TYPE = "gauss"
SMEARING_WIDTH = 0.05  # eV
N_ENERGIES = 512


def _featurize_fermi_centered(dir_, material, e_min, e_max):
    fermi_energy = material.bands.fermi_energy
    energies = np.linspace(fermi_energy + e_min, fermi_energy + e_max, N_ENERGIES)
    dos = material.bands.compute_smeared_dos(
        energies=energies, smearing_type=SMEARING_TYPE, smearing_width=SMEARING_WIDTH
    )
    feature_id = f"{e_min:0.2f}_{e_max:0.2f}_{N_ENERGIES}"
    np.savez_compressed(
        dir_ / f"dos_fermi_centered_{SMEARING_TYPE}_{SMEARING_WIDTH}_{feature_id}.npz",
        energies=energies,
        dos=dos,
        fermi_energy=fermi_energy,
        vbm=material.bands.vbm,
        cbm=material.bands.cbm,
    )


def _featurize_vbm_centered(dir_, material, e_min, e_max):
    if material.bands.is_insulating():
        vbm = material.bands.vbm
    else:
        vbm = material.bands.fermi_energy
    energies = np.linspace(vbm + e_min, vbm + e_max, N_ENERGIES)
    dos = material.bands.compute_smeared_dos(
        energies=energies, smearing_type=SMEARING_TYPE, smearing_width=SMEARING_WIDTH
    )
    feature_id = f"{e_min:0.2f}_{e_max:0.2f}_{N_ENERGIES}"
    np.savez_compressed(
        dir_ / f"dos_vbm_centered_{SMEARING_TYPE}_{SMEARING_WIDTH}_{feature_id}.npz",
        energies=energies,
        dos=dos,
        fermi_energy=material.bands.fermi_energy,
        vbm=vbm,
        cbm=material.bands.cbm,
    )


def _featurize_cbm_centered(dir_, material, e_min, e_max):
    if material.bands.is_insulating():
        cbm = material.bands.cbm
    else:
        cbm = material.bands.fermi_energy
    energies = np.linspace(cbm + e_min, cbm + e_max, N_ENERGIES)
    dos = material.bands.compute_smeared_dos(
        energies=energies, smearing_type=SMEARING_TYPE, smearing_width=SMEARING_WIDTH
    )
    feature_id = f"{e_min:0.2f}_{e_max:0.2f}_{N_ENERGIES}"
    np.savez_compressed(
        dir_ / f"dos_cbm_centered_{SMEARING_TYPE}_{SMEARING_WIDTH}_{feature_id}.npz",
        energies=energies,
        dos=dos,
        fermi_energy=material.bands.fermi_energy,
        vbm=material.bands.vbm,
        cbm=cbm,
    )


def _featurize_fermi_scissor(dir_, material, vbm_e_lims, cbm_e_lims):
    if material.bands.is_insulating():
        vbm = material.bands.vbm
        cbm = material.bands.cbm
    else:
        vbm = material.bands.fermi_energy
        cbm = material.bands.fermi_energy
    energies = np.concatenate(
        [
            np.linspace(vbm + vbm_e_lims[0], vbm + vbm_e_lims[1], N_ENERGIES // 2),
            np.linspace(cbm + cbm_e_lims[0], cbm + cbm_e_lims[1], N_ENERGIES // 2),
        ]
    )
    dos = material.bands.compute_smeared_dos(
        energies=energies, smearing_type=SMEARING_TYPE, smearing_width=SMEARING_WIDTH
    )
    feature_id = f"{vbm_e_lims[0]:0.2f}_{vbm_e_lims[1]:0.2f}_{cbm_e_lims[0]:0.2f}_{cbm_e_lims[1]:0.2f}_{N_ENERGIES}"
    np.savez_compressed(
        dir_ / f"dos_fermi_scissor_{SMEARING_TYPE}_{SMEARING_WIDTH}_{feature_id}.npz",
        energies=energies,
        dos=dos,
        fermi_energy=material.bands.fermi_energy,
        vbm=vbm,
        cbm=cbm,
    )


def _featurize_vfc_concat(dir_, material, vbm_e_lims, fermi_e_lims, cbm_e_lims):
    fermi_energy = material.bands.fermi_energy
    if material.bands.is_insulating():
        vbm = material.bands.vbm
        cbm = material.bands.cbm
    else:
        vbm = fermi_energy
        cbm = fermi_energy
    energies = np.concatenate(
        [
            np.linspace(vbm + vbm_e_lims[0], vbm + vbm_e_lims[1], N_ENERGIES // 3),
            np.linspace(
                fermi_energy + fermi_e_lims[0],
                fermi_energy + fermi_e_lims[1],
                N_ENERGIES // 3,
            ),
            np.linspace(cbm + cbm_e_lims[0], cbm + cbm_e_lims[1], N_ENERGIES // 3),
        ]
    )
    dos = material.bands.compute_smeared_dos(
        energies=energies, smearing_type=SMEARING_TYPE, smearing_width=SMEARING_WIDTH
    )
    feature_id = f"{vbm_e_lims[0]:0.2f}_{vbm_e_lims[1]:0.2f}_{fermi_e_lims[0]:0.2f}_{fermi_e_lims[1]:0.2f}_{cbm_e_lims[0]:0.2f}_{cbm_e_lims[1]:0.2f}_{N_ENERGIES}"
    np.savez_compressed(
        dir_ / f"dos_vfc_concat_{SMEARING_TYPE}_{SMEARING_WIDTH}_{feature_id}.npz",
        energies=energies,
        dos=dos,
        fermi_energy=fermi_energy,
        vbm=vbm,
        cbm=cbm,
    )


def _featurize_separate_files(dir_):
    material = sorep.MaterialData.from_dir(dir_)
    # _featurize_fermi_centered(dir_, material, -5, +5)
    # _featurize_vbm_centered(dir_, material, -2, +6)
    # _featurize_cbm_centered(dir_, material, -6, +2)
    # _featurize_fermi_scissor(dir_, material, (-2, +3 * SMEARING_WIDTH), (-3 * SMEARING_WIDTH, +2))
    # _featurize_fermi_scissor(dir_, material, (-2, +2), (-2, +2))
    _featurize_vfc_concat(dir_, material, (-1, +1), (-1, +1), (-1, +1))


# %%
def main():
    dirs = list(pl.Path("../data/mc3d/").glob("*/single_shot")) + list(pl.Path("../data/mc3d/").glob("*/scf"))
    pbar = tqdm(dirs, desc="Compute SOREP features", ncols=80)
    with Pool(processes=12, maxtasksperchild=1) as p:
        p.map(_featurize_separate_files, pbar)


# %%

if __name__ == "__main__":
    main()
