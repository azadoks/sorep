# %%
import json
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import sorep

# %%
dirs = list(pl.Path('../data/mc3d/').glob('*/scf/*'))
# %%
mismatches = []
with open('mismatches.txt', 'w', buffering=1, encoding='utf-8') as fp:
    fp.write(
        f'{"i":6s} {"dir":50s} {"ef_sorep":>12s} {"ne_sorep":>12s} {"metal_sorep":>12s} {"ef_qe":>12s} {"ne_qe":>12s} {"metal_qe":>12s}\n'
    )
    for (i, dir_) in tqdm(enumerate(dirs)):
        material = sorep.MaterialData.from_dir(dir_)
        ef_qe = material.bands.fermi_energy
        ef_sorep = material.bands.find_fermi_energy(
            material.metadata['smearing_type'],
            material.metadata['degauss'],
            n_electrons_tol=1e-4)
        if np.abs(ef_sorep - ef_qe) > 0.5:
            is_metal_qe = material.bands.is_metallic()
            ne_qe = material.bands.compute_n_electrons(
                material.metadata['smearing_type'],
                material.metadata['degauss'])

            material.bands.fermi_energy = ef_sorep
            is_metal_sorep = material.bands.is_metallic()
            ne_sorep = material.bands.compute_n_electrons(
                material.metadata['smearing_type'],
                material.metadata['degauss'])

            mismatches.append({
                'dir': str(dir_),
                'ef_qe': ef_qe,
                'is_metal_qe': is_metal_qe,
                'n_electrons_qe': ne_qe,
                'ef_sorep': ef_sorep,
                'is_metal_sorep': is_metal_sorep,
                'n_electrons_sorep': ne_sorep
            })
            fp.write(
                f'{i:<6d} {str(dir_):50s} {ef_sorep:12.6f} {ne_sorep:12.6f} {is_metal_sorep:12b} {ef_qe:12.6f} {ne_qe:12.6f} {is_metal_qe:12b}\n'
            )
with open('mismatches.json', 'w', encoding='utf-8') as fp:
    json.dump(mismatches, fp)


# %%
def plot_bands(row):
    material = sorep.MaterialData.from_dir(pl.Path(row['dir']))
    fig, ax = plt.subplots()
    for (i, ls) in zip(range(material.bands.n_spins), ('-', '--')):
        ax.plot(material.bands.bands[i], c='k', linestyle=ls)
    ax.axhline(row['ef_sorep'], c='b', linestyle='-.', label='ef_sorep')
    ax.axhline(row['ef_qe'], c='r', linestyle=':', label='ef_qe')
    ax.set_ylim(row['ef_qe'] + 5, row['ef_qe'] - 5)
    ax.legend()
    return fig, ax


def plot_ne(row):
    material = sorep.MaterialData.from_dir(pl.Path(row['dir']))
    ef_diff = np.abs(row['ef_sorep'] - row['ef_qe'])
    x = np.linspace(row['ef_sorep'] - 3 * ef_diff, row['ef_sorep'] + 3 * ef_diff, 1000)
    y = np.array([
        material.bands.compute_n_electrons(material.metadata['smearing_type'],
                                           material.metadata['degauss'], ef)
        for ef in x
    ]) - material.bands.n_electrons
    fig, ax = plt.subplots()
    ax.plot(x, y, c='k')
    ax.axhline(0, c='grey', linestyle='--', zorder=-1)
    ax.axvline(row['ef_sorep'], c='tab:blue', linestyle='-.', label='ef_sorep')
    ax.axvline(row['ef_qe'], c='tab:red', linestyle=':', label='ef_qe')
    ax.set_xlabel('Chemical potential (eV)')
    ax.set_ylabel('$N_{e^-} - N_{e^-}^{\mathrm{true}}$')
    ax.set_ylim(-1, 1)
    ax.legend()
    return fig, ax

# %%
mismatch_df = pd.read_csv('mismatches.txt', sep='\s+')
double_mismatch_df = mismatch_df[mismatch_df['metal_sorep'] !=
                                 mismatch_df['metal_qe']]

row = mismatch_df.iloc[0]
plot_ne(row)
# %%
