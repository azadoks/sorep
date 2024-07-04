# %%
import pathlib as pl

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

import sorep

jax.config.update('jax_enable_x64', True)
# %%
dirs = list(pl.Path('../data/mc3d/').glob('*/scf/*'))
# %%
%%time
material = sorep.MaterialData.from_dir(pl.Path('../test/resources/one_spin_insulator/'))

sorep.fermi.find_fermi_energy(
    material.bands.bands,
    material.bands.weights,
    material.metadata['smearing_type'],
    material.metadata['degauss'],
    material.bands.n_electrons,
    n_electrons_tol=1e-4
), material.bands.fermi_energy, material.bands.is_insulating(), material.bands.vbm, material.bands.cbm

# %%
