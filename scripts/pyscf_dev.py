# %%
import typing as ty

from ase import Atoms
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pyscf.pbc.gto import Cell

import sorep

# %%
with h5py.File("../data/mc3d_data.h5", "r") as f:
    material_ids = list(f.keys())
    material = sorep.MaterialData.from_hdf(f[material_ids[0]]["single_shot"])
# %%
kbs = {}
for basis in ("ano-ml-os", "ano-ml-ae", "sto-3g", "def2-svp"):
    kbs[basis] = sorep.spectra.pyscf.compute_one_electron_spectrum(
        material.atoms, basis, kdensity=0.1, eigen_solver="eigh", use_symmetries=True
    )
# %%
energies = np.linspace(0, 225, 8192)
for i, (k, v) in enumerate(kbs.items()):
    dos = v.compute_smeared_dos(energies, "gauss", 0.05)
    plt.plot(energies, np.sum(dos, axis=0) + i * 25, label=k)
plt.legend()
# %%
