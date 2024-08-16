# %%
from functools import partial
from multiprocessing import Pool
import pathlib as pl

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from tqdm import tqdm

import sorep

# %%
DATA_DIR = pl.Path("../data/")
DATABASE = "batio3"
CALCULATION_TYPE = "single_shot"

KIN_BANDS_PARAMS = {
    "basis": "ano-ml-os",
    "operator": "int1e_kin",
    "kdensity": 0.5,  # 1/angstrom
    "eigen_solver": "eigh",
    "use_symmetries": True,
}

KIN_DOS_PARAMS = {
    "center": 0.0,
    "e_min": 0.0,
    "e_max": 180.0,
    "n_energies": 1024,
    "smearing_type": "gauss",
    "smearing_width": 0.8,
}
# %%
with h5py.File(DATA_DIR / DATABASE / "materials.h5", "r") as f:
    materials = [sorep.MaterialData.from_hdf(g[CALCULATION_TYPE]) for g in f.values()]

# %%
kin_bands = []
for material in tqdm(materials):
    bands = sorep.spectra.pyscf.compute_one_electron_spectrum(material.atoms, **KIN_BANDS_PARAMS)
    kin_bands.append(bands)

with h5py.File(DATA_DIR / DATABASE / "kinetic_bands.h5", "w") as f:
    for material, bands in zip(materials, kin_bands):
        id_ = material.metadata["atoms"]["id"]
        g = f[id_] if id_ in f else f.create_group(id_)
        for key, value in KIN_BANDS_PARAMS.items():
            g.attrs[key] = value
        bands.to_hdf(g)
# %%
fig, ax = plt.subplots()
ax.ecdf(np.concatenate([np.ravel(bands.eigenvalues) for bands in kin_bands]))
ax.set_xlim(0, 360)
# ax.set_xscale("log")
# %%
kin_dos = []
for bands in tqdm(kin_bands):
    dos = sorep.features._dos_featurize(bands, **KIN_DOS_PARAMS)
    kin_dos.append(dos)

# with h5py.File(DATA_DIR / DATABASE / "features.h5", "a") as f:
#     g = f["kinetic/0"] if "kinetic/0" in f else f.create_group("kinetic/0")
#     for key, value in KIN_DOS_PARAMS.items():
#         g.attrs[key] = value
#     g.create_dataset(
#         "id",
#         data=[str(material.metadata["atoms"]["id"]) for material in materials],
#         dtype=h5py.string_dtype(),
#     )
#     g.create_dataset("features", data=kin_dos, compression="gzip", shuffle=True)

prototype = [material.atoms.info["prototype_name"] for material in materials]
# %%
energies = np.linspace(KIN_DOS_PARAMS["e_min"], KIN_DOS_PARAMS["e_max"], KIN_DOS_PARAMS["n_energies"])
unique_protos = list(set(prototype))
colors = [unique_protos.index(proto) for proto in prototype]

fig, ax = plt.subplots()
for dos, color in zip(kin_dos, colors):
    ax.plot(energies, dos + color * 6, color=plt.cm.tab10(color))
# %%
