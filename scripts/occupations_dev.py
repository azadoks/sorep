# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

import sorep

# %%
ef1 = []
ef2 = []
diff_metal = []
with h5py.File("../data/mc3d/materials.h5", "r") as f:
    for id_, mat_g in f.items():
        material = sorep.MaterialData.from_hdf(mat_g["single_shot"])
        ef_saved = material.bands.fermi_energy
        is_met_saved = material.bands.is_metallic()
        ef_found = material.bands.find_fermi_energy(
            material.metadata["bands"]["smearing"], material.metadata["bands"]["degauss"]
        )
        material.bands.fermi_energy = ef_found
        is_met_found = material.bands.is_metallic()
        material.bands.fermi_energy = ef_saved

        ef1.append(ef_saved)
        ef2.append(ef_found)
        diff_metal.append(is_met_saved != is_met_found)
# %%
ef1 = np.array(ef1)
ef2 = np.array(ef2)
diff_metal = np.array(diff_metal)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(ef1[diff_metal], ef2[diff_metal])
ax[0].scatter(ef1[~diff_metal], ef2[~diff_metal], alpha=0.01)
ax[1].hist(ef1 - ef2, bins=100)
ax[1].set_yscale("log")
# %%
