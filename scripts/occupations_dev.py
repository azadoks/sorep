# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

import sorep

# %%
with h5py.File("../data/mc3d/data.h5", "r") as f:
    material = sorep.MaterialData.from_hdf(f["mc3d-10"]["scf"])

# %%
bands = material.bands
smearing_type = "cold" # material.metadata["smearing"]
smearing_width = material.metadata["degauss"] / sorep.constants.RY_TO_EV
# %%
%%time
res = (
    sorep.occupation.compute_n_electrons(
        bands.eigenvalues, bands.weights, bands.fermi_energy, smearing_type, smearing_width
    ),
    sorep.occupation.compute_n_electrons_derivative(
        bands.eigenvalues, bands.weights, bands.fermi_energy, smearing_type, smearing_width
    ),
    sorep.occupation.compute_n_electrons_2nd_derivative(
        bands.eigenvalues, bands.weights, bands.fermi_energy, smearing_type, smearing_width
    ),
)
# %%
%%time
res_py = (
    sorep.occupation.compute_n_electrons_python(
        bands.eigenvalues, bands.weights, bands.fermi_energy, smearing_type, smearing_width
    ),
    sorep.occupation.compute_n_electrons_derivative_python(
        bands.eigenvalues, bands.weights, bands.fermi_energy, smearing_type, smearing_width
    ),
    sorep.occupation.compute_n_electrons_2nd_derivative_python(
        bands.eigenvalues, bands.weights, bands.fermi_energy, smearing_type, smearing_width
    ),
)
# %%
res, res_py
# %%
%%timeit
sorep.fermi.find_fermi_energy_two_stage(
    bands.eigenvalues, bands.weights, smearing_type, smearing_width, material.metadata["number_of_electrons"]
)

# %%
# ef = material.bands.fermi_energy
ef = sorep.fermi.find_fermi_energy_two_stage(
    bands.eigenvalues, bands.weights, smearing_type, smearing_width, material.metadata["number_of_electrons"]
)
energies = np.linspace(ef - 5, ef + 5, 1024)
dos = bands.compute_smeared_dos(energies, "gauss", 0.05)
plt.plot(energies, dos.sum(axis=0))
# %%
