# %%
import pathlib as pl

import h5py

import sorep

# %%
KWARGS = {"compression": "gzip", "shuffle": True}
with h5py.File("../data/mc3d_data.h5", "w") as f:
    for material_dir in pl.Path("../data/mc3d").iterdir():
        material_id = material_dir.name
        for calculation_dir in material_dir.iterdir():
            calculation_type = calculation_dir.name
            material = sorep.MaterialData.from_dir(calculation_dir)

            g = f.create_group(f"{material_id}/{calculation_type}")
            for key, value in material.metadata.items():
                g.attrs[key] = value or ""

            bands_group = g.create_group("bands")
            bands_group.create_dataset("energies", data=material.bands.bands, **KWARGS)
            bands_group["energies"].attrs["units"] = "eV"
            bands_group.create_dataset("kpoints", data=material.bands.fractional_kpoints, **KWARGS)
            bands_group["kpoints"].attrs["units"] = "dimensionless"
            bands_group.create_dataset("weights", data=material.bands.weights, **KWARGS)
            bands_group["weights"].attrs["units"] = "dimensionless"
            bands_group.create_dataset("cell", data=material.bands.cell, **KWARGS)
            bands_group["cell"].attrs["units"] = "angstrom"
            bands_group.create_dataset("occupations", data=material.bands.occupations, **KWARGS)
            bands_group["occupations"].attrs["units"] = "dimensionless"
            bands_group.create_dataset("labels", data=[str(label) for label in material.bands.labels], **KWARGS)
            bands_group.create_dataset("label_numbers", data=material.bands.label_numbers, **KWARGS)
            bands_group.create_dataset("fermi_energy", data=material.bands.fermi_energy)
            bands_group["fermi_energy"].attrs["units"] = "eV"
            bands_group.create_dataset("n_electrons", data=material.bands.fermi_energy)
            atoms_group = g.create_group("atoms")

            for key, value in material.atoms.arrays.items():
                atoms_group.create_dataset(key, data=value, **KWARGS)
            atoms_group["positions"].attrs["units"] = "angstrom"
            if "masses" in atoms_group:
                atoms_group["masses"].attrs["units"] = "amu"
            atoms_group.create_dataset("cell", data=material.atoms.cell, **KWARGS)
            atoms_group["cell"].attrs["units"] = "angstrom"
            atoms_group.create_dataset("pbc", data=material.atoms.pbc, **KWARGS)

# %%
