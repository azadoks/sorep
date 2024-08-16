# %%
import json

from aiida import load_profile, orm, plugins
import h5py
import numpy as np

import sorep
from sorep.constants import RY_TO_EV

PwBaseWorkChain = plugins.WorkflowFactory("quantumespresso.pw.base")
PwBandsWorkChain = plugins.WorkflowFactory("quantumespresso.pw.bands")


load_profile("batio3")
# %%
STRUCTURE_GROUP_LABEL = "BaTiO3/structures/optimized"
PROTOTYPICAL_STRUCTURES = [
    {"id": "mpds|1.0.0|S1638131", "group": "Layered perovskite 1"},
    {"id": "mpds|1.0.0|S1129458", "group": "Rhombohedral"},
    {"id": "icsd|2017.2|109327", "group": "Supertetragonal non-perovskite"},
    {"id": "icsd|2017.2|100804", "group": "Tetragonal"},
    {"id": "mpds|1.0.0|S1638130", "group": "Layered perovskite 2"},
    {"id": "mpds|1.0.0|S1823200", "group": "Cubic"},
    {"id": "cod|176429|1542189", "group": "Cubic Ba/Ti swapped"},
    {"id": "mpds|1.0.0|S1129461", "group": "Orthorhombic"},
    {"id": "icsd|2017.2|154346", "group": "Orthorhombic non-perovskite"},
]
PROTOTYPICAL_SOURCE_IDS = {v["id"]: v["group"] for v in PROTOTYPICAL_STRUCTURES}

with open("../../data/mc3d/util/source_to_mc3d_id.json", "r", encoding="utf-8") as f:
    SOURCE_TO_MC3D = json.load(f)

INPUT_PROJECT = ["attributes.SYSTEM.occupations", "attributes.SYSTEM.smearing", "attributes.SYSTEM.degauss"]
OUTPUT_PROJECT = [
    "attributes.wfc_cutoff",
    "attributes.wfc_cutoff_units",
    "attributes.rho_cutoff",
    "attributes.rho_cutoff_units",
    "attributes.volume",
    "attributes.number_of_species",
    "attributes.number_of_atoms",
    "attributes.number_of_electrons",
    "attributes.number_of_spin_components",
    "attributes.number_of_bands",
    "attributes.number_of_k_points",
    "attributes.monkhorst_pack_grid",
    "attributes.monkhorst_pack_offset",
    "attributes.lsda",
    "attributes.do_magnetization",
    "attributes.fermi_energy",
    "attributes.fermi_energy_units",
    "attributes.wall_time_seconds",
    "attributes.dft_exchange_correlation",
    "attributes.creator_version",
]

HDF_ARRAY_KWARGS = {"compression": "gzip", "shuffle": True}


# %%
def _query():
    qb = orm.QueryBuilder()
    qb.append(orm.Group, filters={"label": STRUCTURE_GROUP_LABEL}, tag="group")
    qb.append(
        orm.StructureData,
        with_group="group",
        project=[
            "*",
            "uuid",
            "extras.source.id",
            "extras.source.database",
            "extras.source.version",
            "extras.duplicates",
        ],
        tag="structure",
    )

    qb.append(PwBaseWorkChain, with_incoming="structure", project=["*", "uuid", "ctime"], tag="single_shot")
    qb.append(
        orm.Dict,
        with_outgoing="single_shot",
        edge_filters={"label": "pw__parameters"},
        filters={"attributes.ELECTRONS.electron_maxstep": 0},
        project=INPUT_PROJECT,
        tag="single_shot_input",
    )
    qb.append(
        orm.Dict,
        with_incoming="single_shot",
        edge_filters={"label": "output_parameters"},
        project=OUTPUT_PROJECT,
        tag="single_shot_output",
    )
    qb.append(
        orm.BandsData,
        with_incoming="single_shot",
        project=["*", "uuid", "attributes.labels", "attributes.label_numbers"],
        tag="single_shot_bands",
    )

    # qb.append(PwBaseWorkChain, with_incoming="structure", project=["*", "uuid", "ctime"], tag="scf")
    # qb.append(
    #     orm.Dict,
    #     with_outgoing="scf",
    #     edge_filters={"label": "pw__parameters"},
    #     filters={"attributes.ELECTRONS.electron_maxstep": {">": 0}},
    #     project=INPUT_PROJECT,
    #     tag="scf_input",
    # )
    # qb.append(
    #     orm.Dict,
    #     with_incoming="scf",
    #     edge_filters={"label": "output_parameters"},
    #     project=OUTPUT_PROJECT,
    #     tag="scf_output",
    # )
    # qb.append(
    #     orm.BandsData,
    #     with_incoming="scf",
    #     project=["*", "uuid", "attributes.labels", "attributes.label_numbers"],
    #     tag="scf_bands",
    # )
    return qb.dict()


# %%
def _get_structure_metadata(res, id_to_proto_id):
    ss_out = res["single_shot_output"]
    structure = res["structure"]
    id_ = "|".join(
        [structure["extras.source.database"], structure["extras.source.version"], structure["extras.source.id"]]
    )
    return {
        "uuid": structure["uuid"],
        "id": id_,
        "prototype_id": id_to_proto_id[id_],
        "prototype_name": PROTOTYPICAL_SOURCE_IDS[id_to_proto_id[id_]],
        "source_database": structure["extras.source.database"],
        "source_version": structure["extras.source.version"],
        "source_id": structure["extras.source.id"],
        "volume": ss_out["attributes.volume"],
        "volume_units": "angstrom^3",
        "number_of_species": ss_out["attributes.number_of_species"],
        "number_of_atoms": ss_out["attributes.number_of_atoms"],
    }


def _get_calculation_metadata(res):
    ss_calc = res["single_shot"]
    ss_out = res["single_shot_output"]
    return {
        "uuid": ss_calc["uuid"],
        "ctime": str(ss_calc["ctime"]),
        "wall_time_seconds": (
            -1.0 if ss_out["attributes.wall_time_seconds"] is None else ss_out["attributes.wall_time_seconds"]
        ),
        "creator_version": ss_out["attributes.creator_version"],
        "dft_exchange_correlation": ss_out["attributes.dft_exchange_correlation"],
        "wfc_cutoff": ss_out["attributes.wfc_cutoff"],
        "wfc_cutoff_units": ss_out["attributes.wfc_cutoff_units"],
        "rho_cutoff": ss_out["attributes.rho_cutoff"],
        "rho_cutoff_units": ss_out["attributes.rho_cutoff_units"],
    }


def _get_bands_metadata(res):
    ss_in = res["single_shot_input"]
    ss_out = res["single_shot_output"]
    ss_bands = res["single_shot_bands"]
    return {
        "uuid": ss_bands["uuid"],
        "qe_fermi_energy": ss_out["attributes.fermi_energy"],
        "qe_fermi_energy_units": ss_out["attributes.fermi_energy_units"],
        "number_of_electrons": ss_out["attributes.number_of_electrons"],
        "number_of_bands": ss_out["attributes.number_of_bands"],
        "number_of_k_points": ss_out["attributes.number_of_k_points"],
        "number_of_spin_components": ss_out["attributes.number_of_spin_components"],
        "monkhorst_pack_grid": (
            [] if ss_out["attributes.monkhorst_pack_grid"] is None else ss_out["attributes.monkhorst_pack_grid"]
        ),
        "monkhorst_pack_offset": (
            [] if ss_out["attributes.monkhorst_pack_offset"] is None else ss_out["attributes.monkhorst_pack_offset"]
        ),
        "smearing": ss_in["attributes.SYSTEM.smearing"],
        "occupations": ss_in["attributes.SYSTEM.occupations"],
        "degauss": (
            0.0 if ss_in["attributes.SYSTEM.degauss"] is None else ss_in["attributes.SYSTEM.degauss"] * RY_TO_EV
        ),
        "degauss_units": "eV",
    }


def _get_atoms_arrays(res):
    atoms = res["structure"]["*"].get_ase()
    return {
        "positions": {"data": atoms.arrays["positions"], "attrs": {"units": "angstrom"}, **HDF_ARRAY_KWARGS},
        "numbers": {"data": atoms.arrays["numbers"], **HDF_ARRAY_KWARGS},
        "masses": {"data": atoms.arrays["masses"], "attrs": {"units": "amu"}, **HDF_ARRAY_KWARGS},
        "cell": {"data": np.array(atoms.cell), "attrs": {"units": "angstrom"}, **HDF_ARRAY_KWARGS},
        "pbc": {"data": atoms.pbc},
    }


def _get_bands_arrays(res):
    ss_bands = res["single_shot_bands"]
    metadata = _get_bands_metadata(res)

    bands_arrays = {
        "eigenvalues": {"data": ss_bands["*"].get_array("bands"), "attrs": {"units": "eV"}, **HDF_ARRAY_KWARGS},
        "kpoints": {
            "data": ss_bands["*"].get_array("kpoints"),
            "attrs": {"units": "dimensionless"},
            **HDF_ARRAY_KWARGS,
        },
        "weights": {"data": ss_bands["*"].get_array("weights"), **HDF_ARRAY_KWARGS},
        "cell": {"data": np.array(res["structure"]["*"].cell), "attrs": {"units": "angstrom"}, **HDF_ARRAY_KWARGS},
        "labels": {
            "data": ss_bands.get("attributes.labels", []),
            "dtype": h5py.string_dtype(encoding="utf-8", length=None),
        },
        "label_numbers": {"data": np.array(ss_bands.get("attributes.label_numbers", []), dtype=int)},
        "n_electrons": {"data": metadata["number_of_electrons"]},
        # Completely ignore QE Fermi energy and occupations -- often garbage or missing (<=v6.8 w/ cold or single_shot)
        # "fermi_energy": metadata["qe_fermi_energy"],
        # "occupations": ss_bands["*"].get_array("occupations"),
    }
    # Add a spin dimension if missing
    bands_arrays["eigenvalues"]["data"] = (
        bands_arrays["eigenvalues"]["data"]
        if bands_arrays["eigenvalues"]["data"].ndim == 3
        else np.expand_dims(bands_arrays["eigenvalues"]["data"], 0)
    )

    bandstructure = sorep.BandStructure(**{k: v["data"] for k, v in bands_arrays.items()})
    fermi_energy = bandstructure.find_fermi_energy(metadata["smearing"], metadata["degauss"], n_electrons_tol=1e-4)
    # Move the Fermi energy to mid-gap if insulating
    bandstructure.fermi_energy = fermi_energy
    if bandstructure.is_insulating():
        fermi_energy = bandstructure.vbm + (bandstructure.cbm - bandstructure.vbm) / 2
    occupations = bandstructure.compute_occupations(metadata["smearing"], metadata["degauss"], fermi_energy)
    bands_arrays["fermi_energy"] = {"data": fermi_energy, "attrs": {"units": "eV"}}
    bands_arrays["occupations"] = {"data": occupations}

    return bands_arrays


# %%
def main():
    qr = _query()

    proto_id_to_dups = {}
    for res in qr:
        id_ = "|".join(
            [
                res["structure"]["extras.source.database"],
                res["structure"]["extras.source.version"],
                res["structure"]["extras.source.id"],
            ]
        )
        if id_ in PROTOTYPICAL_SOURCE_IDS:
            proto_id_to_dups[id_] = res["structure"]["extras.duplicates"] + [id_]

    id_to_proto_id = {}
    for proto_id, dups in proto_id_to_dups.items():
        for dup in dups:
            id_to_proto_id[dup] = proto_id

    with h5py.File("../../data/batio3/materials.h5", "w") as f:
        for res in qr:
            metadata = {
                "calculation": _get_calculation_metadata(res),
                "atoms": _get_structure_metadata(res, id_to_proto_id),
                "bands": _get_bands_metadata(res),
            }
            arrays = {
                "atoms": _get_atoms_arrays(res),
                "bands": _get_bands_arrays(res),
            }

            mat_group = f.create_group(metadata["atoms"]["id"])

            calc_group = mat_group.create_group("single_shot")
            for k, v in metadata["calculation"].items():
                calc_group.attrs[k] = v

            atoms_group = calc_group.create_group("atoms")
            for k, v in metadata["atoms"].items():
                atoms_group.attrs[k] = v
            for k, v in arrays["atoms"].items():
                attrs = v.pop("attrs", {})
                atoms_group.create_dataset(k, **v)
                for k2, v2 in attrs.items():
                    atoms_group[k].attrs[k2] = v2

            bands_group = calc_group.create_group("bands")
            for k, v in metadata["bands"].items():
                bands_group.attrs[k] = v
            for k, v in arrays["bands"].items():
                attrs = v.pop("attrs", {})
                bands_group.create_dataset(k, **v)
                for k2, v2 in attrs.items():
                    bands_group[k].attrs[k2] = v2


# %%
f = h5py.File("../../data/batio3/materials.h5", "r")


# %%
