# %%
from collections.abc import MutableMapping
from datetime import datetime
import json
import pathlib as pl

from aiida import load_profile, orm, plugins
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sorep

load_profile("3dd_sorep")

PwCalculation = plugins.CalculationFactory("quantumespresso.pw")
PwBaseWorkChain = plugins.WorkflowFactory("quantumespresso.pw.base")
PwBandsWorkChain = plugins.WorkflowFactory("quantumespresso.pw.bands")
# %%
HDF_ARRAY_KWARGS = {"compression": "gzip", "shuffle": True}

STRUCTURE_KWARGS = {
    "cls": orm.StructureData,
    "tag": "structure",
    "with_outgoing": "calculation",
    "project": [
        "*",
        "uuid",
        "extras.source_id",
        "extras.source.id",
        "extras.source.database",
        "extras.source.version",
        "extras.formula_hill",
        "extras.spacegroup_international",
        "extras.matproj_duplicate",
        "extras.mc3d_id",
    ],
    "filters": {"extras": {"has_key": "mc3d_id"}},
}

BANDS_KWARGS = {
    "cls": orm.BandsData,
    "tag": "bands",
    "with_incoming": "calculation",
    "project": ["*", "uuid", "attributes.labels", "attributes.label_numbers"],
}

OUTPUT_KWARGS = {
    "cls": orm.Dict,
    "tag": "output",
    "with_incoming": "calculation",
    "edge_filters": {"label": "output_parameters"},
    "project": [
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
    ],
}

INPUT_KWARGS = {
    "cls": orm.Dict,
    "tag": "input",
    "with_outgoing": "calculation",
    "edge_filters": {"label": "parameters"},
    "project": ["attributes.SYSTEM.occupations", "attributes.SYSTEM.smearing", "attributes.SYSTEM.degauss"],
}

with open("../data/mc3d/util/source_to_mc3d_id.json", "r", encoding="utf-8") as fp:
    SOURCE_TO_MC3D = json.load(fp)

DATA_DIR = pl.Path("../data/mc3d/")


# %%
def _flatten(dictionary, parent_key="", separator="__"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(_flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _get_id(res: dict) -> str:
    # Source database
    if res["structure"]["extras.source.database"] is not None:
        source_db = res["structure"]["extras.source.database"]
    else:
        source_db = ""
    # Source database version
    if res["structure"]["extras.source.version"] is not None:
        source_version = res["structure"]["extras.source.version"]
    else:
        source_version = ""
    # ID in source database
    if res["structure"]["extras.source.id"] is not None:
        source_id = res["structure"]["extras.source.id"]
    elif res["structure"]["extras.source_id"] is not None:
        source_id = res["structure"]["extras.source_id"]
    else:
        source_id = ""
    source_id = str(source_id)
    if source_id.startswith("mp-"):
        source_id = source_id[3:]
        source_db = "mp"
    # Fall back to structure UUID
    if (source_id == "None") and (source_db == "None") and (source_version == "None"):
        source_id = res["structure"]["uuid"]
        source_db = ""
        source_version = ""
    # Construct full source ID to map to MC3D ID
    if source_db == "mp":
        id_ = "-".join([source_db, source_id])
    else:
        full_source_id = "|".join([source_db, source_version, source_id])
        id_ = SOURCE_TO_MC3D.get(full_source_id, full_source_id)
    return id_, source_db, source_version, source_id


def _get_metadata(res: dict) -> dict:
    """Construct metadata dictionary from a query result

    Args:
        res (dict): Query result from QueryBuilder.dict

    Returns:
        dict: Metadata dictionary.
    """
    id_, source_db, source_version, source_id = _get_id(res)
    return {
        "calculation": {
            "uuid": res["calculation"]["uuid"],
            "ctime": str(res["calculation"]["ctime"]),
            "wall_time_seconds": (
                -1.0
                if res["output"]["attributes.wall_time_seconds"] is None
                else res["output"]["attributes.wall_time_seconds"]
            ),
            "creator_version": res["output"]["attributes.creator_version"],
            "dft_exchange_correlation": res["output"]["attributes.dft_exchange_correlation"],
            "wfc_cutoff": res["output"]["attributes.wfc_cutoff"],
            "wfc_cutoff_units": res["output"]["attributes.wfc_cutoff_units"],
            "rho_cutoff": res["output"]["attributes.rho_cutoff"],
            "rho_cutoff_units": res["output"]["attributes.rho_cutoff_units"],
        },
        "structure": {
            "uuid": res["structure"]["uuid"],
            "id": id_,
            "formula_hill": res["structure"]["extras.formula_hill"],
            "spacegroup_international": res["structure"]["extras.spacegroup_international"],
            "matproj_duplicate": (
                ""
                if res["structure"]["extras.matproj_duplicate"] is None
                else res["structure"]["extras.matproj_duplicate"]
            ),
            "source_database": source_db,
            "source_version": source_version,
            "source_id": source_id,
            "volume": res["output"]["attributes.volume"],
            "volume_units": "angstrom^3",
            "number_of_species": res["output"]["attributes.number_of_species"],
            "number_of_atoms": res["output"]["attributes.number_of_atoms"],
        },
        "bands": {
            "uuid": res["bands"]["uuid"],
            "qe_fermi_energy": res["output"]["attributes.fermi_energy"],
            "qe_fermi_energy_units": res["output"]["attributes.fermi_energy_units"],
            "number_of_electrons": res["output"]["attributes.number_of_electrons"],
            "number_of_bands": res["output"]["attributes.number_of_bands"],
            "number_of_k_points": res["output"]["attributes.number_of_k_points"],
            "number_of_spin_components": res["output"]["attributes.number_of_spin_components"],
            "monkhorst_pack_grid": (
                []
                if res["output"]["attributes.monkhorst_pack_grid"] is None
                else res["output"]["attributes.monkhorst_pack_grid"]
            ),
            "monkhorst_pack_offset": (
                []
                if res["output"]["attributes.monkhorst_pack_offset"] is None
                else res["output"]["attributes.monkhorst_pack_offset"]
            ),
            "smearing": res["input"]["attributes.SYSTEM.smearing"],
            "occupations": res["input"]["attributes.SYSTEM.occupations"],
            "degauss": (
                0.0
                if res["input"]["attributes.SYSTEM.degauss"] is None
                else res["input"]["attributes.SYSTEM.degauss"] * sorep.constants.RY_TO_EV
            ),
            "degauss_units": "eV",
        },
    }


def _get_bands_arrays(res: dict) -> dict:
    metadata = _get_metadata(res)["bands"]
    bands_arrays = {
        "eigenvalues": {"data": res["bands"]["*"].get_array("bands"), "attrs": {"units": "eV"}, **HDF_ARRAY_KWARGS},
        "kpoints": {
            "data": res["bands"]["*"].get_array("kpoints"),
            "attrs": {"units": "dimensionless"},
            **HDF_ARRAY_KWARGS,
        },
        "weights": {"data": res["bands"]["*"].get_array("weights"), **HDF_ARRAY_KWARGS},
        "cell": {"data": np.array(res["structure"]["*"].cell), "attrs": {"units": "angstrom"}, **HDF_ARRAY_KWARGS},
        "labels": {
            "data": res["bands"].get("attributes.labels", []),
            "dtype": h5py.string_dtype(encoding="utf-8", length=None),
        },
        "label_numbers": {"data": np.array(res["bands"].get("attributes.label_numbers", []), dtype=int)},
        "n_electrons": {"data": metadata["number_of_electrons"]},
        # Completely ignore QE Fermi energy and occupations -- often garbage or missing (<=v6.8 w/ cold or single_shot)
        # "fermi_energy": metadata["qe_fermi_energy"],
        # "occupations": res["bands"]["*"].get_array("occupations"),
    }
    # Add a spin dimension if missing
    bands_arrays["eigenvalues"]["data"] = (
        bands_arrays["eigenvalues"]["data"]
        if bands_arrays["eigenvalues"]["data"].ndim == 3
        else np.expand_dims(bands_arrays["eigenvalues"]["data"], 0)
    )

    # Construct BandStructure object
    bandstructure = sorep.BandStructure(**{key: value["data"] for (key, value) in bands_arrays.items()})
    smearing_type = metadata["smearing"]
    smearing_width = metadata["degauss"]

    # Recompute Fermi energy
    bandstructure.fermi_energy = bandstructure.find_fermi_energy(
        smearing_type,
        smearing_width,
        n_electrons_tol=1e-6,
        n_electrons_kwargs=N_ELECTRONS_KWARGS,
        dn_electrons_kwargs=DN_ELECTRONS_KWARGS,
        ddn_electrons_kwargs=DDN_ELECTRONS_KWARGS,
        newton_kwargs=NEWTON_KWARGS,
    )["fermi_energy"]

    # Move the Fermi energy to mid-gap if insulating
    if bandstructure.is_insulating():
        fermi_energy_mid_gap = bandstructure.vbm + (bandstructure.cbm - bandstructure.vbm) / 2
        n_electrons_mid_gap = bandstructure.compute_n_electrons("delta", 0.0, fermi_energy_mid_gap)
        if np.abs(n_electrons_mid_gap - bandstructure.n_electrons) < 1e-6:
            bandstructure.fermi_energy = fermi_energy_mid_gap
            smearing_type = "delta"
            smearing_width = 0.0

    # Check that the number of electrons from the Fermi level is within tolerance
    n_electrons = bandstructure.compute_n_electrons(smearing_type, smearing_width)
    if np.abs(n_electrons - bandstructure.n_electrons) > 1e-6:
        raise ValueError(f"Number of electrons mismatch: {n_electrons} != {bandstructure.n_electrons}")

    # Recompute occupations
    bandstructure.occupations = bandstructure.compute_occupations(smearing_type, smearing_width)

    # Save new Fermi energy and occupations in the bands arrays
    bands_arrays["fermi_energy"] = {"data": bandstructure.fermi_energy, "attrs": {"units": "eV"}}
    bands_arrays["occupations"] = {"data": bandstructure.occupations}

    return bands_arrays


def _get_atoms_arrays(res: dict) -> dict:
    structure = res["structure"]["*"]
    atoms = structure.get_ase()
    return {
        "positions": {"data": atoms.arrays["positions"], "attrs": {"units": "angstrom"}, **HDF_ARRAY_KWARGS},
        "numbers": {"data": atoms.arrays["numbers"], **HDF_ARRAY_KWARGS},
        "masses": {"data": atoms.arrays["masses"], "attrs": {"units": "amu"}, **HDF_ARRAY_KWARGS},
        "cell": {"data": np.array(atoms.cell), "attrs": {"units": "angstrom"}, **HDF_ARRAY_KWARGS},
        "pbc": {"data": atoms.pbc},
    }


def _get_arrays(res: dict) -> dict:
    return {
        "metadata": _get_metadata(res),
        "bands": _get_bands_arrays(res),
        "atoms": _get_atoms_arrays(res),
    }


def _hdf_result(hdf_group, arrays: dict) -> None:
    metadata = arrays["metadata"]

    for key, value in metadata["calculation"].items():
        hdf_group.attrs[key] = value

    bands_group = hdf_group.create_group("bands")
    for key, value in metadata["bands"].items():
        bands_group.attrs[key] = value
    for key, value in arrays["bands"].items():
        attrs = value.pop("attrs", {})
        bands_group.create_dataset(key, **value)
        value["attrs"] = attrs
        for k, v in value["attrs"].items():
            bands_group[key].attrs[k] = v

    atoms_group = hdf_group.create_group("atoms")
    for key, value in metadata["structure"].items():
        atoms_group.attrs[key] = value
    for key, value in arrays["atoms"].items():
        attrs = value.pop("attrs", {})
        atoms_group.create_dataset(key, **value)
        value["attrs"] = attrs
        for k, v in value["attrs"].items():
            atoms_group[key].attrs[k] = v


def _single_shot_query() -> list[dict]:
    qb = orm.QueryBuilder()
    qb.append(
        PwCalculation,
        tag="calculation",
        project=["uuid", "ctime", "attributes.version.core", "attributes.version.plugin"],
    )
    qb.append(**STRUCTURE_KWARGS)
    qb.append(**BANDS_KWARGS)
    qb.append(**{**INPUT_KWARGS, "filters": {"attributes.ELECTRONS.electron_maxstep": 0}})
    qb.append(**{**OUTPUT_KWARGS, "filters": {"attributes.dft_exchange_correlation": "PBE"}})
    return qb.dict()


def _scf_query() -> list[dict]:
    qb = orm.QueryBuilder()
    qb.append(
        PwBaseWorkChain,
        tag="calculation",
        project=["uuid", "ctime", "attributes.version.core", "attributes.version.plugin"],
    )
    qb.append(**STRUCTURE_KWARGS)
    qb.append(**BANDS_KWARGS)
    qb.append(
        **{
            **INPUT_KWARGS,
            "edge_filters": {"label": "pw__parameters"},
            "filters": {"attributes.ELECTRONS.electron_maxstep": {">": 0}, "attributes.CONTROL.calculation": "scf"},
        }
    )
    qb.append(**{**OUTPUT_KWARGS, "filters": {"attributes.dft_exchange_correlation": "PBE"}})
    return qb.dict()


def _bands_query() -> list[dict]:
    qb = orm.QueryBuilder()
    qb.append(
        PwBandsWorkChain,
        tag="calculation",
        project=["uuid", "ctime", "attributes.version.core", "attributes.version.plugin"],
    )
    qb.append(**STRUCTURE_KWARGS)
    qb.append(**{**BANDS_KWARGS, "project": ["*", "uuid", "attributes.labels", "attributes.label_numbers"]})
    qb.append(**{**INPUT_KWARGS, "edge_filters": {"label": "bands__pw__parameters"}})
    qb.append(
        **{
            **OUTPUT_KWARGS,
            "edge_filters": {"label": "band_parameters"},
            "filters": {"attributes.dft_exchange_correlation": "PBESOL"},
        }
    )
    return qb.dict()


def _deduplicate(qr: list[dict]) -> list[dict]:
    metadata_df = pd.DataFrame([_flatten(_get_metadata(r)) for r in qr])
    metadata_df = metadata_df[
        metadata_df.structure__id.apply(lambda x: x.startswith("mc3d"))
    ]  # keep only MC3D structures
    metadata_df["calculation__ctime"] = metadata_df["calculation__ctime"].apply(datetime.fromisoformat)
    metadata_df = metadata_df.sort_values(by="calculation__ctime")  # oldest first
    metadata_df = metadata_df.drop_duplicates(subset="structure__id", keep="last")  # get the newest
    return [qr[i] for i in metadata_df.index]


# %%
N_ELECTRONS_KWARGS = {"max_exponent": np.inf}
DN_ELECTRONS_KWARGS = {"max_exponent": np.inf}
DDN_ELECTRONS_KWARGS = {"max_exponent": np.inf}
NEWTON_KWARGS = {"maxiter": 500}


def main():
    name = "SINGLE-SHOT"
    print(f"[{name:>12s}] Querying for calculations...")
    single_shot_qr = _single_shot_query()
    print(f"[{name:>12s}] Found {len(single_shot_qr)}")
    print(f"[{name:>12s}] De-duplicating...")
    single_shot_qr = _deduplicate(single_shot_qr)
    print(f"[{name:>12s}] Unique {len(single_shot_qr)}")
    print()

    name = "SCF"
    print(f"[{name:>12s}] Querying for calculations...")
    scf_qr = _scf_query()
    print(f"[{name:>12s}] Found {len(scf_qr)}")
    print(f"[{name:>12s}] De-duplicating...")
    scf_qr = _deduplicate(scf_qr)
    print(f"[{name:>12s}] Unique {len(scf_qr)}")
    print()

    name = "BANDS"
    print(f"[{name:>12s}] Querying for calculations...")
    bands_qr = _bands_query()
    print(f"[{name:>12s}] Found {len(bands_qr)}")
    print(f"[{name:>12s}] De-duplicating...")
    bands_qr = _deduplicate(bands_qr)
    print(f"[{name:>12s}] Unique {len(bands_qr)}")
    print()

    combined_qr = {}
    for res in single_shot_qr:
        mc3d_id = res["structure"]["extras.mc3d_id"]
        if mc3d_id in combined_qr:
            combined_qr[mc3d_id]["single_shot"] = res
        else:
            combined_qr[mc3d_id] = {"single_shot": res}
    for res in scf_qr:
        mc3d_id = res["structure"]["extras.mc3d_id"]
        if mc3d_id in combined_qr:
            combined_qr[mc3d_id]["scf"] = res
    for res in bands_qr:
        mc3d_id = res["structure"]["extras.mc3d_id"]
        if mc3d_id in combined_qr:
            combined_qr[mc3d_id]["bands"] = res

    combined_qr = {mc3d_id: combined_res for (mc3d_id, combined_res) in combined_qr.items() if len(combined_res) == 3}
    combined_qr = [item[1] for item in sorted(combined_qr.items())]
    print(f"[{'COMBINED':>12s}] Found {len(combined_qr)}")

    combined_arrays = []
    for res in tqdm(combined_qr, desc="[COMBINED] Getting arrays...", ncols=80):
        try:
            combined_arrays.append({k: _get_arrays(v) for (k, v) in res.items()})
        except Exception as exc:
            print(f"Failed to get arrays for {res['single_shot']['structure']['extras.mc3d_id']}: {exc}")

    train_arrays, test_arrays = train_test_split(combined_arrays, test_size=0.1, random_state=9997)
    print(f"[{'TRAIN-TEST':>12s}] {len(train_arrays)} - {len(test_arrays)}")

    f = h5py.File(DATA_DIR / "materials_train_test.h5", "w")
    for train_test, material_set in zip(["train", "test"], [train_arrays, test_arrays]):
        train_test_group = f.create_group(train_test)
        for material_arrays in tqdm(material_set, desc=f"[{train_test:>12s}] Dumping...", ncols=80):
            mc3d_id = material_arrays["single_shot"]["metadata"]["structure"]["id"]
            for calc_type, calc_type_arrays in material_arrays.items():
                g = train_test_group.create_group(f"{mc3d_id}/{calc_type}")
                _hdf_result(g, calc_type_arrays)

    return None


# %%
if __name__ == "__main__":
    main()

# %%
