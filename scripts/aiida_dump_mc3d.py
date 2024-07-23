# %%
from collections.abc import MutableMapping
from datetime import datetime
import json
import pathlib as pl

from aiida import load_profile, orm, plugins
import h5py
import numpy as np
import pandas as pd
from qe_tools import CONSTANTS
from tqdm import tqdm

import sorep

load_profile("3dd_sorep")

PwCalculation = plugins.CalculationFactory("quantumespresso.pw")
PwBaseWorkChain = plugins.WorkflowFactory("quantumespresso.pw.base")
PwBandsWorkChain = plugins.WorkflowFactory("quantumespresso.pw.bands")
# TODO: not up-to-date on the HDF5 format (see to_hdf5.py)
# %%
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
    ],
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

with open("../data/mc3d/source_to_mc3d_id.json", "r", encoding="utf-8") as fp:
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
                else res["input"]["attributes.SYSTEM.degauss"] * CONSTANTS.ry_to_ev
            ),
            "degauss_units": "eV",
        },
    }


def _get_bands_arrays(res: dict) -> dict:
    metadata = _get_metadata(res)["bands"]
    bands_arrays = {
        "eigenvalues": res["bands"]["*"].get_array("bands"),
        "kpoints": res["bands"]["*"].get_array("kpoints"),
        "weights": res["bands"]["*"].get_array("weights"),
        "cell": np.array(res["structure"]["*"].cell),
        "labels": res["bands"].get("attributes.labels", []),
        "label_numbers": np.array(res["bands"].get("attributes.label_numbers", []), dtype=int),
        "n_electrons": metadata["number_of_electrons"],
        # Completely ignore QE Fermi energy and occupations -- often garbage or missing (<=v6.8 w/ cold or single_shot)
        # "fermi_energy": metadata["qe_fermi_energy"],
        # "occupations": res["bands"]["*"].get_array("occupations"),
    }
    # Add a spin dimension if missing
    bands_arrays["bands"] = (
        bands_arrays["bands"] if bands_arrays["bands"].ndim == 3 else np.expand_dims(bands_arrays["bands"], 0)
    )

    bandstructure = sorep.BandStructure(**bands_arrays)
    fermi_energy = bandstructure.find_fermi_energy(metadata["smearing"], metadata["degauss"], n_electrons_tol=1e-4)
    # Move the Fermi energy to mid-gap if insulating
    bandstructure.fermi_energy = fermi_energy
    if bandstructure.is_insulating():
        fermi_energy = bandstructure.vbm + (bandstructure.cbm - bandstructure.vbm) / 2
    occupations = bandstructure.compute_occupations(metadata["smearing"], metadata["degauss"], fermi_energy)
    bands_arrays["fermi_energy"] = fermi_energy
    bands_arrays["occupations"] = occupations

    return bands_arrays


def _get_atoms_arrays(res: dict) -> dict:
    structure = res["structure"]["*"]
    atoms = structure.get_ase()
    return {**atoms.arrays, "cell": atoms.cell.array, "pbc": atoms.pbc}


def _hdf_result(hdf_group, res: dict) -> None:
    metadata = _get_metadata(res)

    for key, value in metadata["calculation"].items():
        hdf_group.attrs[key] = value

    bands_group = hdf_group.create_group("bands")
    for key, value in metadata["bands"].items():
        bands_group.attrs[key] = value
    for key, value in _get_bands_arrays(res).items():
        if isinstance(value, (np.ndarray, list)):
            bands_group.create_dataset(key, data=value, compression="gzip", shuffle=True)
        else:
            bands_group.create_dataset(key, data=value)

    atoms_group = hdf_group.create_group("atoms")
    for key, value in metadata["structure"].items():
        atoms_group.attrs[key] = value
    for key, value in _get_atoms_arrays(res).items():
        if isinstance(value, (np.ndarray, list)):
            atoms_group.create_dataset(key, data=value, compression="gzip", shuffle=True)
        else:
            atoms_group.create_dataset(key, data=value)


def _single_shot_query() -> list[dict]:
    qb = orm.QueryBuilder()
    qb.append(PwCalculation, tag="calculation", project=["uuid", "ctime"])
    qb.append(**STRUCTURE_KWARGS)
    qb.append(**BANDS_KWARGS)
    qb.append(**{**INPUT_KWARGS, "filters": {"attributes.ELECTRONS.electron_maxstep": 0}})
    qb.append(**{**OUTPUT_KWARGS, "filters": {"attributes.dft_exchange_correlation": "PBE"}})
    return qb.dict()


def _scf_query(single_shot_qr: list[dict]) -> list[dict]:
    single_shot_source_ids = [r["structure"]["extras.source.id"] for r in single_shot_qr]
    qb = orm.QueryBuilder()
    qb.append(PwBaseWorkChain, tag="calculation", project=["uuid", "ctime"])
    qb.append(
        **STRUCTURE_KWARGS,
        filters={"extras.source.id": {"in": [id_ for id_ in single_shot_source_ids if id_ is not None]}},
    )
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


def _bands_query(single_shot_qr: list[dict]) -> list[dict]:
    single_shot_source_ids = [r["structure"]["extras.source.id"] for r in single_shot_qr]
    qb = orm.QueryBuilder()
    qb.append(PwBandsWorkChain, tag="calculation", project=["uuid", "ctime"])
    qb.append(
        **STRUCTURE_KWARGS,
        filters={"extras.source.id": {"in": [id_ for id_ in single_shot_source_ids if id_ is not None]}},
    )
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
    metadata_df["calculation__ctime"] = metadata_df["calculation__ctime"].apply(datetime.fromisoformat)
    metadata_df = metadata_df.sort_values(by="calculation__ctime")  # oldest first
    metadata_df = metadata_df.drop_duplicates(subset="structure__id", keep="last")  # get the newest
    metadata_df = metadata_df[
        metadata_df.structure__id.apply(lambda x: x.startswith("mc3d"))
    ]  # keep only MC3D structures
    return [qr[i] for i in metadata_df.index]


# %%
def main(dry_run: bool = False):
    f = h5py.File(DATA_DIR / "data_hdf.h5", "w")

    name = "SINGLE-SHOT"
    print(f"[{name:9s}] Querying for calculations...")
    single_shot_qr = _single_shot_query()
    print(f"[{name:9s}] Found {len(single_shot_qr)}")
    print(f"[{name:9s}] De-duplicating...")
    single_shot_qr = _deduplicate(single_shot_qr)
    print(f"[{name:9s}] Unique {len(single_shot_qr)}")
    for res in tqdm(single_shot_qr, desc=f"[{name:9s}] Dumping...", ncols=80):
        id_, *_ = _get_id(res)
        try:
            g = f.create_group(f"{id_}/single_shot")
            _hdf_result(g, res)
        except ValueError as e:
            raise ValueError(f"Error for {id_}: {id_}/single_shot") from e
    print()

    name = "SCF"
    print(f"[{name:9s}] Querying for calculations...")
    scf_qr = _scf_query(single_shot_qr)
    print(f"[{name:9s}] Found {len(scf_qr)}")
    print(f"[{name:9s}] De-duplicating...")
    scf_qr = _deduplicate(scf_qr)
    print(f"[{name:9s}] Unique {len(scf_qr)}")
    for res in tqdm(scf_qr, desc=f"[{name:9s}] Dumping...", ncols=80):
        id_, *_ = _get_id(res)
        g = f.create_group(f"{id_}/scf")
        _hdf_result(g, res)
    print()

    name = "BANDS"
    print(f"[{name:9s}] Querying for calculations...")
    bands_qr = _bands_query(single_shot_qr)
    print(f"[{name:9s}] Found {len(bands_qr)}")
    print(f"[{name:9s}] De-duplicating...")
    bands_qr = _deduplicate(bands_qr)
    print(f"[{name:9s}] Unique {len(bands_qr)}")
    for res in tqdm(bands_qr, desc=f"[{name:9s}] Dumping...", ncols=80):
        id_, *_ = _get_id(res)
        g = f.create_group(f"{id_}/bands")
        _hdf_result(g, res)

    return single_shot_qr, scf_qr, bands_qr


# %%
if __name__ == "__main__":
    main()
