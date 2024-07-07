# %%
import json
import pathlib as pl
from datetime import datetime

from aiida import load_profile, orm, plugins
from ase.io.extxyz import write_extxyz
import numpy as np
import pandas as pd
from qe_tools import CONSTANTS
from tqdm import tqdm

import sorep

load_profile("3dd_sorep")

PwCalculation = plugins.CalculationFactory("quantumespresso.pw")
PwBaseWorkChain = plugins.WorkflowFactory("quantumespresso.pw.base")
PwBandsWorkChain = plugins.WorkflowFactory("quantumespresso.pw.bands")
# %%
STRUCTURE_KWARGS = {
    "cls": orm.StructureData,
    "tag": "structure",
    "with_outgoing": "calculation",
    "project": ["*", "uuid", "extras.source_id", "extras.source.id", "extras.source.database", "extras.source.version"],
}

BANDS_KWARGS = {
    "cls": orm.BandsData,
    "tag": "bands",
    "with_incoming": "calculation",
    "project": [
        "*",
        "uuid",
        "attributes.labels",
        "attributes.label_numbers"
    ]
}

OUTPUT_KWARGS = {
    "cls": orm.Dict,
    "tag": "output",
    "with_incoming": "calculation",
    "edge_filters": {"label": "output_parameters"},
    "project": [
        "attributes.wfc_cutoff",
        "attributes.rho_cutoff",
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

DATA_DIR = pl.Path("../data/mc3d/")
# %%
def _get_metadata(res: dict) -> dict:
    """Construct metadata dictionary from a query result

    Args:
        res (dict): Query result from QueryBuilder.dict

    Returns:
        dict: Metadata dictionary.
    """
    id_ = res["structure"].get("extras.source.id", "unknown_id")
    id_ = "unknown_id" if id_ is None else id_
    id_ = id_[3:] if id_.startswith("mp-") else id_

    db = res["structure"].get("extras.source.database", "unknown_db")
    db = "unknown_db" if db is None else db

    db_version = res["structure"].get("extras.source.version", "unknown_version")
    db_version = "unknown_version" if db_version is None else db_version

    sorep_id = "|".join([db, db_version, id_])

    degauss = res["input"]["attributes.SYSTEM.degauss"]
    if degauss is not None:
        res["input"]["attributes.SYSTEM.degauss"] = degauss * CONSTANTS.ry_to_ev

    return {
        # Calculation Node
        "calculation_uuid": res["calculation"]["uuid"],
        "calculation_ctime": str(res["calculation"]["ctime"]),
        # Structure Node + extras
        "structure_uuid": res["structure"]["uuid"],
        "sorep_id": sorep_id,
        "structure_source_database": db,
        "structure_source_database_version": db_version,
        "structure_source_database_id": id_,
        # Bands Node
        "bands_uuid": res["bands"]["uuid"],
        # Output parameters
        **{key.split(".")[-1]: value for (key, value) in res["output"].items()},
        # Input parameters
        **{key.split(".")[-1]: value for (key, value) in res["input"].items()}
    }

def _dump_result(
    res: dict,
    calc_type: str,
    recompute_fermi_occupations: bool=True,
    dry_run: bool=False
) -> None:
    """Write the structure, bands, and metadata for a calculation from a query result dictionary.

    Args:
        res (dict): Result dictionary.
        calc_type (str): Calculation type (used to name sub-directory under calculation ID)
        recompute_fermi_occupations (bool, optional): Recompute the Fermi energy and occupations. Defaults to True.
        dry_run (bool, optional): Perform all processing but do not write any files. Defaults to False.
    """
    metadata = _get_metadata(res)

    structure_calc_type_dir = DATA_DIR / metadata["sorep_id"] / calc_type
    count = len(list(structure_calc_type_dir.glob("*")))
    dest_dir = structure_calc_type_dir / f"{count}"
    if not dry_run:
        dest_dir.mkdir(exist_ok=False, parents=True)

    bands_arrays = {
        "bands": res["bands"]["*"].get_array("bands"),
        "kpoints": res["bands"]["*"].get_array("kpoints"),
        "weights": res["bands"]["*"].get_array("weights"),
        "occupations": res["bands"]["*"].get_array("occupations"),
        "labels": np.array(res["bands"].get("attributes.labels", []), dtype="<U32"),
        "label_numbers": np.array(res["bands"].get("attributes.label_numbers", []), dtype=int),
    }

    if recompute_fermi_occupations:
        bandstructure = sorep.BandStructure(**bands_arrays, n_electrons=metadata['number_of_electrons'])
        fermi_energy = bandstructure.find_fermi_energy(metadata['smearing'], metadata['degauss'], n_electrons_tol=1e-4)
        occupations = bandstructure.compute_occupations(metadata['smearing'], metadata['degauss'], fermi_energy)
        metadata['fermi_energy'] = fermi_energy
        bands_arrays['occupations'] = occupations

    if not dry_run:
        # Write structure
        with open(dest_dir / "structure.xyz", "w", encoding="utf-8") as fp:
            write_extxyz(fp, res["structure"]["*"].get_ase())
        # Write bands
        np.savez_compressed(file=dest_dir / "bands.npz", **bands_arrays)
        # Write metadata
        with open(dest_dir / "metadata.json", "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

def _zero_shot_query() -> list[dict]:
    qb = orm.QueryBuilder()
    qb.append(PwCalculation, tag="calculation", project=["uuid", "ctime"])
    qb.append(
        orm.Dict,
        tag="input_parameters",
        with_outgoing="calculation",
        filters={"attributes.ELECTRONS.electron_maxstep": 0},
        edge_filters={"label": "parameters"},
    )
    qb.append(**STRUCTURE_KWARGS)
    qb.append(**BANDS_KWARGS)
    qb.append(**OUTPUT_KWARGS)
    qb.append(**INPUT_KWARGS)
    return qb.dict()

def _scf_query(zero_shot_qr: list[dict]) -> list[dict]:
    zero_shot_source_ids = [r["structure"]["extras.source.id"] for r in zero_shot_qr]
    qb = orm.QueryBuilder()
    qb.append(PwBaseWorkChain, tag="calculation", project=["uuid", "ctime"])
    qb.append(
        orm.Dict,
        tag="input_parameters",
        with_outgoing="calculation",
        filters={"attributes.ELECTRONS.electron_maxstep": {">": 0}, "attributes.CONTROL.calculation": "scf"},
        edge_filters={"label": "pw__parameters"},
    )
    qb.append(
        **STRUCTURE_KWARGS,
        filters={"extras.source.id": {"in": [id_ for id_ in zero_shot_source_ids if id_ is not None]}},
    )
    qb.append(**BANDS_KWARGS)
    qb.append(**OUTPUT_KWARGS)
    qb.append(**{**INPUT_KWARGS, "edge_filters": {"label": "pw__parameters"}})
    return qb.dict()

def _bands_query(zero_shot_qr: list[dict]) -> list[dict]:
    zero_shot_source_ids = [r["structure"]["extras.source.id"] for r in zero_shot_qr]
    qb = orm.QueryBuilder()
    qb.append(PwBandsWorkChain, tag="calculation", project=["uuid", "ctime"])
    qb.append(
        **STRUCTURE_KWARGS,
        filters={"extras.source.id": {"in": [id_ for id_ in zero_shot_source_ids if id_ is not None]}},
    )
    qb.append(**{**BANDS_KWARGS, "project": ["*", "uuid", "attributes.labels", "attributes.label_numbers"]})
    qb.append(**{**OUTPUT_KWARGS, "edge_filters": {"label": "band_parameters"}})
    qb.append(**{**INPUT_KWARGS, "edge_filters": {"label": "bands__pw__parameters"}})
    return qb.dict()

def _deduplicate(qr: list[dict]) -> list[dict]:
    metadata_df = pd.DataFrame([_get_metadata(r) for r in qr])
    metadata_df['calculation_ctime'] = metadata_df['calculation_ctime'].apply(datetime.fromisoformat)
    metadata_df = metadata_df.sort_values(by='calculation_ctime')  # oldest first
    metadata_df = metadata_df.drop_duplicates(subset='sorep_id', keep='last')
    return [qr[i] for i in metadata_df.index]
# %%
def main(dry_run: bool=True):
    print('[ZERO-SHOT] Querying for calculations...')
    zero_shot_qr = _zero_shot_query()
    print(f'[ZERO-SHOT] Found {len(zero_shot_qr)}')
    print('[ZERO-SHOT] De-duplicating...')
    zero_shot_qr = _deduplicate(zero_shot_qr)
    print(f'[ZERO-SHOT] Unique {len(zero_shot_qr)}')
    for res in tqdm(zero_shot_qr, desc='[ZERO-SHOT] Dumping...', ncols=80):
        _dump_result(res, calc_type='zero_shot', recompute_fermi_occupations=True, dry_run=dry_run)

    print('[SCF] Querying for calculations...')
    scf_qr = _scf_query(zero_shot_qr)
    print(f'[SCF] Found {len(scf_qr)}')
    print('[SCF] De-duplicating...')
    scf_qr = _deduplicate(scf_qr)
    print(f'[SCF] Unique {len(scf_qr)}')
    for res in tqdm(scf_qr, desc='[SCF] Dumping...', ncols=80):
        _dump_result(res, calc_type='scf', recompute_fermi_occupations=True, dry_run=dry_run)

    print('[BANDS] Querying for calculations...')
    bands_qr = _scf_query(zero_shot_qr)
    print(f'[BANDS] Found {len(bands_qr)}')
    print('[BANDS] De-duplicating...')
    bands_qr = _deduplicate(bands_qr)
    print(f'[SCF] Unique {len(bands_qr)}')
    for res in tqdm(bands_qr, desc='[BANDS] Dumping...', ncols=80):
        _dump_result(res, calc_type='bands', recompute_fermi_occupations=True, dry_run=dry_run)
# %%
if __name__ == '__main__':
    main()
