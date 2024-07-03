# %%
import json
import pathlib as pl
import numpy as np
from tqdm import tqdm
from aiida import orm, plugins, load_profile
from ase.io.extxyz import write_extxyz
from qe_tools import CONSTANTS

load_profile('3dd_sorep')

PwCalculation = plugins.CalculationFactory('quantumespresso.pw')
PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
PwBandsWorkChain = plugins.WorkflowFactory('quantumespresso.pw.bands')
# %%
STRUCTURE_KWARGS = {
    'cls': orm.StructureData,
    'tag': 'structure',
    'with_outgoing': 'calculation',
    'project': ['*', 'uuid', 'extras.source_id', 'extras.source.id', 'extras.source.database', 'extras.source.version']
}

BANDS_KWARGS = {
    'cls': orm.BandsData,
    'tag': 'bands',
    'with_incoming': 'calculation',
    'project': ['*', 'uuid']
}

OUTPUT_KWARGS = {
    'cls': orm.Dict,
    'tag': 'output',
    'with_incoming': 'calculation',
    'edge_filters': {'label': 'output_parameters'},
    'project': [
        'attributes.wfc_cutoff',
        'attributes.rho_cuotff',
        'attributes.volume',
        'attributes.number_of_species',
        'attributes.number_of_atoms',
        'attributes.number_of_electrons',
        'attributes.number_of_spin_components',
        'attributes.number_of_k_points',
        'attributes.monkhorst_pack_grid',
        'attributes.monkhorst_pack_offset',
        'attributes.lsda',
        'attributes.do_magnetization',
        'attributes.occupations',
        'attributes.smearing_type',
        'attributes.degauss',
        'attributes.fermi_energy',
        'attributes.wall_time_seconds'
    ]
}

INPUT_KWARGS = {
    'cls': orm.Dict,
    'tag': 'input',
    'with_outgoing': 'calculation',
    'edge_filters': {'label': 'parameters'},
    'project': [
        'attributes.SYSTEM.occupations',
        'attributes.SYSTEM.smearing',
        'attributes.SYSTEM.degauss'
    ]
}

DATA_DIR = pl.Path('../data/mc3d/')
# %% zero-shot
qb = orm.QueryBuilder()
qb.append(
    PwCalculation,
    tag='calculation',
    project=['uuid', 'ctime']
)
qb.append(
    orm.Dict,
    tag='input_parameters',
    with_outgoing='calculation',
    filters={'attributes.ELECTRONS.electron_maxstep': 0},
    edge_filters={'label': 'parameters'}
)
qb.append(**STRUCTURE_KWARGS)
qb.append(**BANDS_KWARGS)
qb.append(**OUTPUT_KWARGS)
qb.append(**INPUT_KWARGS)

n_zero_shot = qb.count()
zero_shot_qr = qb.dict()
zero_shot_source_ids = [r['structure']['extras.source.id'] for r in zero_shot_qr]
# %%
for r in tqdm(zero_shot_qr):
    id_ = r['structure'].get('extras.source.id', 'unknown_id')
    id_ = 'unknown_id' if id_ is None else id_
    id_ = id_[3:] if id_.startswith('mp-') else id_

    db = r['structure'].get('extras.source.database', 'unknown_db')
    db = 'unknown_db' if db is None else db

    db_version = r['structure'].get('extras.source.version', 'unknown_version')
    db_version = 'unknown_version' if db_version is None else db_version

    fullname = f'{db}|{db_version}|{id_}'

    structure_dir = DATA_DIR / fullname
    structure_dir.mkdir(exist_ok=True)

    zero_shot_dir = structure_dir / 'zero_shot'
    zero_shot_count = 0
    if zero_shot_dir.exists():
        zero_shot_count = len(list(zero_shot_dir.glob('*')))
    else:
        zero_shot_dir.mkdir()

    dest_dir = zero_shot_dir / f'{zero_shot_count}'
    assert not dest_dir.exists()
    dest_dir.mkdir(exist_ok=True)

    with open(dest_dir / 'structure.xyz', 'w', encoding='utf-8') as fp:
        write_extxyz(fp, r['structure']['*'].get_ase())

    np.savez_compressed(
        file=dest_dir / 'bands.npz',
        bands=r['bands']['*'].get_array('bands'),
        kpoints=r['bands']['*'].get_array('kpoints'),
        weights=r['bands']['*'].get_array('weights'),
        occupations=r['bands']['*'].get_array('occupations'),
    )

    with open(dest_dir / 'metadata.json', 'w', encoding='utf-8') as fp:
        degauss = r['input']['attributes.SYSTEM.degauss']
        if degauss is not None:
            degauss = degauss * CONSTANTS.ry_to_ev
        json.dump({
            'calculation_uuid': r['calculation']['uuid'],
            'calculation_ctime': str(r['calculation']['ctime']),
            'structure_uuid': r['structure']['uuid'],
            'structure_source_database': db,
            'structure_source_database_version': db_version,
            'structure_source_database_id': id_,
            'bands_uuid': r['bands']['uuid'],
            'wfc_cutoff': r['output']['attributes.wfc_cutoff'],  # eV
            'rho_cutoff': r['output']['attributes.rho_cutoff'],  # eV
            'volume': r['output']['attributes.volume'],  # A^3
            'number_of_species': r['output']['attributes.number_of_species'],
            'number_of_atoms': r['output']['attributes.number_of_atoms'],
            'number_of_bands': r['output']['attributes.number_of_bands'],
            'number_of_electrons': r['output']['attributes.number_of_electrons'],
            'number_of_spin_components': r['output']['attributes.number_of_spin_components'],
            'number_of_k_points': r['output']['attributes.number_of_k_points'],
            'monkhorst_pack_grid': r['output']['attributes.monkhorst_pack_grid'],
            'monkhorst_pack_offset': r['output']['attributes.monkhorst_pack_offset'],
            'lsda': r['output']['attributes.lsda'],
            'do_magnetization': r['output']['attributes.do_magnetization'],
            'occupations': r['input']['attributes.SYSTEM.occupations'],
            'smearing_type': r['input']['attributes.SYSTEM.smearing'],
            'degauss': degauss,
            'fermi_energy': r['output']['attributes.fermi_energy'],  # eV
            'wall_time': r['output']['attributes.wall_time_seconds'],  # s
        }, fp)
# %%
##########################################################################################
# %% self-consistent
qb = orm.QueryBuilder()
qb.append(
    PwBaseWorkChain,
    tag='calculation',
    project=['uuid', 'ctime']
)
qb.append(
    orm.Dict,
    tag='input_parameters',
    with_outgoing='calculation',
    filters={
        'attributes.ELECTRONS.electron_maxstep': {'>': 0},
        'attributes.CONTROL.calculation': 'scf'
    },
    edge_filters={'label': 'pw__parameters'}
)
qb.append(
    **STRUCTURE_KWARGS,
    filters={'extras.source.id': {'in': [id_ for id_ in zero_shot_source_ids if id_ is not None]}},
)
qb.append(**BANDS_KWARGS)
qb.append(**OUTPUT_KWARGS)
qb.append(**{**INPUT_KWARGS, 'edge_filters': {'label': 'pw__parameters'}})

n_self_consistent = qb.count()
self_consistent_qr = qb.dict()
self_consistent_source_ids = [r['structure']['extras.source.id'] for r in self_consistent_qr]
# %%
for r in tqdm(self_consistent_qr):
    id_ = r['structure'].get('extras.source.id', 'unknown_id')
    id_ = 'unknown_id' if id_ is None else id_
    id_ = id_[3:] if id_.startswith('mp-') else id_

    db = r['structure'].get('extras.source.database', 'unknown_db')
    db = 'unknown_db' if db is None else db

    db_version = r['structure'].get('extras.source.version', 'unknown_version')
    db_version = 'unknown_version' if db_version is None else db_version

    fullname = f'{db}|{db_version}|{id_}'

    structure_dir = DATA_DIR / fullname
    structure_dir.mkdir(exist_ok=True)

    self_consistent_dir = structure_dir / 'scf'
    self_consistent_count = 0
    if self_consistent_dir.exists():
        self_consistent_count = len(list(self_consistent_dir.glob('*')))
    else:
        self_consistent_dir.mkdir()

    dest_dir = self_consistent_dir / f'{self_consistent_count}'
    assert not dest_dir.exists()
    dest_dir.mkdir(exist_ok=True)

    with open(dest_dir / 'structure.xyz', 'w', encoding='utf-8') as fp:
        write_extxyz(fp, r['structure']['*'].get_ase())

    np.savez_compressed(
        file=dest_dir / 'bands.npz',
        bands=r['bands']['*'].get_array('bands'),
        kpoints=r['bands']['*'].get_array('kpoints'),
        weights=r['bands']['*'].get_array('weights'),
        occupations=r['bands']['*'].get_array('occupations')
    )

    with open(dest_dir / 'metadata.json', 'w', encoding='utf-8') as fp:
        json.dump({
            'calculation_uuid': r['calculation']['uuid'],
            'calculation_ctime': str(r['calculation']['ctime']),
            'structure_uuid': r['structure']['uuid'],
            'structure_source_database': db,
            'structure_source_database_version': db_version,
            'structure_source_database_id': id_,
            'bands_uuid': r['bands']['uuid'],
            'wfc_cutoff': r['output']['*']['wfc_cutoff'],  # eV
            'rho_cutoff': r['output']['*']['rho_cutoff'],  # eV
            'volume': r['output']['*']['volume'],  # A^3
            'number_of_species': r['output']['*']['number_of_species'],
            'number_of_atoms': r['output']['*']['number_of_atoms'],
            'number_of_bands': r['output']['*']['number_of_bands'],
            'number_of_electrons': r['output']['*']['number_of_electrons'],
            'number_of_spin_components': r['output']['*']['number_of_spin_components'],
            'number_of_k_points': r['output']['*']['number_of_k_points'],
            'monkhorst_pack_grid': r['output']['*'].get('monkhorst_pack_grid'),
            'monkhorst_pack_offset': r['output']['*'].get('monkhorst_pack_offset'),
            'lsda': r['output']['*']['lsda'],
            'do_magnetization': r['output']['*']['do_magnetization'],
            'occupations': r['output']['*']['occupations'],
            # TODO: take the smearing type and degauss from INPUT
            'smearing_type': r['output']['*'].get('smearing_type'),
            'degauss': r['output']['*'].get('degauss'),  # eV
            'fermi_energy': r['output']['*']['fermi_energy'],  # eV
            'wall_time': r['output']['*'].get('wall_time_seconds'),  # s
        }, fp)
# %%
##########################################################################################
# %% bands
qb = orm.QueryBuilder()
qb.append(
    PwBandsWorkChain,
    tag='calculation',
    project=['uuid', 'ctime']
)
qb.append(
    **STRUCTURE_KWARGS,
    filters={'extras.source.id': {'in': [id_ for id_ in zero_shot_source_ids if id_ is not None]}},
)
qb.append(**{**BANDS_KWARGS, 'project': ['*', 'uuid', 'attributes.labels', 'attributes.label_numbers']})
qb.append(**{**OUTPUT_KWARGS, 'edge_filters': {'label': 'band_parameters'}})
# qb.append(**{**INPUT_KWARGS, 'edge_filters': {'label': }})

n_bands = qb.count()
bands_qr = qb.dict()
bands_source_ids = [r['structure']['extras.source.id'] for r in bands_qr]

# %%
for r in tqdm(bands_qr):
    id_ = r['structure'].get('extras.source.id', 'unknown_id')
    id_ = 'unknown_id' if id_ is None else id_
    id_ = id_[3:] if id_.startswith('mp-') else id_

    db = r['structure'].get('extras.source.database', 'unknown_db')
    db = 'unknown_db' if db is None else db

    db_version = r['structure'].get('extras.source.version', 'unknown_version')
    db_version = 'unknown_version' if db_version is None else db_version

    fullname = f'{db}|{db_version}|{id_}'

    structure_dir = DATA_DIR / fullname
    structure_dir.mkdir(exist_ok=True)

    bands_dir = structure_dir / 'bands'
    bands_count = 0
    if bands_dir.exists():
        bands_count = len(list(bands_dir.glob('*')))
    else:
        bands_dir.mkdir()

    dest_dir = bands_dir / f'{bands_count}'
    assert not dest_dir.exists()
    dest_dir.mkdir(exist_ok=True)

    with open(dest_dir / 'structure.xyz', 'w', encoding='utf-8') as fp:
        write_extxyz(fp, r['structure']['*'].get_ase())

    np.savez_compressed(
        file=dest_dir / 'bands.npz',
        bands=r['bands']['*'].get_array('bands'),
        kpoints=r['bands']['*'].get_array('kpoints'),
        weights=r['bands']['*'].get_array('weights'),
        occupations=r['bands']['*'].get_array('occupations'),
        labels=np.array(r['bands']['attributes.labels'], dtype='<U32'),
        label_numbers = np.array(r['bands']['attributes.label_numbers'], dtype=int)
    )

    with open(dest_dir / 'metadata.json', 'w', encoding='utf-8') as fp:
        json.dump({
            'calculation_uuid': r['calculation']['uuid'],
            'calculation_ctime': str(r['calculation']['ctime']),
            'structure_uuid': r['structure']['uuid'],
            'structure_source_database': db,
            'structure_source_database_version': db_version,
            'structure_source_database_id': id_,
            'bands_uuid': r['bands']['uuid'],
            'wfc_cutoff': r['output']['*']['wfc_cutoff'],  # eV
            'rho_cutoff': r['output']['*']['rho_cutoff'],  # eV
            'volume': r['output']['*']['volume'],  # A^3
            'number_of_species': r['output']['*']['number_of_species'],
            'number_of_atoms': r['output']['*']['number_of_atoms'],
            'number_of_bands': r['output']['*']['number_of_bands'],
            'number_of_electrons': r['output']['*']['number_of_electrons'],
            'number_of_spin_components': r['output']['*']['number_of_spin_components'],
            'number_of_k_points': r['output']['*']['number_of_k_points'],
            'monkhorst_pack_grid': r['output']['*'].get('monkhorst_pack_grid'),
            'monkhorst_pack_offset': r['output']['*'].get('monkhorst_pack_offset'),
            'lsda': r['output']['*']['lsda'],
            'do_magnetization': r['output']['*']['do_magnetization'],
            'occupations': r['output']['*']['occupations'],
            # TODO: take the smearing type and degauss from INPUT
            'smearing_type': r['output']['*'].get('smearing_type'),
            'degauss': r['output']['*'].get('degauss'),  # eV
            'fermi_energy': r['output']['*']['fermi_energy'],  # eV
            'wall_time': r['output']['*'].get('wall_time_seconds'),  # s
        }, fp)

# %%
import json
import pathlib as pl
from aiida import orm, plugins, load_profile
from qe_tools import CONSTANTS

load_profile('3dd_sorep')

PwCalculation = plugins.CalculationFactory('quantumespresso.pw')
PwBaseWorkChain = plugins.WorkflowFactory('quantumespresso.pw.base')
PwBandsWorkChain = plugins.WorkflowFactory('quantumespresso.pw.bands')

OUTPUT_KWARGS = {
    'cls': orm.Dict,
    'tag': 'output',
    'with_incoming': 'calculation',
    'edge_filters': {'label': 'output_parameters'},
    'project': [
        'attributes.wfc_cutoff',
        'attributes.rho_cutoff',
        'attributes.volume',
        'attributes.number_of_species',
        'attributes.number_of_atoms',
        'attributes.number_of_electrons',
        'attributes.number_of_spin_components',
        'attributes.number_of_bands',
        'attributes.number_of_k_points',
        'attributes.monkhorst_pack_grid',
        'attributes.monkhorst_pack_offset',
        'attributes.lsda',
        'attributes.do_magnetization',
        'attributes.occupations',
        'attributes.smearing_type',
        'attributes.degauss',
        'attributes.fermi_energy',
        'attributes.wall_time_seconds'
    ]
}

INPUT_KWARGS = {
    'cls': orm.Dict,
    'tag': 'input',
    'with_outgoing': 'calculation',
    'edge_filters': {'label': 'parameters'},
    'project': [
        'attributes.SYSTEM.occupations',
        'attributes.SYSTEM.smearing',
        'attributes.SYSTEM.degauss'
    ]
}
# %%
zero_shot_uuids = {}
for json_path in pl.Path('/home/azadoks/Source/git/sorep-npj/data/mc3d/').glob('*/zero_shot/*/metadata.json'):
    with open(json_path, 'r', encoding='utf-8') as fp:
        zero_shot_uuids[str(json_path)] = json.load(fp)['calculation_uuid']
uuid_to_json_path = {v: k for (k, v) in zero_shot_uuids.items()}

qb = orm.QueryBuilder().append(
    PwCalculation,
    tag='calculation',
    filters={'uuid': {'in': list(zero_shot_uuids.values())}},
    project='uuid'
).append(
    **INPUT_KWARGS
).append(
    **OUTPUT_KWARGS
)

qr = qb.dict()
# %%
scf_uuids = {}
for json_path in pl.Path('/home/azadoks/Source/git/sorep-npj/data/mc3d/').glob('*/scf/*/metadata.json'):
    with open(json_path, 'r', encoding='utf-8') as fp:
        scf_uuids[str(json_path)] = json.load(fp)['calculation_uuid']
uuid_to_json_path = {v: k for (k, v) in scf_uuids.items()}

qb = orm.QueryBuilder().append(
    PwBaseWorkChain,
    tag='calculation',
    filters={'uuid': {'in': list(scf_uuids.values())}},
    project='uuid'
).append(
    **{**INPUT_KWARGS, 'edge_filters': {'label': 'pw__parameters'}}
).append(
    **OUTPUT_KWARGS
)

qr = qb.dict()
# %%
bands_uuids = {}
for json_path in pl.Path('/home/azadoks/Source/git/sorep-npj/data/mc3d/').glob('*/bands/*/metadata.json'):
    with open(json_path, 'r', encoding='utf-8') as fp:
        bands_uuids[str(json_path)] = json.load(fp)['calculation_uuid']
uuid_to_json_path = {v: k for (k, v) in bands_uuids.items()}

qb = orm.QueryBuilder().append(
    PwBandsWorkChain,
    tag='calculation',
    filters={'uuid': {'in': list(bands_uuids.values())}},
    project='uuid'
).append(
    **{**INPUT_KWARGS, 'edge_filters': {'label': 'bands__pw__parameters'}}
).append(
    **{**OUTPUT_KWARGS, 'edge_filters': {'label': 'band_parameters'}}
)

qr = qb.dict()
# %%
def write_metadata(query_results, uuid_to_path, dry_run=True):
    for r in query_results:
        with open(uuid_to_path[r['calculation']['uuid']], 'r', encoding='utf-8') as fp:
            metadata = json.load(fp)

        degauss = r['input']['attributes.SYSTEM.degauss']
        if degauss is not None:
            degauss = degauss * CONSTANTS.ry_to_ev

        new_metadata = {
            'calculation_uuid': metadata['calculation_uuid'],
            'calculation_ctime': metadata['calculation_ctime'],
            'structure_uuid': metadata['structure_uuid'],
            'structure_source_database': metadata['structure_source_database'],
            'structure_source_database_version': metadata['structure_source_database_version'],
            'structure_source_database_id': metadata['structure_source_database_id'],
            'bands_uuid': metadata['bands_uuid'],
            'wfc_cutoff': r['output']['attributes.wfc_cutoff'],  # eV
            'rho_cutoff': r['output']['attributes.rho_cutoff'],  # eV
            'volume': r['output']['attributes.volume'],  # A^3
            'number_of_species': r['output']['attributes.number_of_species'],
            'number_of_atoms': r['output']['attributes.number_of_atoms'],
            'number_of_bands': r['output']['attributes.number_of_bands'],
            'number_of_electrons': r['output']['attributes.number_of_electrons'],
            'number_of_spin_components': r['output']['attributes.number_of_spin_components'],
            'number_of_k_points': r['output']['attributes.number_of_k_points'],
            'monkhorst_pack_grid': r['output']['attributes.monkhorst_pack_grid'],
            'monkhorst_pack_offset': r['output']['attributes.monkhorst_pack_offset'],
            'lsda': r['output']['attributes.lsda'],
            'do_magnetization': r['output']['attributes.do_magnetization'],
            'occupations': r['input']['attributes.SYSTEM.occupations'],
            'smearing_type': r['input']['attributes.SYSTEM.smearing'],
            'degauss': degauss,
            'fermi_energy': r['output']['attributes.fermi_energy'],  # eV
            'wall_time': r['output']['attributes.wall_time_seconds'],  # s
        }

        if not dry_run:
            with open(uuid_to_path[r['calculation']['uuid']], 'w', encoding='utf-8') as fp:
                json.dump(new_metadata, fp)
# %%
