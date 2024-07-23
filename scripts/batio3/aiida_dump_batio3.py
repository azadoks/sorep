# %%
import json

from aiida import load_profile, orm
import h5py
import numpy as np
import pandas as pd

load_profile("batio3")
# %%
STRUCTURE_GROUP_LABEL = "BaTiO3/structures/optimized"
PROTOTYPICAL_STRUCTURES = [
    {"material_id": "mpds|1.0.0|S1638131", "group": "Layered perovskite 1"},
    {"material_id": "mpds|1.0.0|S1129458", "group": "Rhombohedral"},
    {"material_id": "icsd|2017.2|109327", "group": "Supertetragonal non-perovskite"},
    {"material_id": "icsd|2017.2|100804", "group": "Tetragonal"},
    {"material_id": "mpds|1.0.0|S1638130", "group": "Layered perovskite 2"},
    {"material_id": "mpds|1.0.0|S1823200", "group": "Cubic"},
    {"material_id": "cod|176429|1542189", "group": "Cubic Ba/Ti swapped"},
    {"material_id": "mpds|1.0.0|S1129461", "group": "Orthorhombic"},
    {"material_id": "icsd|2017.2|154346", "group": "Orthorhombic non-perovskite"},
]


# %%
def _query_structures():
    qb = orm.QueryBuilder()
    qb.append(orm.Group, filters={"label": STRUCTURE_GROUP_LABEL}, tag="group")
    qb.append(
        orm.StructureData,
        with_group="group",
        project=["*", "uuid", "extras.source", "extras.duplicates"],
        tag="structure",
    )
    return qb.dict()


# %%
qr = _query_structures()
qr_df = pd.DataFrame([r["structure"] for r in qr])
qr_df
# %%
