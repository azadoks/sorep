# %%
import json

import h5py
import numpy as np
from tqdm import tqdm

# %%
with open("../data/mc3d/id_map.json", "r") as f:
    id_map = json.load(f)

source_to_mc3d = {}
for mc3d_id, source_info in id_map.items():
    source_id = "|".join([source_info["database"], source_info["version"], source_info["id"]])
    assert source_id not in source_to_mc3d
    source_to_mc3d[source_id] = mc3d_id

# %%
# with h5py.File("../data/mc3d/data.h5", "a") as f:
#     for source_id in tqdm(list(f.keys())):
#         if source_id.startswith("mp|"):
#             new_id = "-".join(["mp", source_id.split("|")[-1]])
#         else:
#             new_id = source_to_mc3d.get(source_id, source_id)
#         f.move(source_id, new_id)
# %%
# with h5py.File("../data/mc3d/distances_single_shot.h5", "a") as f:
#     for feature_type, feature_group in f.items():
#         for feature_instance, instance_group in feature_group.items():
#             for distance_type, distance_group in instance_group.items():
#                 material_ids = distance_group["material_id"][()].astype(str)
#                 new_material_ids = [source_to_mc3d.get(mid, mid) for mid in material_ids]
#                 distance_group["material_id"][:] = new_material_ids

# %%
# with h5py.File("../data/mc3d/features_single_shot.h5", "a") as f:
#     for feature_type, feature_group in f.items():
#         for feature_instance, instance_group in feature_group.items():
#             print(instance_group, instance_group.keys())
#             material_ids = instance_group["material_id"][()].astype(str)
#             new_material_ids = [source_to_mc3d.get(mid, mid) for mid in material_ids]
#             instance_group["material_id"][:] = new_material_ids
# %%
# targets = dict(np.load("../data/mc3d/targets.npz"))
# targets["material_id"] = np.array([source_to_mc3d.get(mid, mid) for mid in targets["material_id"]])
# np.savez_compressed("../data/mc3d/targets.npz", **targets)

# %%
