# %%
from collections.abc import MutableMapping

from fastdist import fastdist
import h5py
import numpy as np

# Warm up the JIT
for i in range(10):
    fastdist.matrix_pairwise_distance(
        np.random.standard_normal((10, 10)).astype(np.float64), fastdist.euclidean, "euclidean"
    )
    fastdist.matrix_pairwise_distance(
        np.random.standard_normal((10, 10)).astype(np.float32), fastdist.euclidean, "euclidean"
    )


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


# %%
METRICS = {
    "cosine": {"metric": fastdist.cosine, "metric_name": "cosine"},
    "euclidean": {"metric": fastdist.euclidean, "metric_name": "euclidean"},
}
FEATURE_PATHS = ["fermi_centered/0", "vbm_fermi_cbm_concatenated/0"]
CALCULATION_TYPES = ["single_shot", "scf"]
# %%
for calculation_type in CALCULATION_TYPES:
    f_features = h5py.File("../data/mc3d/features.h5", "r")
    f_distances = h5py.File("../data/mc3d/distances.h5", "a")
    g_distances = (
        f_distances[calculation_type] if calculation_type in f_distances else f_distances.create_group(calculation_type)
    )
    for feature_path in FEATURE_PATHS:
        # Load features
        features = np.concatenate(
            [
                f_features["train"][calculation_type][feature_path]["features"][()],
                f_features["test"][calculation_type][feature_path]["features"][()],
            ]
        )
        ids = np.concatenate(
            [
                f_features["train"][calculation_type][feature_path]["id"][()],
                f_features["test"][calculation_type][feature_path]["id"][()],
            ]
        )
        # Compute distances
        for metric, metric_kwargs in METRICS.items():
            key = f"{feature_path}/{metric}"
            if key in g_distances:
                del g_distances[key]
            print(f"Computing {key} (N={features.shape[0]})")
            # Returns ((N * (N - 1) // 2 - N), 1) array, which is flattened below
            D = fastdist.matrix_pairwise_distance(features, **metric_kwargs, return_matrix=False)
            # Overwite existing dataset
            # Save flat upper triangle of distance matrix along with material ids
            g_distances.create_dataset(f"{key}/distance", data=np.squeeze(D), compression="gzip", shuffle=True)
            g_distances.create_dataset(f"{key}/id", data=ids)
            # Try to free up some memory
            del D
# %%
