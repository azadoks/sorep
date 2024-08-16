# %%
from datetime import datetime
from functools import partial
from multiprocessing import Pool
import os
import pathlib as pl
import platform

import h5py
import imblearn.ensemble as imbe
import joblib
from joblib import dump
import numpy as np
import pandas as pd
import sklearn.dummy as skd
import sklearn.inspection as ski
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.pipeline as skpl
import sklearn.preprocessing as skpp
from tqdm import tqdm

if platform.system() == "Linux":
    import sklearnex

    sklearnex.patch_sklearn()


# %%
def load_data(data_dir: os.PathLike, database: str, calculation_type: str, feature_path: str):
    with h5py.File(data_dir / database / "targets.h5", "r") as f:
        target_df = pd.DataFrame(
            data={
                "id": f["test"]["id"][()].astype(str),
                "meets_tcm_criteria": f["test"]["meets_tcm_criteria"][()],
            }
        )

    with h5py.File(data_dir / database / "features.h5", "r") as f:
        feature_df = pd.DataFrame(
            data={
                "id": f[f"test/{calculation_type}/{feature_path}"]["id"][()].astype(str),
                "features": f[f"test/{calculation_type}/{feature_path}"]["features"][()].tolist(),
            }
        )

    df = pd.merge(target_df, feature_df, on="id")
    X = np.array(df["features"].tolist())
    y = df["meets_tcm_criteria"].to_numpy()
    id_ = df["id"].to_numpy()

    return X, y, id_


def compute_metrics(model, X, y, id_):
    if isinstance(model, (str, pl.Path)):
        model_path = pl.Path(model)
        model = joblib.load(model_path)
        model_info_strings = model_path.stem.split("_")
        model_info = {
            "train_size": float(model_info_strings[0]),
            "random_state": int(model_info_strings[1]),
            "ctime": datetime.fromisoformat(model_info_strings[2].strip("()")).timestamp(),
        }
    else:
        model_info = {}
    y_pred = model.predict(X)
    try:
        impurity_feature_importances = model.named_steps["brfc"].feature_importances_
    except AttributeError:
        impurity_feature_importances = None
    metrics = {
        "accuracy": skm.accuracy_score(y, y_pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y, y_pred, adjusted=True),
        "f1": skm.f1_score(y, y_pred),
        "precision": skm.precision_score(y, y_pred),
        "recall": skm.recall_score(y, y_pred),
        "roc_auc_weighted": skm.roc_auc_score(y, y_pred, average="weighted"),
        "confusion_matrix": skm.confusion_matrix(y, y_pred),
        "impurity_feature_importances": impurity_feature_importances,
        "y_pred": y_pred,
        **model_info,
    }
    return metrics


# %%
DATA_DIR = "../data/"
DATABASE = "mc3d"
CALCULATION_TYPE = "single_shot"
FEATURE_PATHS = [
    "fermi_centered/0",  # -5:513:+5
    "vbm_centered/0",  # -2:513:+6
    "cbm_centered/0",  # -6:513:+2
    "vbm_cbm_concatenated/0",  # -2:257:+3σ, -3σ:257:+2
    "vbm_cbm_concatenated/1",  # -2:257:+2, -2:257:+2
    "vbm_fermi_cbm_concatenated/0",  # -1:171:+1, -1:171:+1, -1:171:+1
    "soap/0",  # No species (~550 features, n=10, l=9, r=6)
]
# %%
data = {}
for feature_path in FEATURE_PATHS:
    X, y, id_ = load_data(pl.Path(DATA_DIR), DATABASE, CALCULATION_TYPE, feature_path)
    data[feature_path] = {"X": X, "y": y, "id_": id_}
# %%
metrics = {}
for feature_path in FEATURE_PATHS:
    model_paths = tqdm(list(pl.Path(f"../models/{CALCULATION_TYPE}/{feature_path}").glob("*pkl")), desc=feature_path)
    with Pool(processes=14) as pool:
        metrics[feature_path] = pool.map(
            partial(
                compute_metrics, X=data[feature_path]["X"], y=data[feature_path]["y"], id_=data[feature_path]["id_"]
            ),
            model_paths,
        )

# %%
metrics_flat = {}
for feature_path, feature_metrics in metrics.items():
    metrics_flat[feature_path] = {key: [] for key in feature_metrics[0].keys()}
    for metric in feature_metrics:
        for key, value in metric.items():
            metrics_flat[feature_path][key].append(value)
    for key, value in metrics_flat[feature_path].items():
        metrics_flat[feature_path][key] = np.array(value)

# %%
with h5py.File(f"../data/{DATABASE}/metrics_test.h5", "a") as f:
    if CALCULATION_TYPE in f:
        del f[CALCULATION_TYPE]
    g_type = f.create_group(CALCULATION_TYPE)
    for feature_path, feature_metrics in metrics_flat.items():
        g_feature = g_type.create_group(feature_path)
        for key, value in feature_metrics.items():
            g_feature.create_dataset(key, data=value, compression="gzip", shuffle=True)
# %%
