# %%
import os
import pathlib as pl

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.dummy as skd
import sklearn.inspection as ski
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.pipeline as skpl
import sklearn.preprocessing as skpp
from tqdm import tqdm

# %%
FEATURE_KWARGS = {
    "vbm_centered/0": {
        "title": r"$E_{\mathrm{VBM}}(-2,+6) \mathrm{eV}$",
        "shortname": "VBM",
        "line_kwargs": {"c": "#aa0000"},
        "fill_kwargs": {"color": "#aa0000", "alpha": 0.3},
        "errorbar_kwargs": {"c": "#aa0000", "capsize": 3, "marker": "."},
        "line_x": np.linspace(-2, 6, 513),
        "xticks": [-2, 0, 2, 4, 6],
        "xtick_labels": [-2, r"$E_{\mathrm{VBM}}$", 2, 4, 6],
        "vlines": [
            {"x": 0.5, "label": r"$E_{g}=0.5$", "c": "k", "ls": ":", "alpha": 0.5},
            {"x": 0.0, "label": r"$E_{VBM}$", "c": "k", "ls": "-.", "alpha": 0.5},
        ],
    },
    "fermi_centered/0": {
        "title": r"$E_{F} \pm 5 \mathrm{eV}$",
        "shortname": "Fermi",
        "line_kwargs": {"c": "#008000"},
        "fill_kwargs": {"color": "#008000", "alpha": 0.3},
        "errorbar_kwargs": {"c": "#008000", "capsize": 3, "marker": "."},
        "line_x": np.linspace(-5, 5, 513),
        "xticks": [-5, -2.5, 0, 2.5, 5],
        "xtick_labels": [-5, -2.5, r"$E_{F}$", 2.5, 5],
        "vlines": [
            {"x": -0.25, "label": r"$E_{g}=0.5$", "c": "k", "ls": ":", "alpha": 0.5},
            {"x": 0.25, "c": "k", "ls": ":", "alpha": 0.5},
            {"x": 0.0, "label": r"$E_{F}$", "c": "k", "ls": "-.", "alpha": 0.5},
        ],
    },
    "cbm_centered/0": {
        "title": r"$E_{\mathrm{CBM}} (-6,+2) \mathrm{eV}$",
        "shortname": "CBM",
        "line_kwargs": {"c": "#000080"},
        "fill_kwargs": {"color": "#000080", "alpha": 0.3},
        "errorbar_kwargs": {"c": "#000080", "capsize": 3, "marker": "."},
        "line_x": np.linspace(-6, 2, 513),
        "xticks": [-6, -4, -2, 0, 2],
        "xtick_labels": [-6, -4, -2, r"$E_{\mathrm{CBM}}$", 2],
        "vlines": [
            {"x": -0.5, "label": r"$E_{g}=0.5$", "c": "k", "ls": ":", "alpha": 0.5},
            {"x": 0.0, "label": r"$E_{CBM}$", "c": "k", "ls": "-.", "alpha": 0.5},
        ],
    },
    # "vbm_cbm_concatenated/0": {
    #     "title": r"$E_{F}^{\mathrm{scissor}} \pm 2 \mathrm{eV}$",
    #     "line_kwargs": {"c": "#000080"},
    #     "fill_kwargs": {"color": "#000080", "alpha": 0.3},
    #     "line_x": np.linspace(-2.15, 2.15, 512),
    #     "xticks": [-2, -1, -0.15, 0.15, 1, 2],
    #     "xtick_labels": [-2, -1, r"$E_{\mathrm{VBM}}$", r"$E_{\mathrm{CBM}}$", 1, 2],
    #     "xticks_kwargs": {"rotation": 90},
    # },
    # "vbm_cbm_concatenated/1": {
    #     "title": r"$E_{\mathrm{VBM}} \pm 2 \mathrm{eV} || E_{\mathrm{CBM}} \pm 2 \mathrm{eV}$",
    #     "line_kwargs": {"c": "k"},
    #     "fill_kwargs": {"color": "k", "alpha": 0.3},
    #     "errorbar_kwargs": {"c": "k", "capsize": 3, "marker": "."},
    #     "line_x": np.linspace(-4, 4, 514),
    #     "xticks": [-4, -2, 0, 2, 4],
    #     "xtick_labels": [-2, r"$E_{\mathrm{VBM}}$", r"$\pm$2", r"$E_{\mathrm{CBM}}$", 2],
    #     "vlines": [
    #         {"x": -1.5, "label": r"$E_{g}=0.5$", "c": "grey", "ls": ":", "alpha": 0.5},
    #         {"x": 1.5, "c": "grey", "ls": ":", "alpha": 0.5},
    #     ],
    # },
    "vbm_fermi_cbm_concatenated/0": {
        "title": r"$E_{\mathrm{VBM}} \pm 1 \mathrm{eV} || E_{F} \pm 1 \mathrm{eV} || E_{\mathrm{CBM}} \pm 1 \mathrm{eV}$",
        "shortname": "Concatenated",
        "line_kwargs": {"c": "k"},
        "fill_kwargs": {"color": "k", "alpha": 0.3},
        "errorbar_kwargs": {"c": "k", "capsize": 3, "marker": "."},
        "line_x": np.linspace(-3, 3, 513),
        "xticks": [-3, -2, -1, 0, 1, 2, 3],
        "xtick_labels": [-1, r"$E_{\mathrm{VBM}}$", r"$\pm$1", r"$E_{F}$", r"$\pm$1", r"$E_{\mathrm{CBM}}$", 1],
        "vlines": [
            {"x": -0.25, "label": r"$E_{g}=0.5$", "c": "k", "ls": ":", "alpha": 0.5},
            {"x": +0.25, "c": "k", "ls": ":", "alpha": 0.5},
            {"x": -1.5, "c": "k", "ls": ":", "alpha": 0.5},
            {"x": +1.5, "c": "k", "ls": ":", "alpha": 0.5},
            {"x": -2.0, "label": r"$E_{VBM}$", "c": "#aa0000", "ls": "-.", "alpha": 0.5},
            {"x": 0.0, "label": r"$E_{F}$", "c": "#008000", "ls": "-.", "alpha": 0.5},
            {"x": +2.0, "label": r"$E_{CBM}$", "c": "#000080", "ls": "-.", "alpha": 0.5},
        ],
    },
    # "soap/0": {
    #     "title": r"SOAP",
    #     "line_kwargs": {"c": "grey"},
    #     "fill_kwargs": {"color": "grey", "alpha": 0.3},
    #     "errorbar_kwargs": {"c": "grey", "capsize": 3, "marker": "."},
    #     "line_x": np.arange(105),
    #     # "xticks": [-3, -2 -1, 0, 1, 2, 3],
    #     # "xtick_labels": [-1, r"$E_{\mathrm{VBM}}$", r"$\pm$1", r"$E_{F}$", r"$\pm$1", r"$E_{\mathrm{CBM}}$", 1],
    #     # "vlines": [],
    # },
    "soap/2": {
        "title": r"SOAP ($n_{\mathrm{max}}=10,l_{\mathrm{max}}=9$)",
        "shortname": "SOAP",
        "line_kwargs": {"c": "grey"},
        "fill_kwargs": {"color": "grey", "alpha": 0.3},
        "errorbar_kwargs": {"c": "grey", "capsize": 3, "marker": "."},
        "line_x": np.arange(550),
        # "xticks": [-3, -2 -1, 0, 1, 2, 3],
        # "xtick_labels": [-1, r"$E_{\mathrm{VBM}}$", r"$\pm$1", r"$E_{F}$", r"$\pm$1", r"$E_{\mathrm{CBM}}$", 1],
        # "vlines": [],
    },
}


# %%
def load_data(data_dir: os.PathLike, database: str, calculation_type: str, feature_path: str):
    data_dir = pl.Path(data_dir)
    with open(data_dir / f"{database}_targets.npz", "rb") as fp:
        npz = np.load(fp)
        target_df = pd.DataFrame(data=dict(npz))

    with h5py.File(data_dir / f"{database}_features_{calculation_type}.h5", "r") as f:
        feature_df = pd.DataFrame(
            data={
                "material_id": f[feature_path]["material_id"][()].astype(str),
                "features": f[feature_path]["features"][()].tolist(),
            }
        )

    df = pd.merge(target_df, feature_df, on="material_id")
    X = np.array(df["features"].tolist())
    y = df["meets_tcm_criteria"].to_numpy()
    id_ = df["material_id"].to_numpy()

    # Create constant hold-out validation set of 10% of the data
    X, X_val, y, y_val, id_, id_val = skms.train_test_split(X, y, id_, test_size=0.1, random_state=9997)

    return (X, X_val, y, y_val, id_, id_val)


def load_models(model_dir: os.PathLike, feature_type: str, train_size: float) -> list[dict]:
    models = []
    for pkl_file in (pl.Path(model_dir) / feature_type).glob(f"{train_size:0.4f}*pkl"):
        models.append(
            {
                "model": joblib.load(pkl_file),
                "random_state": int(pkl_file.stem.split("_")[1]),
                "creation_time": pkl_file.stem.split("_")[2].strip("()"),
            }
        )
    return models


# %%
TRAIN_SIZE = 0.01

importances = {}
for feature_type in tqdm(FEATURE_KWARGS):
    if "soap" in feature_type:
        continue
    X, X_val, y, y_val, id_, id_val = load_data("../data", "mc3d", "single_shot", feature_type)
    models = load_models("../models", feature_type, TRAIN_SIZE)

    for model in tqdm(models, leave=False):
        importances = ski.permutation_importance(
            model["model"], X_val, y_val, scoring="balanced_accuracy", n_repeats=30, random_state=9997, n_jobs=12
        )
        with h5py.File("../data/mc3d_permutation_importance_single_shot.h5", "a") as f:
            g = f.create_group(f"{feature_type}/{model['random_state']}")
            for key, value in importances.items():
                g.create_dataset(key, data=value, compression="gzip", shuffle=True)

# %%
