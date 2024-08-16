# %%
from datetime import datetime
from functools import partial
from multiprocessing import Pool
import os
import pathlib as pl
import platform

import h5py
import imblearn.ensemble as imbe
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
def grid_cross_validate(X_train, y_train):
    pipe = skpl.Pipeline(
        [
            ("scaler", skpp.StandardScaler()),
            (
                "gscv",
                skms.GridSearchCV(
                    imbe.BalancedRandomForestClassifier(),
                    param_grid={
                        "n_estimators": [500],
                        "class_weight": ["balanced", "balanced_subsample"],
                        "sampling_strategy": ["all", "majority", 0.5],
                        "replacement": [True, False],
                        "bootstrap": [True, False],
                        "ccp_alpha": [0e-0, 1e-3, 5e-3, 1e-2],
                        "random_state": [0],
                    },
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipe.fit(X_train, y_train)


def train_brfc(X_train, y_train, random_state):
    pipe = skpl.Pipeline(
        [
            ("scaler", skpp.StandardScaler()),
            (
                "brfc",
                # Parameters from 5-fold gridded cross-validation
                # with train_size=0.6
                imbe.BalancedRandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced_subsample",
                    sampling_strategy=0.5,
                    replacement=True,
                    bootstrap=True,
                    ccp_alpha=0.0,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return pipe.fit(X_train, y_train)


def train_dummy(X_train, y_train):
    dummy = skd.DummyClassifier(strategy="stratified")
    return dummy.fit(X_train, y_train)


def yield_score(y_test, y_pred, y_train):
    n_train = len(y_train)
    n_found_train = np.sum(y_train)
    _, fp, _, tp = skm.confusion_matrix(y_test, y_pred).ravel()
    return (n_found_train + tp) / (n_train + tp + fp)


def evaluate_model(model, X_test, y_test, X_train, y_train):
    y_test_pred = model.predict(X_test)
    try:
        impurity_feature_importances = model.named_steps["brfc"].feature_importances_
    except AttributeError:
        impurity_feature_importances = None
    # permutation_feature_importances = ski.permutation_importance(
    #     model, X_test, y_test, scoring="balanced_accuracy", n_repeats=10, random_state=0
    # )
    metrics = {
        "balanced_accuracy": skm.balanced_accuracy_score(y_test, y_test_pred, adjusted=True),
        "f1": skm.f1_score(y_test, y_test_pred),
        "precision": skm.precision_score(y_test, y_test_pred),
        "recall": skm.recall_score(y_test, y_test_pred),
        "roc_auc": skm.roc_auc_score(y_test, y_test_pred),
        "confusion_matrix": skm.confusion_matrix(y_test, y_test_pred),
        "yield": yield_score(y_test, y_test_pred, y_train),
        "impurity_feature_importances": impurity_feature_importances,
        # "permutation_feature_importances": permutation_feature_importances["importances_mean"],
    }
    return metrics


def load_data(data_dir: os.PathLike, database: str, calculation_type: str, feature_path: str):
    with h5py.File(data_dir / database / "targets.h5", "r") as f:
        target_df = pd.DataFrame(
            data={
                "id": f["train"]["id"][()].astype(str),
                "meets_tcm_criteria": f["train"]["meets_tcm_criteria"][()],
            }
        )

    with h5py.File(data_dir / database / "features.h5", "r") as f:
        feature_df = pd.DataFrame(
            data={
                "id": f[f"train/{calculation_type}/{feature_path}"]["id"][()].astype(str),
                "features": f[f"train/{calculation_type}/{feature_path}"]["features"][()].tolist(),
            }
        )

    df = pd.merge(target_df, feature_df, on="id")
    X = np.array(df["features"].tolist())
    y = df["meets_tcm_criteria"].to_numpy()
    id_ = df["id"].to_numpy()

    return X, y, id_


def get_random_states(labels, train_sizes, min_populations=None, seed=9997):
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))

    if min_populations is None:
        min_populations = {k: 0 for k in np.unique(y)}

    random_states = []
    for train_size in train_sizes:
        do_generate = True
        while do_generate:
            random_state = rs.randint(low=0, high=2**32 - 1)
            train, test = skms.train_test_split(labels, train_size=train_size, random_state=random_state)
            if all(np.sum(train == k) >= v for k, v in min_populations.items()) and all(
                np.sum(test == k) >= v for k, v in min_populations.items()
            ):
                do_generate = False
                random_states.append(random_state)

    return random_states


def train_evaluate(feature_path, train_sizes, random_states):
    results = []
    X, y, id_ = load_data(pl.Path(DATA_DIR), DATABASE, CALCULATION_TYPE, feature_path)

    for train_size, random_state in zip(train_sizes, random_states):
        X_train, X_test, y_train, y_test, id_train, id_test = skms.train_test_split(
            X, y, id_, train_size=train_size, random_state=random_state
        )

        model = train_brfc(X_train, y_train, random_state)
        metrics = evaluate_model(model, X_test, y_test, X_train, y_train)

        results.append(
            {
                "train_size": train_size,
                "train_number": X_train.shape[0],
                "random_state": random_state,
                **metrics,
            }
        )
        creation_time = datetime.isoformat(datetime.now())
        path = pl.Path(
            f"../models/{CALCULATION_TYPE}/{feature_path}/{train_size:.4f}_{random_state}_{creation_time}).pkl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fp:
            dump(model, fp, protocol=5)
        print(f"{feature_path:<40s} {train_size:8.4f} {metrics['balanced_accuracy']: 8.4f} {metrics['yield']: 8.4f}")

    results = pd.DataFrame(results).to_dict(orient="list")
    return results, creation_time


# %% Train models
DATA_DIR = "../data/"
DATABASE = "mc3d"
CALCULATION_TYPE = "scf"
PARALLEL = True
FEATURE_PATHS = [
    "fermi_centered/0",  # -5:513:+5
    "vbm_centered/0",  # -2:513:+6
    "cbm_centered/0",  # -6:513:+2
    "vbm_cbm_concatenated/0",  # -2:257:+3σ, -3σ:257:+2
    "vbm_cbm_concatenated/1",  # -2:257:+2, -2:257:+2
    "vbm_fermi_cbm_concatenated/0",  # -1:171:+1, -1:171:+1, -1:171:+1
    "soap/0",  # No species (~550 features, n=10, l=9, r=6)
]
TRAIN_SIZES = [
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
]
N_REPEATS = 30


def main():
    # Get constant random states for each train size and repeat so that the splits are the same for all feature types,
    # assuming the data are in the same order across feature types
    train_sizes = np.repeat(TRAIN_SIZES, N_REPEATS)
    _, y, _ = load_data(pl.Path(DATA_DIR), DATABASE, CALCULATION_TYPE, FEATURE_PATHS[0])
    random_states = get_random_states(y, train_sizes, min_populations={0: 1, 1: 1})

    _train_evaluate = partial(train_evaluate, train_sizes=train_sizes, random_states=random_states)

    feature_path_pbar = tqdm(FEATURE_PATHS, ncols=120)
    if PARALLEL:
        with Pool(processes=14, maxtasksperchild=1) as pool:
            results = pool.map(_train_evaluate, feature_path_pbar)
    else:
        results = []
        for feature_path in feature_path_pbar:
            results.append(_train_evaluate(feature_path))

    print("Saving results")
    with h5py.File(pl.Path(DATA_DIR) / DATABASE / "metrics_val.h5", "a") as f:
        feature_paths = [f"{CALCULATION_TYPE}/{feature_path}" for feature_path in FEATURE_PATHS]
        for feature_path, feature_results in zip(feature_paths, results):
            feature_results, creation_time = feature_results
            if feature_path in f:
                del f[feature_path]
            g = f.create_group(feature_path)
            g.attrs["ctime"] = creation_time
            for key, val in feature_results.items():
                g.create_dataset(key, data=val)


# %%
if __name__ == "__main__":
    main()

# %%
