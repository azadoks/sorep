# %%
from functools import partial
from multiprocessing import Pool
import os
import pathlib as pl

import imblearn.ensemble as imbe
import numpy as np
import pandas as pd
import sklearn.dummy as skd
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.pipeline as skpl
import sklearn.preprocessing as skpp
from tqdm import tqdm


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


def train_rfc(X_train, y_train, random_state):
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
        feature_importances = model.named_steps["brfc"].feature_importances_
    except AttributeError:
        feature_importances = None
    metrics = {
        "y_pred": y_test_pred,
        "balanced_accuracy": skm.balanced_accuracy_score(y_test, y_test_pred, adjusted=True),
        "f1": skm.f1_score(y_test, y_test_pred),
        "precision": skm.precision_score(y_test, y_test_pred),
        "recall": skm.recall_score(y_test, y_test_pred),
        "roc_auc": skm.roc_auc_score(y_test, y_test_pred),
        "confusion_matrix": skm.confusion_matrix(y_test, y_test_pred),
        "yield": yield_score(y_test, y_test_pred, y_train),
        "feature_importances": feature_importances,
    }
    return metrics


def load_data(data_dir: os.PathLike, database: str, calculation_type: str, feature_id: str):
    with open(data_dir / f"{database}_targets.npz", "rb") as fp:
        npz = np.load(fp)
        target_df = pd.DataFrame(data=dict(npz))

    with open(data_dir / f"{database}_features_{calculation_type}_{feature_id}.npz", "rb") as fp:
        npz = np.load(fp)
        feature_df = pd.DataFrame(data={k: v.tolist() for (k, v) in npz.items()})

    df = pd.merge(target_df, feature_df, on="material_id")
    X = np.array(df["features"].tolist())
    y = df["meets_tcm_criteria"].to_numpy()

    # Create constant hold-out validation set of 10% of the data
    X, X_val, y, y_val = skms.train_test_split(X, y, test_size=0.1, random_state=9997)

    return (X, X_val, y, y_val)


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


def train_evaluate(feature_id, train_sizes, random_states):
    results = []
    X, _, y, _ = load_data(pl.Path("../data"), DATASET, CALCULATION_TYPE, feature_id)

    for train_size, random_state in zip(train_sizes, random_states):
        X_train, X_test, y_train, y_test = skms.train_test_split(X, y, train_size=train_size, random_state=random_state)

        model = train_rfc(X_train, y_train, random_state)
        metrics = evaluate_model(model, X_test, y_test, X_train, y_train)

        dummy = train_dummy(X_train, y_train)
        metrics_dummy = evaluate_model(dummy, X_test, y_test, X_train, y_train)

        results.append(
            {
                "train_size": train_size,
                "train_number": X_train.shape[0],
                "random_state": random_state,
                **metrics,
                **{f"dummy_{key}": value for (key, value) in metrics_dummy.items()},
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_json(pl.Path(DATA_DIR) / f"{DATASET}_metrics_{CALCULATION_TYPE}_{feature_id}.json")


# %% Train models
DATA_DIR = "../data/"
DATASET = "mc3d"
CALCULATION_TYPE = "single_shot"
FEATURE_IDS = [
    "dos_cbm_centered_gauss_0.05_-6.00_2.00_512",
    "dos_fermi_centered_gauss_0.05_-5.00_5.00_512",
    "dos_vbm_centered_gauss_0.05_-2.00_6.00_512",
    "dos_fermi_scissor_gauss_0.05_-2.00_0.15_-0.15_2.00_512",
    "dos_fermi_scissor_gauss_0.05_-2.00_2.00_-2.00_2.00_512",
]
# TRAIN_SIZES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
    _, _, y, _ = load_data(pl.Path("../data"), DATASET, CALCULATION_TYPE, FEATURE_IDS[0])
    random_states = get_random_states(y, train_sizes, min_populations={0: 1, 1: 1})

    _train_evaluate = partial(train_evaluate, train_sizes=train_sizes, random_states=random_states)

    feature_id_pbar = tqdm(FEATURE_IDS, ncols=120)
    with Pool(processes=12, maxtasksperchild=1) as p:
        p.map(_train_evaluate, feature_id_pbar)


# %%
if __name__ == "__main__":
    main()
