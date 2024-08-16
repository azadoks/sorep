# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
import sklearn.metrics as skm
import sklearn.model_selection as skms
from sklearn.neural_network import MLPRegressor
import sklearn.pipeline as skpipe
import sklearn.preprocessing as skpre
from tqdm import tqdm
from xgboost import XGBRegressor

# %%
EVAL_ON = "test"
# %% Load data
targets = {}
with h5py.File("../data/mc3d/targets.h5", "r") as f:
    for key in ("train", "test"):
        targets[key] = pd.DataFrame({k: v[()] for (k, v) in f[key].items()})
        targets[key]["id"] = targets[key]["id"].apply(lambda x: x.decode("utf-8"))

features = {"train": {}, "test": {}}
with h5py.File("../data/mc3d/features.h5", "r") as f:
    for key in ("train", "test"):
        features[key]["vbm"] = f[f"{key}/single_shot/vbm_centered/0/features"][()]
        features[key]["fermi"] = f[f"{key}/single_shot/fermi_centered/0/features"][()]
        features[key]["cbm"] = f[f"{key}/single_shot/cbm_centered/0/features"][()]
        features[key]["concat"] = f[f"{key}/single_shot/vbm_fermi_cbm_concatenated/0/features"][()]
        features[key]["soap"] = f[f"{key}/single_shot/soap/0/features"][()]
        features[key]["vbm_cbm"] = np.hstack(
            [
                f[f"{key}/single_shot/vbm_centered/0/features"][()],
                f[f"{key}/single_shot/cbm_centered/0/features"][()],
            ]
        )
        features_plus_ss_gap = {}
        for k, v in features[key].items():
            features_plus_ss_gap[f"{k}_ss-gap"] = np.hstack(
                [v, np.expand_dims(targets[key]["single_shot_band_gap"], axis=0).T]
            )
        features[key].update(features_plus_ss_gap)

for key in ("train", "test"):
    is_insulating = np.array((targets[key]["scf_band_gap"] > 0).tolist())
    features[key] = {key: value[is_insulating] for key, value in features[key].items()}
    targets[key] = targets[key][is_insulating]
# %% Mean baseline
eval_targets = targets["train"]
mean_baseline = dict(
    rmsle=np.sqrt(
        skm.mean_squared_log_error(
            np.repeat(eval_targets.scf_band_gap.mean(), eval_targets.shape[0]), eval_targets.scf_band_gap
        )
    ),
    rmse=np.sqrt(
        skm.mean_squared_error(
            np.repeat(eval_targets.scf_band_gap.mean(), eval_targets.shape[0]), eval_targets.scf_band_gap
        )
    ),
    mae=skm.mean_absolute_error(
        np.repeat(eval_targets.scf_band_gap.mean(), eval_targets.shape[0]), eval_targets.scf_band_gap
    ),
)
print(f"RMSLE: {mean_baseline['rmsle']:.4f}, RMSE: {mean_baseline['rmse']:.4f}, MAE: {mean_baseline['mae']:.4f}")
# %% SS-gap Baseline
eval_targets = targets["train" if EVAL_ON == "val" else EVAL_ON]
ss_baseline = dict(
    rmsle=np.sqrt(skm.mean_squared_log_error(eval_targets.single_shot_band_gap, eval_targets.scf_band_gap)),
    rmse=np.sqrt(skm.mean_squared_error(eval_targets.single_shot_band_gap, eval_targets.scf_band_gap)),
    mae=skm.mean_absolute_error(eval_targets.single_shot_band_gap, eval_targets.scf_band_gap),
)
print(f"RMSLE: {ss_baseline['rmsle']:.4f}, RMSE: {ss_baseline['rmse']:.4f}, MAE: {ss_baseline['mae']:.4f}")


# %%
# svm = svm.SVR()
# svm.fit(ss.transform(X_train), y_train)

# krr = kernel_ridge.KernelRidge(kernel="rbf")
# krr.fit(ss.transform(X_train), y_train)

# mlp = MLPRegressor([256, 256, 256, 64, 16], activation="relu", solver="adam")
# mlp.fit(ss.transform(X_train), y_train)


def train_pipeline(X_train, y_train):
    pipe = skpipe.Pipeline(
        [
            ("scaler", skpre.StandardScaler()),
            ("regressor", XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.1, n_jobs=-1)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_pipeline(X_test, y_test):
    y_test_pred = pipe.predict(X_test)
    rmsle = np.sqrt(skm.mean_squared_log_error(y_test, np.clip(y_test_pred, 0.0, np.inf)))
    rmse = np.sqrt(skm.mean_squared_error(y_test, y_test_pred))
    mae = skm.mean_absolute_error(y_test, y_test_pred)
    return {"rmsle": rmsle, "rmse": rmse, "mae": mae}


# %%
results = []
train_sizes = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.4,
]
random_states = np.random.randint(0, 1000, 5)

pbar = tqdm(total=len(features["train"]) * len(random_states) * len(train_sizes))
for key, value in features["train"].items():
    for train_size in train_sizes:
        for i, random_state in enumerate(random_states):
            pbar.set_description(f"Feature: {key}, Train Size: {train_size}, Random State: {i}/{len(random_states)}")
            X_train, X_val, y_train, y_val, id_train, id_test = skms.train_test_split(
                value,
                targets["train"]["scf_band_gap"],
                targets["train"]["id"],
                train_size=train_size,
                random_state=random_state,
            )
            pipe = train_pipeline(X_train, y_train)
            if EVAL_ON == "val":
                model_results = evaluate_pipeline(X_val, y_val)
            if EVAL_ON == "test":
                model_results = evaluate_pipeline(features["test"][key], targets["test"]["scf_band_gap"])
            results.append({"feature": key, "train_size": train_size, "random_state": random_state, **model_results})
            pbar.update()

results_df = pd.DataFrame(results)
results_df.to_json("../data/mc3d/regression_results.json")

# %%
label_map = {
    "cbm": "CBM",
    "vbm": "VBM",
    "fermi": "Fermi",
    "concat": "VBM || EF || CBM",
    "soap": "SOAP",
    "vbm_cbm": "VBM || CBM",
}

color_map = {
    "cbm": "#000080",
    "vbm": "#aa0000",
    "fermi": "#008000",
    "concat": "k",
    "soap": "grey",
    "vbm_cbm": "#800080",
}

ls_map = {
    "cbm": ":",
    "vbm": "-",
    "fermi": "--",
    "concat": "-.",
    "soap": "-",
    "vbm_cbm": ":",
}

metric_key, metric_name = "mae", "MAE"
# metric_key, metric_name = "rmse", "RMSE"
# metric_key, match_name = "rmsle", "RMSLE"

results_df = pd.read_json("../data/mc3d/regression_results.json")
with plt.style.context("../sorep.mplstyle"):
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 7.5), sharey=False, sharex=True)

    for i, (metric_key, metric_name) in enumerate([("mae", "MAE"), ("rmse", "RMSE"), ("rmsle", "RMSLE")]):
        ax = axes[i]
        groups = results_df.groupby("feature")
        for feature, val in groups:
            val = val.groupby("train_size")
            mean = val[metric_key].mean()
            std = val[metric_key].std()
            if "ss-gap" not in feature:
                ax[0].errorbar(mean.index, mean, std, c=color_map[feature], ls=ls_map[feature])
            if "ss-gap" in feature:
                ax[1].errorbar(
                    mean.index,
                    mean,
                    std,
                    c=color_map["_".join(feature.split("_")[:-1])],
                    label=label_map["_".join(feature.split("_")[:-1])],
                    linestyle=ls_map["_".join(feature.split("_")[:-1])],
                )

        ax[0].axhline(ss_baseline[metric_key], label=r"$\hat{y}=E_{g}^{\mathrm{SS}}$", linestyle="-.", c="k")
        ax[0].axhline(mean_baseline[metric_key], label=r"$\hat{y}=\mu$", linestyle=":", c="k")
        ax[1].axhline(ss_baseline[metric_key], linestyle="-.", c="k")
        ax[1].axhline(mean_baseline[metric_key], linestyle=":", c="k")
        ax[0].set_ylim((0.75 * ax[0].get_ylim()[0], ax[0].get_ylim()[1]))
        ax[1].set_ylim(ax[0].get_ylim())

        ax[0].set_ylabel(f"{metric_name} ± σ²")
        ax[1].set_yticklabels([])

        if i == 0:
            ax[1].legend(ncols=2)
            ax[0].legend()
            ax[0].set_title("Standard")
            ax[1].set_title("With single-shot gap as feature")
        if i == 1:
            for a in ax:
                a.set_xlabel("Train fraction")

fig.savefig("../plots/mc3d_scf_band_gap_regression.pdf")
fig
# %%
mean = (
    results_df[results_df["train_size"] == 0.2].drop(columns=["train_size", "random_state"]).groupby("feature").mean()
)
std = results_df[results_df["train_size"] == 0.2].drop(columns=["train_size", "random_state"]).groupby("feature").std()

copy = mean.copy()
for i in range(mean.shape[0]):
    for j in range(mean.shape[1]):
        copy.iloc[i, j] = f"{mean.iloc[i,j]:0.4f} $\pm$ {std.iloc[i,j]:0.4f}"

copy = mean.copy()
for i in range(mean.shape[0]):
    for j in range(mean.shape[1]):
        copy.iloc[i, j] = mean.iloc[i, j] + std.iloc[i, j]
# %%
