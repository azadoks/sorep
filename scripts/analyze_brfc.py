# %%
import os
import pathlib as pl
import typing as ty

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sps
import seaborn as sns
import sklearn.metrics as skm

# %%
DATA_DIR = pl.Path("../data")
DATABASE = "mc3d"
CALCULATION_TYPE = "single_shot"
# %%
with h5py.File(DATA_DIR / DATABASE / "targets.h5", "r") as f:
    TARGET_DF = pd.DataFrame({k: v[()] for (k, v) in f.items()})

METRICS_DFS = {}
with h5py.File(DATA_DIR / DATABASE / "brfc_metrics.h5", "r") as f:
    g = f[CALCULATION_TYPE]
    for feature_name, feature_group in g.items():
        for feature_id, group in feature_group.items():
            METRICS_DFS[f"{feature_name}/{feature_id}"] = pd.DataFrame({k: v[()].tolist() for (k, v) in group.items()})

# IMPORTANCES = {}
# with h5py.File(f"../data/{DATABASE}_permutation_importance_{CALCULATION_TYPE}.h5", "r") as f:
#     for feature_name, feature_group in f.items():
#         for feature_instance, instance_group in feature_group.items():
#             IMPORTANCES[f"{feature_name}/{feature_instance}"] = {"values": [], "mean": [], "std": []}
#             for random_state, random_state_group in instance_group.items():
#                 IMPORTANCES[f"{feature_name}/{feature_instance}"]["values"].append(
#                     random_state_group["importances"][()].tolist()
#                 )
#                 IMPORTANCES[f"{feature_name}/{feature_instance}"]["mean"].append(
#                     random_state_group["importances_mean"][()].tolist()
#                 )
#                 IMPORTANCES[f"{feature_name}/{feature_instance}"]["std"].append(
#                     random_state_group["importances_std"][()].tolist()
#                 )

#     for key, value in IMPORTANCES.items():
#         IMPORTANCES[key]["values"] = np.concatenate(value["values"], axis=1).T
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
    "soap/0": {
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
def plot_yield(target_df, metrics, avg="mean", error="std"):

    with plt.style.context("../sorep.mplstyle"):
        fig, ax = plt.subplots(dpi=300)

        for feature_id, kwargs in FEATURE_KWARGS.items():
            gb = metrics[feature_id].groupby(by="train_size")
            if avg == "mean":
                yield_avg = gb["yield"].mean()
            elif avg == "median":
                yield_avg = gb["yield"].median()

            ax.plot(yield_avg.index, yield_avg, **kwargs["line_kwargs"], marker=".", label=kwargs["shortname"])
            if error == "std":
                yield_std = gb["yield"].std()
                ax.fill_between(yield_avg.index, yield_avg - yield_std, yield_avg + yield_std, **kwargs["fill_kwargs"])
            elif error == "minmax":
                yield_min = gb["yield"].min()
                yield_max = gb["yield"].max()
                ax.fill_between(yield_avg.index, yield_min, yield_max, **kwargs["fill_kwargs"])
            elif error == "iqr":
                yield_q1 = gb["yield"].quantile(0.25)
                yield_q3 = gb["yield"].quantile(0.75)
                ax.fill_between(yield_avg.index, yield_q1, yield_q3, **kwargs["fill_kwargs"])

        ax.axhline(
            target_df.meets_tcm_criteria.mean(),
            c="grey",
            ls=":",
            lw=1,
            label=r"$f_{\mathrm{TCM}}^{\mathrm{background}}$",
        )

        top_sec_ax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * 0.9 * target_df.shape[0], lambda x: x / 0.9 / target_df.shape[0])
        )
        top_sec_ax.set_xlabel(r"$N_{\mathrm{train}}$")

        ax.tick_params(axis="x", which="both", top=False)

        ax.set_xlim(7e-4, 3)
        ax.set_xscale("log")
        ax.set_xlabel(r"$f_{\mathrm{train}}$")
        ax.set_ylabel("Yield (TCM / calculation)")
        ax.legend(loc=[0, -0.35], ncols=3)

    return fig


fig = plot_yield(TARGET_DF, METRICS_DFS, avg="median", error="iqr")
fig.savefig("../plots/mc3d_tcm_yield.pdf")


# %%
def plot_balanced_accuracy(target_df, metrics, avg="mean", error="std"):

    with plt.style.context("../sorep.mplstyle"):
        fig, ax = plt.subplots(dpi=300)

        for feature_id, kwargs in FEATURE_KWARGS.items():
            gb = metrics[feature_id].groupby(by="train_size")

            if avg == "mean":
                balanced_accuracy_avg = gb["balanced_accuracy"].mean()
            elif avg == "median":
                balanced_accuracy_avg = gb["balanced_accuracy"].median()

            ax.plot(
                balanced_accuracy_avg.index,
                balanced_accuracy_avg,
                **kwargs["line_kwargs"],
                marker=".",
                label=kwargs["shortname"],
            )
            if error == "minmax":
                balanced_accuracy_min = gb["balanced_accuracy"].min()
                balanced_accuracy_max = gb["balanced_accuracy"].max()
                ax.fill_between(
                    balanced_accuracy_avg.index,
                    balanced_accuracy_min,
                    balanced_accuracy_max,
                    **kwargs["fill_kwargs"],
                )
            elif error == "std":
                balanced_accuracy_std = gb["balanced_accuracy"].std()
                ax.fill_between(
                    balanced_accuracy_avg.index,
                    balanced_accuracy_avg - balanced_accuracy_std,
                    balanced_accuracy_avg + balanced_accuracy_std,
                    **kwargs["fill_kwargs"],
                )
            elif error == "iqr":
                balanced_accuracy_q1 = gb["balanced_accuracy"].quantile(0.25)
                balanced_accuracy_q3 = gb["balanced_accuracy"].quantile(0.75)
                ax.fill_between(
                    balanced_accuracy_avg.index,
                    balanced_accuracy_q1,
                    balanced_accuracy_q3,
                    **kwargs["fill_kwargs"],
                )

        top_sec_ax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * 0.9 * target_df.shape[0], lambda x: x / 0.9 / target_df.shape[0])
        )
        top_sec_ax.set_xlabel(r"$N_{\mathrm{train}}$")

        ax.axhline(0, c="grey", ls=":", lw=1, label="Random")
        ax.tick_params(axis="x", which="both", top=False)
        ax.set_xscale("log")
        ax.set_xlabel(r"$f_{\mathrm{train}}$")
        ax.set_ylabel("Balanced accuracy")
        ax.legend(loc=[0, -0.35], ncols=3)

    return fig


fig = plot_balanced_accuracy(TARGET_DF, METRICS_DFS)
fig.savefig("../plots/mc3d_tcm_balanced_accuracy.pdf")
fig


# %%
def plot_feature_importances(target_df, metrics, train_size=0.5, axes=None, avg="mean", error="std"):
    with plt.style.context("../sorep.mplstyle"):
        if axes is None:
            fig, axes = plt.subplots(2, 2, dpi=300, figsize=(6, 4), sharey=True)
            axes = axes.flatten()
        else:
            fig = plt.gcf()

        for i, (ax, (feature_id, kwargs)) in enumerate(zip(axes, FEATURE_KWARGS.items())):
            if "soap" in feature_id:
                continue
            feature_metrics = metrics[feature_id]
            feature_importances = np.array(
                feature_metrics[feature_metrics.train_size == train_size]["impurity_feature_importances"].to_list()
            )
            if avg == "mean":
                feature_importances_avg = feature_importances.mean(axis=0)
            elif avg == "median":
                feature_importances_avg = np.median(feature_importances, axis=0)

            ax.set_title(kwargs["title"])
            ax.plot(kwargs["line_x"], feature_importances_avg, **kwargs["line_kwargs"])

            if error is not None and error != "none":
                if error == "std":
                    feature_importances_std = feature_importances.std(axis=0)
                    err_min = feature_importances_avg - feature_importances_std
                    err_max = feature_importances_avg + feature_importances_std
                elif error == "minmax":
                    err_min = feature_importances.min(axis=0)
                    err_max = feature_importances.max(axis=0)
                elif error == "iqr":
                    err_min = np.quantile(feature_importances, 0.25, axis=0)
                    err_max = np.quantile(feature_importances, 0.75, axis=0)

                ax.fill_between(kwargs["line_x"], err_min, err_max, **kwargs["fill_kwargs"])

            for vline in kwargs.get("vlines", []):
                ax.axvline(**vline)

            # ax.grid(visible=True, which="major", axis="both")
            if "xticks" in kwargs:
                ax.set_xticks(kwargs["xticks"], kwargs["xtick_labels"], **kwargs.get("xticks_kwargs", {}))
            if i % 2 == 0:
                ax.set_ylabel("Feature importance")
            if i > 1:
                ax.set_xlabel("Feature")
            ax.set_yticklabels([])
            ax.legend()
        fig.tight_layout()

    return fig


fig = plot_feature_importances(TARGET_DF, METRICS_DFS, train_size=0.5, avg="mean", error=None)
fig.savefig("../plots/mc3d_tcm_feature_importances.pdf", bbox_inches="tight")
fig
# %%
kwargs = {
    "x": "train_size",
    "y": "balanced_accuracy",
    "hue": "feature",
    "showfliers": False,
    "showmeans": False,
    "showcaps": True,
    # "whiskerprops": {"linestyle": ""},
    "boxprops": {"alpha": 0.75},
}

for key, value in METRICS_DFS.items():
    value["feature"] = key

df = pd.concat(
    [
        METRICS_DFS[key]
        for key in ["cbm_centered/0", "vbm_fermi_cbm_concatenated/0", "vbm_centered/0", "fermi_centered/0"]
    ]
)

fig, ax = plt.subplots(figsize=(20, 4))
sns.boxplot(ax=ax, data=df, **kwargs)
ax.set_xlim(-1, 28)
ax.set_xticks(np.linspace(0, 27, 8))
ax.set_xticklabels([f"{x*100:0.0f}" for x in np.linspace(0.001, 0.8, 8)])
ax.set_xlabel(r"$f_{\mathrm{train}}$ (%)")
# ax.set_ylabel(r"Yield (TCM / calculation)")
fig


# %%
def plot_confusion_matrix(feature_label, train_size):
    group = METRICS_DFS[feature_label].groupby("train_size").get_group(train_size)
    with plt.style.context("../sorep.mplstyle"):
        cm = np.array(group["confusion_matrix"].tolist()).mean(axis=0)
        _std = np.array(group["confusion_matrix"].tolist()).std(axis=0)

        neg = cm[0, 0] + cm[0, 1]
        pos = cm[1, 0] + cm[1, 1]
        cmr = np.array(
            [
                [cm[0, 0] / neg, cm[0, 1] / neg],
                [cm[1, 0] / pos, cm[1, 1] / pos],
            ]
        )
        cmr_std = np.array(
            [
                [_std[0, 0] / neg, _std[0, 1] / neg],
                [_std[1, 0] / pos, _std[1, 1] / pos],
            ]
        )
        thresh = (cm.max() + cm.min()) / 2

        fig, ax = plt.subplots()

        im_ = ax.imshow(cm, cmap="Grays")
        cbar = ax.figure.colorbar(im_)
        cmap_max = im_.cmap(0.0)
        cmap_min = im_.cmap(1.0)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:0.2f}$\\pm${_std[i, j]:0.2f}\n{cmr[i, j]:0.2f}$\\pm${cmr_std[i, j]:0.2f}",
                    ha="center",
                    va="center",
                    color=cmap_min if cm[i, j] < thresh else cmap_max,
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{FEATURE_KWARGS[feature_label]['title']} @ $f_{{\\mathrm{{train}}}}={train_size}$")
        return fig


fig = plot_confusion_matrix("vbm_centered/0", 0.01)
# %%
train_size = 0.01
cm_data = []
for feature_type, feature_info in FEATURE_KWARGS.items():
    df = METRICS_DFS[feature_type].groupby("train_size").get_group(train_size)
    ba_array = df["balanced_accuracy"].to_numpy()
    ba_mean = np.mean(ba_array)
    ba_std = np.std(ba_array)
    cm_array = np.array(df["confusion_matrix"].tolist())
    cm_mean = np.mean(cm_array, axis=0)
    cm_std = np.std(cm_array, axis=0)
    cm_median = np.median(cm_array, axis=0)
    pos_mean = cm_mean[1, 0] + cm_mean[1, 1]
    neg_mean = cm_mean[0, 0] + cm_mean[0, 1]
    cmr_mean = np.array([cm_mean[0, :] / neg_mean, cm_mean[1, :] / pos_mean])
    cmr_std = np.array([cm_std[0, :] / neg_mean, cm_std[1, :] / pos_mean])
    cm_data.append(
        {
            "SOREP": feature_info["title"],
            "True Negative": f"{cm_mean[0, 0]:0.0f}$\\pm${cm_std[0, 0]:0.0f}",
            "True Positive": f"{cm_mean[1, 1]:0.0f}$\\pm${cm_std[1, 1]:0.0f}",
            "False Negative": f"{cm_mean[1, 0]:0.0f}$\\pm${cm_std[1, 0]:0.0f}",
            "False Positive": f"{cm_mean[0, 1]:0.0f}$\\pm${cm_std[0, 1]:0.0f}",
            "Balanced accuracy": f"{ba_mean:0.2f}$\\pm${ba_std:0.2f}",
        }
    )
    cm_data.append(
        {
            "SOREP": feature_info["title"] + " Rate",
            "True Negative": f"({cmr_mean[0, 0]:0.3f}$\\pm${cmr_std[0, 0]:0.3f})",
            "True Positive": f"({cmr_mean[1, 1]:0.3f}$\\pm${cmr_std[1, 1]:0.3f})",
            "False Negative": f"({cmr_mean[1, 0]:0.3f}$\\pm${cmr_std[1, 0]:0.3f})",
            "False Positive": f"({cmr_mean[0, 1]:0.3f}$\\pm${cmr_std[0, 1]:0.3f})",
        }
    )
cm_df = pd.DataFrame(cm_data)
print(cm_df.to_latex(index=False))
# %%
FEATURE_ID = "fermi_centered/0"
fig, axes = plt.subplots(2, 1)
axes = axes.flatten()

imp_mean = np.mean(IMPORTANCES[FEATURE_ID]["mean"], axis=0)
imp_std = np.mean(IMPORTANCES[FEATURE_ID]["std"], axis=0)
axes[0].plot(FEATURE_KWARGS[FEATURE_ID]["line_x"], imp_mean, **FEATURE_KWARGS[FEATURE_ID]["line_kwargs"])
axes[0].fill_between(
    FEATURE_KWARGS[FEATURE_ID]["line_x"],
    imp_mean - imp_std,
    imp_mean + imp_std,
    **FEATURE_KWARGS[FEATURE_ID]["fill_kwargs"],
)

metrics = METRICS_DFS[FEATURE_ID].groupby("train_size").get_group(0.6)
imp_mean = np.array(metrics["impurity_feature_importances"].tolist()).mean(axis=0)
imp_std = np.array(metrics["impurity_feature_importances"].tolist()).std(axis=0)
axes[1].plot(FEATURE_KWARGS[FEATURE_ID]["line_x"], imp_mean, **FEATURE_KWARGS[FEATURE_ID]["line_kwargs"])
axes[1].fill_between(
    FEATURE_KWARGS[FEATURE_ID]["line_x"],
    imp_mean - imp_std,
    imp_mean + imp_std,
    **FEATURE_KWARGS[FEATURE_ID]["fill_kwargs"],
)

for ax in axes:
    for vline in FEATURE_KWARGS[FEATURE_ID]["vlines"]:
        ax.axvline(**vline)
    ax.set_xticks(FEATURE_KWARGS[FEATURE_ID]["xticks"])
    ax.set_xticklabels(FEATURE_KWARGS[FEATURE_ID]["xtick_labels"])
    ax.legend()
# %%
