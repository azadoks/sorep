# %%
import os
import pathlib as pl
import typing as ty

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
DATA_DIR = pl.Path("../data")
DATABASE = "mc3d"
CALCULATION_TYPE = "scf"
TRAIN_TEST = "test"

N_MATERIALS = 21691
N_TCM = 1909
F_TCM_BACKGROUND = N_TCM / N_MATERIALS
# %%
with h5py.File(DATA_DIR / DATABASE / "targets.h5", "r") as f:
    TARGET_DF = pd.DataFrame({k: v[()] for (k, v) in f["test"].items()})
    TARGET_DF["id"] = TARGET_DF["id"].apply(lambda x: x.decode("utf-8"))

METRICS_DFS = {}
with h5py.File(DATA_DIR / DATABASE / f"metrics_{TRAIN_TEST}.h5", "r") as f:
    g = f[CALCULATION_TYPE]
    for feature_name, feature_group in g.items():
        for feature_id, group in feature_group.items():
            df = pd.DataFrame({k: v[()].tolist() for (k, v) in group.items()})
            df["true_positive"] = df["confusion_matrix"].apply(lambda x: x[1][1])
            df["false_positive"] = df["confusion_matrix"].apply(lambda x: x[0][1])
            df["true_negative"] = df["confusion_matrix"].apply(lambda x: x[0][0])
            df["false_negative"] = df["confusion_matrix"].apply(lambda x: x[1][0])
            df["true_positive_rate"] = df["true_positive"] / (df["true_positive"] + df["false_negative"])
            df["false_positive_rate"] = df["false_positive"] / (df["false_positive"] + df["true_negative"])
            df["train_number"] = df["train_size"] * TARGET_DF.shape[0]
            df["yield"] = (df["true_positive"] + df["train_number"] * F_TCM_BACKGROUND) / (
                df["false_positive"] + df["true_positive"] + df["train_number"]
            )
            df["speedup"] = df["yield"] / F_TCM_BACKGROUND
            df["f_found"] = (df["train_number"] * F_TCM_BACKGROUND + df["true_positive_rate"] * N_TCM) / N_TCM
            METRICS_DFS[f"{feature_name}/{feature_id}"] = df
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
            {"x": 0.0, "label": r"$E_{VBM}$", "c": "#aa0000", "ls": "-.", "alpha": 0.5},
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
            {"x": 0.0, "label": r"$E_{F}$", "c": "#008000", "ls": "-.", "alpha": 0.5},
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
            {"x": 0.0, "label": r"$E_{CBM}$", "c": "#000080", "ls": "-.", "alpha": 0.5},
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
    },
}


# %%
def plot_yield(target_df, metrics, avg="mean", error="std"):
    metric_key = "yield"
    with plt.style.context("../sorep.mplstyle"):
        fig, ax = plt.subplots(dpi=300)

        for feature_id, kwargs in FEATURE_KWARGS.items():
            gb = metrics[feature_id].groupby(by="train_size")
            if avg == "mean":
                yield_avg = gb[metric_key].mean()
            elif avg == "median":
                yield_avg = gb[metric_key].median()

            ax.plot(yield_avg.index, yield_avg, **kwargs["line_kwargs"], marker=".", label=kwargs["shortname"])
            if error == "std":
                yield_std = gb[metric_key].std()
                ax.fill_between(yield_avg.index, yield_avg - yield_std, yield_avg + yield_std, **kwargs["fill_kwargs"])
            elif error == "minmax":
                yield_min = gb[metric_key].min()
                yield_max = gb[metric_key].max()
                ax.fill_between(yield_avg.index, yield_min, yield_max, **kwargs["fill_kwargs"])
            elif error == "iqr":
                yield_q1 = gb[metric_key].quantile(0.25)
                yield_q3 = gb[metric_key].quantile(0.75)
                ax.fill_between(yield_avg.index, yield_q1, yield_q3, **kwargs["fill_kwargs"])

        ax.axhline(
            target_df.meets_tcm_criteria.mean(),
            c="grey",
            ls=":",
            lw=1,
            label=r"$f_{\mathrm{TCM}}^{\mathrm{background}}$",
        )

        top_sec_ax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * 0.9 * N_MATERIALS, lambda x: x / 0.9 / N_MATERIALS)
        )
        top_sec_ax.set_xlabel(r"$N_{\mathrm{train}}$")

        ax.tick_params(axis="x", which="both", top=False)

        # ax.set_xlim(7e-4, 1.2)
        ax.set_xscale("log")
        ax.set_xlabel(r"$f_{\mathrm{train}}$")
        ax.set_ylabel("Yield (TCM / calculation)")
        ax.legend(loc=[0, -0.35], ncols=3)

    return fig


fig = plot_yield(TARGET_DF, METRICS_DFS, avg="mean", error="std")
fig.savefig(f"../plots/mc3d_{TRAIN_TEST}_{CALCULATION_TYPE}_tcm_yield.pdf", bbox_inches="tight")
plt.close(fig)


# %%
def plot_metric(target_df, metrics, metric_key, metric_name, avg="mean", error="std", ax=None):
    with plt.style.context("../sorep.mplstyle"):
        if ax is None:
            fig, ax = plt.subplots(dpi=300)
        else:
            fig = ax.get_figure()

        for feature_id, kwargs in FEATURE_KWARGS.items():
            gb = metrics[feature_id].groupby(by="train_size")

            if avg == "mean":
                balanced_accuracy_avg = gb[metric_key].mean()
            elif avg == "median":
                balanced_accuracy_avg = gb[metric_key].median()

            ax.plot(
                balanced_accuracy_avg.index,
                balanced_accuracy_avg,
                **kwargs["line_kwargs"],
                marker=".",
                label=kwargs["shortname"],
            )
            if error == "minmax":
                balanced_accuracy_min = gb[metric_key].min()
                balanced_accuracy_max = gb[metric_key].max()
                ax.fill_between(
                    balanced_accuracy_avg.index,
                    balanced_accuracy_min,
                    balanced_accuracy_max,
                    **kwargs["fill_kwargs"],
                )
            elif error == "std":
                balanced_accuracy_std = gb[metric_key].std()
                ax.fill_between(
                    balanced_accuracy_avg.index,
                    balanced_accuracy_avg - balanced_accuracy_std,
                    balanced_accuracy_avg + balanced_accuracy_std,
                    **kwargs["fill_kwargs"],
                )
            elif error == "iqr":
                balanced_accuracy_q1 = gb[metric_key].quantile(0.25)
                balanced_accuracy_q3 = gb[metric_key].quantile(0.75)
                ax.fill_between(
                    balanced_accuracy_avg.index,
                    balanced_accuracy_q1,
                    balanced_accuracy_q3,
                    **kwargs["fill_kwargs"],
                )

        top_sec_ax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * 0.9 * N_MATERIALS, lambda x: x / 0.9 / N_MATERIALS)
        )
        top_sec_ax.set_xlabel(r"$N_{\mathrm{train}}$")

        ax.tick_params(axis="x", which="both", top=False)
        ax.set_xscale("log")
        ax.set_xlabel(r"$f_{\mathrm{train}}$")
        ax.set_ylabel(metric_name)
        ax.legend(loc=[0, -0.35], ncols=3)

    return fig


for metric_key, metric_name in [
    ("f1", r"$F_{1}$ score"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("speedup", "Speedup"),
    ("roc_auc_weighted", "ROC-AUC (weighted)"),
    ("precision", "Precision"),
    ("recall", "Recall"),
]:
    fig = plot_metric(TARGET_DF, METRICS_DFS, metric_key=metric_key, metric_name=metric_name, avg="mean", error="std")
    fig.savefig(f"../plots/mc3d_{TRAIN_TEST}_{CALCULATION_TYPE}_tcm_{metric_key}.pdf", bbox_inches="tight")
    plt.close(fig)

# %%
with plt.style.context("../sorep.mplstyle"):
    fig, ax = plt.subplots(dpi=300)
ax.plot(np.logspace(-3, 0, 100), np.logspace(-3, 0, 100), c="k", ls=":", lw=1, label="HT")
fig = plot_metric(
    TARGET_DF,
    METRICS_DFS,
    metric_key="f_found",
    metric_name=r"$f_{\mathrm{TCM}}$ found",
    avg="mean",
    error="std",
    ax=ax,
)
fig.savefig(f"../plots/mc3d_{TRAIN_TEST}_{CALCULATION_TYPE}_tcm_f_found.pdf", bbox_inches="tight")
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
                ax.set_ylabel("MDI feature importance")
            if i > 1:
                ax.set_xlabel("Feature")
            ax.set_yticklabels([])
            ax.legend(frameon=True)
        fig.tight_layout()

    return fig


fig = plot_feature_importances(TARGET_DF, METRICS_DFS, train_size=0.5, avg="mean", error=None)
fig.savefig(f"../plots/mc3d_{CALCULATION_TYPE}_tcm_feature_importances.pdf", bbox_inches="tight")
fig
# %%
import sorep


def plot_example_feature_importance(id_, target_df, metrics, train_size=0.5, avg="mean", error="std"):
    with h5py.File("../data/mc3d/materials.h5", "r") as f:
        material = sorep.MaterialData.from_hdf(f[id_]["single_shot"])

    targets = target_df[target_df.id == id_].iloc[0]

    ef = material.bands.fermi_energy
    energies = np.linspace(ef - 5, ef + 5, 4096)
    dos = material.bands.compute_smeared_dos(energies, "gauss", 0.05).sum(axis=0)

    with plt.style.context("../sorep.mplstyle"):
        fig, ax = plt.subplots()
        tax = ax.twinx()
        ax.plot(energies - ef, dos, c="k", alpha=0.25, linestyle="-")
        ax.fill_between(energies - ef, dos, 0, color="k", alpha=0.2)
        # ax.set_title(
        #     " | ".join(
        #         [
        #             sorep.prettify.latex_chemical_formula(material.atoms.info["formula_hill"]),
        #             f"$E_g={targets['band_gap']:.2f}$ eV",
        #             f"$m^*_e={targets['electron_effective_mass']:.2f}$",
        #             f"$m^*_h={targets['hole_effective_mass']:.2f}$",
        #         ]
        #     )
        # )

        feature_metrics = metrics["vbm_centered/0"]
        tax.plot(
            FEATURE_KWARGS["vbm_centered/0"]["line_x"] + material.bands.vbm - ef,
            np.array(
                feature_metrics[feature_metrics.train_size == train_size]["impurity_feature_importances"].to_list()
            ).mean(axis=0),
            **FEATURE_KWARGS["vbm_centered/0"]["line_kwargs"],
        )

        feature_metrics = metrics["fermi_centered/0"]
        tax.plot(
            FEATURE_KWARGS["fermi_centered/0"]["line_x"] + material.bands.fermi_energy - ef,
            np.array(
                feature_metrics[feature_metrics.train_size == train_size]["impurity_feature_importances"].to_list()
            ).mean(axis=0),
            **FEATURE_KWARGS["fermi_centered/0"]["line_kwargs"],
        )

        feature_metrics = metrics["cbm_centered/0"]
        tax.plot(
            FEATURE_KWARGS["cbm_centered/0"]["line_x"] + material.bands.cbm - ef,
            np.array(
                feature_metrics[feature_metrics.train_size == train_size]["impurity_feature_importances"].to_list()
            ).mean(axis=0),
            **FEATURE_KWARGS["cbm_centered/0"]["line_kwargs"],
        )

        feature_metrics = metrics["vbm_fermi_cbm_concatenated/0"]
        tax.plot(
            np.linspace(-1, 1, 171) + material.bands.vbm - ef,
            np.array(
                feature_metrics[feature_metrics.train_size == train_size]["impurity_feature_importances"].to_list()
            )[:, :171].mean(axis=0),
            **FEATURE_KWARGS["vbm_fermi_cbm_concatenated/0"]["line_kwargs"],
        )
        tax.plot(
            np.linspace(-1, 1, 171) + material.bands.fermi_energy - ef,
            np.array(
                feature_metrics[feature_metrics.train_size == train_size]["impurity_feature_importances"].to_list()
            )[:, 171:342].mean(axis=0),
            **FEATURE_KWARGS["vbm_fermi_cbm_concatenated/0"]["line_kwargs"],
        )
        tax.plot(
            np.linspace(-1, 1, 171) + material.bands.cbm - ef,
            np.array(
                feature_metrics[feature_metrics.train_size == train_size]["impurity_feature_importances"].to_list()
            )[:, 342:].mean(axis=0),
            **FEATURE_KWARGS["vbm_fermi_cbm_concatenated/0"]["line_kwargs"],
        )

        ax.axvline(material.bands.vbm - ef + 0.5, c="k", ls=":", alpha=0.5, label=r"$E_{g}=0.5$")
        ax.axvline(material.bands.fermi_energy - ef - 0.25, c="k", ls=":", alpha=0.5)
        ax.axvline(material.bands.fermi_energy - ef + 0.25, c="k", ls=":", alpha=0.5)
        ax.axvline(material.bands.cbm - ef - 0.5, c="k", ls=":", alpha=0.5)

        ax.axvline(material.bands.vbm - ef, c="#aa0000", ls="-.", alpha=0.5, label=r"$E_{\mathrm{VBM}}$")
        ax.axvline(material.bands.fermi_energy - ef, c="#008000", ls="-.", alpha=0.5, label=r"$E_{F}$")
        ax.axvline(material.bands.cbm - ef, c="#000080", ls="-.", alpha=0.5, label=r"$E_{\mathrm{CBM}}$")

        ax.set_yticklabels([])
        ax.set_ylabel(r"$\mathrm{DOS}(E) (arb. u.)$")

        ax.set_xticks([-4, material.bands.vbm - ef, -2, 0, 2, material.bands.cbm - ef, 4])
        ax.set_xticklabels([-4, r"$E_{\mathrm{VBM}}$", -2, r"$E_{F}$", 2, r"$E_{\mathrm{CBM}}$", 4])

        tax.set_yticklabels([])
        tax.set_ylabel("MDI feature importance")

        fig.legend(loc="upper center", ncol=5)

    return fig


fig = plot_example_feature_importance("mc3d-21923", TARGET_DF, METRICS_DFS, train_size=0.5)
fig.savefig("../plots/mc3d_tcm_feature_importances_example.pdf")
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
fig
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
