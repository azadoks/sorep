# %%
import os
import pathlib as pl
import typing as ty

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
DATABASE = "mc3d"
CALCULATION_TYPE = "single_shot"

FEATURE_KWARGS = {
    "dos_vbm_centered_gauss_0.05_-2.00_6.00_512": {
        "title": r"$E_{\mathrm{VBM}}(-2,+6) \mathrm{eV}$",
        "line_kwargs": {"c": "#aa0000"},
        "fill_kwargs": {"color": "#aa0000", "alpha": 0.3},
        "errorbar_kwargs": {"c": "#aa0000", "capsize": 3, "marker": "."},
        "line_x": np.linspace(-2, 6, 512),
        "xticks": [-2, 0, 2, 4, 6],
        "xtick_labels": [-2, r"$E_{\mathrm{VBM}}$", 2, 4, 6],
        "vlines": [{"x": 0.5, "label": r"$E_{g}=0.5$", "c": "grey", "ls": ":"}],
    },
    "dos_fermi_centered_gauss_0.05_-5.00_5.00_512": {
        "title": r"$E_{F} \pm 5 \mathrm{eV}$",
        "line_kwargs": {"c": "#008000"},
        "fill_kwargs": {"color": "#008000", "alpha": 0.3},
        "errorbar_kwargs": {"c": "#008000", "capsize": 3, "marker": "."},
        "line_x": np.linspace(-5, 5, 512),
        "xticks": [-5, -2.5, 0, 2.5, 5],
        "xtick_labels": [-5, -2.5, r"$E_{F}$", 2.5, 5],
        "vlines": [
            {"x": -0.25, "label": r"$E_{g}=0.5$", "c": "grey", "ls": ":"},
            {"x": 0.25, "c": "grey", "ls": ":"},
        ],
    },
    "dos_cbm_centered_gauss_0.05_-6.00_2.00_512": {
        "title": r"$E_{\mathrm{CBM}} (-6,+2) \mathrm{eV}$",
        "line_kwargs": {
            "c": "k",
        },
        "fill_kwargs": {
            "color": "k",
            "alpha": 0.3,
        },
        "errorbar_kwargs": {"c": "k", "capsize": 3, "marker": "."},
        "line_x": np.linspace(-6, 2, 512),
        "xticks": [-6, -4, -2, 0, 2],
        "xtick_labels": [-6, -4, -2, r"$E_{\mathrm{CBM}}$", 2],
        "vlines": [{"x": -0.5, "label": r"$E_{g}=0.5$", "c": "grey", "ls": ":"}],
    },
    # "dos_fermi_scissor_gauss_0.05_-2.00_0.15_-0.15_2.00_512": {
    #     "title": r"$E_{F}^{\mathrm{scissor}} \pm 2 \mathrm{eV}$",
    #     "line_kwargs": {"c": "#000080"},
    #     "fill_kwargs": {"color": "#000080", "alpha": 0.3},
    #     "line_x": np.linspace(-2.15, 2.15, 512),
    #     "xticks": [-2, -1, -0.15, 0.15, 1, 2],
    #     "xtick_labels": [-2, -1, r"$E_{\mathrm{VBM}}$", r"$E_{\mathrm{CBM}}$", 1, 2],
    #     "xticks_kwargs": {"rotation": 90},
    # },
    "dos_fermi_scissor_gauss_0.05_-2.00_2.00_-2.00_2.00_512": {
        "title": r"$E_{\mathrm{VBM}} \pm 2 \mathrm{eV} || E_{\mathrm{CBM}} \pm 2 \mathrm{eV}$",
        "line_kwargs": {"c": "#000080"},
        "fill_kwargs": {"color": "#000080", "alpha": 0.3},
        "errorbar_kwargs": {"c": "#000080", "capsize": 3, "marker": "."},
        "line_x": np.linspace(-4, 4, 512),
        "xticks": [-4, -2, 0, 2, 4],
        "xtick_labels": [-2, r"$E_{\mathrm{VBM}}$", r"$\pm$2", r"$E_{\mathrm{CBM}}$", 2],
        "vlines": [
            {"x": -1.5, "label": r"$E_{g}=0.5$", "c": "grey", "ls": ":"},
            {"x": 1.5, "c": "grey", "ls": ":"},
        ],
    },
}


# %%
def plot_yield(database, calculation_type):
    target_df = pd.DataFrame({k: v.tolist() for (k, v) in np.load(pl.Path(f"../data/{database}_targets.npz")).items()})

    with plt.style.context("../sorep.mplstyle"):
        fig, ax = plt.subplots(dpi=300)

        for feature_id, kwargs in FEATURE_KWARGS.items():
            metrics = pd.read_json(pl.Path(f"../data/{DATABASE}_metrics_{calculation_type}_{feature_id}.json"))
            gb = metrics.groupby(by="train_size")
            yield_mean = gb["yield"].mean()
            yield_std = gb["yield"].std()
            ax.plot(yield_mean.index, yield_mean, **kwargs["line_kwargs"], marker=".", label=kwargs["title"])
            ax.fill_between(yield_mean.index, yield_mean - yield_std, yield_mean + yield_std, **kwargs["fill_kwargs"])

        ax.axhline(target_df.meets_tcm_criteria.mean(), c="grey", ls="--", label=r"$f_{\mathrm{TCM}}$")

        top_sec_ax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * 0.9 * target_df.shape[0], lambda x: x / 0.9 / target_df.shape[0])
        )
        top_sec_ax.set_xlabel(r"$N_{\mathrm{train}}$")

        ax.tick_params(axis="x", which="both", top=False)

        ax.set_xscale("log")
        ax.set_xlabel(r"$f_{\mathrm{train}}$")
        ax.set_ylabel("Yield (TCM / calculation)")
        ax.legend(loc=[0.3, 0.05])

    return fig


plot_yield(DATABASE, CALCULATION_TYPE)


# %%
def plot_balanced_accuracy(database, calculation_type):
    target_df = pd.DataFrame({k: v.tolist() for (k, v) in np.load(pl.Path(f"../data/{database}_targets.npz")).items()})

    with plt.style.context("../sorep.mplstyle"):
        fig, ax = plt.subplots(dpi=300)

        for feature_id, kwargs in FEATURE_KWARGS.items():
            metrics = pd.read_json(pl.Path(f"../data/{DATABASE}_metrics_{calculation_type}_{feature_id}.json"))
            gb = metrics.groupby(by="train_size")
            balanced_accuracy_mean = gb["balanced_accuracy"].mean()
            balanced_accuracy_std = gb["balanced_accuracy"].std()
            # ax.errorbar(
            #     balanced_accuracy_mean.index,
            #     balanced_accuracy_mean,
            #     yerr=balanced_accuracy_std,
            #     **kwargs["errorbar_kwargs"],
            #     label=kwargs["title"],
            # )
            ax.plot(
                balanced_accuracy_mean.index,
                balanced_accuracy_mean,
                **kwargs["line_kwargs"],
                marker=".",
                label=kwargs["title"],
            )
            ax.fill_between(
                balanced_accuracy_mean.index,
                balanced_accuracy_mean - balanced_accuracy_std,
                balanced_accuracy_mean + balanced_accuracy_std,
                **kwargs["fill_kwargs"],
            )

        top_sec_ax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * 0.9 * target_df.shape[0], lambda x: x / 0.9 / target_df.shape[0])
        )
        top_sec_ax.set_xlabel(r"$N_{\mathrm{train}}$")

        ax.tick_params(axis="x", which="both", top=False)

        ax.set_xscale("log")
        ax.set_xlabel(r"$f_{\mathrm{train}}$")
        ax.set_ylabel("Balanced accuracy")
        ax.legend(loc=[0.4, 0.1])

    return fig


plot_balanced_accuracy(DATABASE, CALCULATION_TYPE)


# %%
def plot_feature_importances(database, calculation_type, train_size=0.5, axes=None):
    with plt.style.context("../sorep.mplstyle"):
        if axes is None:
            fig, axes = plt.subplots(2, 2, dpi=300, figsize=(4.5, 4), sharey=True)
            axes = axes.flatten()
        else:
            fig = plt.gcf()

        for i, (ax, (feature_id, kwargs)) in enumerate(zip(axes, FEATURE_KWARGS.items())):
            metrics = pd.read_json(pl.Path(f"../data/{database}_metrics_{calculation_type}_{feature_id}.json"))

            feature_importances = np.array(metrics[metrics.train_size == train_size]["feature_importances"].to_list())
            feature_importances_mean = feature_importances.mean(axis=0)
            feature_importances_std = feature_importances.std(axis=0)

            ax.set_title(kwargs["title"])
            ax.plot(kwargs["line_x"], feature_importances_mean, **kwargs["line_kwargs"])[0]

            ax.fill_between(
                kwargs["line_x"],
                feature_importances_mean - feature_importances_std,
                feature_importances_mean + feature_importances_std,
                **kwargs["fill_kwargs"],
            )

            for vline in kwargs.get("vlines", []):
                ax.axvline(**vline)

            ax.grid(visible=True, which="major", axis="both")
            ax.set_xticks(kwargs["xticks"], kwargs["xtick_labels"], **kwargs.get("xticks_kwargs", {}))
            if i % 2 == 0:
                ax.set_ylabel("Feature importance")
            if i > 1:
                ax.set_xlabel("Feature")
            ax.set_yticklabels([])
            ax.legend()
        fig.tight_layout()

    return fig


fig = plot_feature_importances(DATABASE, CALCULATION_TYPE, train_size=0.5)
fig.savefig("../plots/feature_importances.pdf", bbox_inches="tight")
fig
# %%
