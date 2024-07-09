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
FEATURE_ID = "gauss_0.05"

target_df = pd.DataFrame({k: v.tolist() for (k, v) in np.load(pl.Path(f"../data/{DATABASE}_targets.npz")).items()})

# %%

feature_kwargs = {
    # "dos_vbm_centered": {"label": r"$E_{\mathrm{VBM}}(-2,+6) \mathrm{eV}$", "c": "#aa0000", "marker": "."},
    "dos_fermi_scissor": {"label": r"$E_{F}^{\mathrm{scissor}} \pm 2 \mathrm{eV}$", "c": "#000080", "marker": "."},
    "dos_fermi_centered": {"label": r"$E_{F} \pm 5 \mathrm{eV}$", "c": "#008000", "marker": "."},
}

with plt.style.context("../sorep.mplstyle"):
    fig, ax = plt.subplots(dpi=300)

    handles = []
    for feature_type, kwargs in feature_kwargs.items():
        metrics = pd.read_json(
            pl.Path(f"../data/{DATABASE}_metrics_{CALCULATION_TYPE}_{feature_type}_{FEATURE_ID}.json")
        )
        gb = metrics.groupby(by="train_size")
        yield_mean = gb["yield"].mean()
        yield_std = gb["yield"].std()
        dummy_yield_mean = gb["dummy_yield"].mean()
        dummy_yield_std = gb["dummy_yield"].std()
        errorbar = ax.errorbar(yield_mean.index, yield_mean, yerr=yield_std, **kwargs, capsize=3)
        handles.append(errorbar)

    hline = ax.axhline(
        target_df.meets_tcm_criteria.mean(), c="grey", ls="--", label=r"$N_{\mathrm{TCM}}^{\mathrm{total}}$"
    )
    handles.append(hline)

    top_sec_ax = ax.secondary_xaxis(
        "top", functions=(lambda x: x * 0.9 * target_df.shape[0], lambda x: x / 0.9 / target_df.shape[0])
    )
    top_sec_ax.set_xlabel(r"$N_{\mathrm{train}}$")

    ax.tick_params(axis="x", which="both", top=False)

    ax.set_xscale("log")
    ax.set_xlabel(r"$f_{\mathrm{train}}$")
    ax.set_ylabel("Yield (TCM / calculation)")
    ax.legend(handles=handles, loc=[0.55, 0.6])

    # fig.tight_layout()
fig
# %%
