# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial.distance as spdist
import seaborn as sns

# %%
FEATURE_TYPE = "fermi_centered/0"
DISTANCE_TYPE = "euclidean"

distance_data = {}
for calculation_type in ["scf", "single_shot"]:
    distance_data[calculation_type] = {}
    with h5py.File(f"../data/mc3d_distances_{calculation_type}.h5", "r") as f:
        distance_data[calculation_type]["distance"] = f[FEATURE_TYPE][DISTANCE_TYPE]["distance"][()]
        distance_data[calculation_type]["material_id"] = f[FEATURE_TYPE][DISTANCE_TYPE]["material_id"][()].astype(str)
# %%
common_ids = list(set(distance_data["scf"]["material_id"]) & set(distance_data["single_shot"]["material_id"]))
for calculation_type, dat in distance_data.items():
    to_drop = np.where(~np.isin(dat["material_id"], common_ids))
    if len(to_drop) > 0:
        distance_data[calculation_type]["distance"] = spdist.squareform(
            np.delete(np.delete(spdist.squareform(dat["distance"]), to_drop, axis=0), to_drop, axis=1)
        )
        distance_data[calculation_type]["material_id"] = np.delete(dat["material_id"], to_drop, axis=0)
# %%
assert np.all(distance_data["scf"]["material_id"] == distance_data["single_shot"]["material_id"])
# %%
N = len(common_ids)
M = N * (N + 1) // 2 - N

norm = False
xymin = -60
xymax = 675
levels = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

with plt.style.context("../sorep.mplstyle"):
    fig, ax = plt.subplots()

    # idx = np.random.choice(M, int(0.01 * M), replace=False)  # _very_ slow!
    idx = np.random.randint(0, M, 5_000)
    x = distance_data["scf"]["distance"][idx]
    y = distance_data["single_shot"]["distance"][idx]

    if norm:
        x = x / x.max()
        y = y / y.max()
        xymin = -0.05
        xymax = 0.60

    sns_ax = sns.kdeplot(ax=ax, x=x, y=y, fill=True, cbar=True, common_norm=True, levels=levels, cmap="viridis")
    sns_ax = sns.kdeplot(ax=ax, x=x, y=y, common_norm=True, levels=levels, linewidths=0.5, color="k")

    ax.set_xlim(xymin, xymax)
    ax.set_xlabel(r"$D_{\mathrm{SCF}}$")
    ax.set_ylabel(r"$D_{\mathrm{SOREP}}$")
    ax.set_ylim(xymin, xymax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")

    cax = fig.get_axes()[1]
    cax.set_yticklabels(levels)
    cax.set_ylabel("Density outside contour", rotation=270)

    fig.tight_layout()

    # fig.savefig(f"../plots/mc3d_distance_comparison_{FEATURE_TYPE.replace('/', '-')}_bare.pdf", dpi=300)

    ax.plot([xymin, xymax], [xymin, xymax], c="k", linestyle=":", linewidth=1, label=r"$1:1$")
    ax.legend()
    # fig.savefig(f"../plots/mc3d_distance_comparison_{FEATURE_TYPE.replace('/', '-')}_1-1.pdf", dpi=300)

    x_lr = distance_data["scf"]["distance"]
    y_lr = distance_data["single_shot"]["distance"]
    if norm:
        x_lr = x_lr / x_lr.max()
        y_lr = y_lr / y_lr.max()
    res = sp.stats.linregress(x_lr, y_lr)
    linex = np.linspace(xymin, xymax, 100)
    ax.plot(linex, res.intercept + res.slope * linex, "k-.", linewidth=1, label=f"$r^2 = {res.rvalue**2:.2f}$")
    ax.legend()
    # fig.savefig(f"../plots/mc3d_distance_comparison_{FEATURE_TYPE.replace('/', '-')}_1-1_regress.pdf", dpi=300)

    ax.legend()
fig
# %%
