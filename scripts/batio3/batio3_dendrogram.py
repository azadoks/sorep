# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch

# %%
N = 12.0
Z3 = np.array(
    [
        [0.0, 1.0, 2e-3, 72],  # tetra + tetra1 -> N+1
        [5.0, 6.0, 1e-2, 2],  # rhomb4 + rhomb3 -> N+2
        [2.0, 3.0, 1.2e-2, 29],  # ortho2 + ortho -> N+3
        [4.0, N + 2, 2e-2, 13],  # rhomb3 + rhomb -> N+4
        [N + 3, N + 4, 7e-2, 42],  # ortho + rhomb -> N+5
        [N + 1, N + 5, 1.0, 114],  # N+6 imaginary merger of ortho/romb w/ tetra
        [N + 6, 7.0, 1.0, 118],  # N+7 cubic
        [N + 7, 8.0, 1.0, 119],  # N+8 cubic ba/ti swap
        [N + 8, 9.0, 1.0, 122],  # N+9 layered 1
        [N + 9, 10.0, 1.0, 125],  # N+10 layered 2
        [N + 10, 11.0, 1.0, 126],  # ortho. non-perov.
        [N + 11, 12.0, 1.0, 127],  # super-tet. non-perov.
    ]
)

with plt.style.context("../sorep.mplstyle"):
    fig, ax = plt.subplots()
    dend = sch.dendrogram(
        Z3,
        ax=ax,
        labels=[
            "(3)",
            "Tet. (69)",
            "(1)",
            "Ortho. (28)",
            "(1)",
            "(1)",
            "Rhomb. (11)",
            "Cubic (4)",
            "Cubic Ba/Ti swap (1)",
            "Layered Perov. 1 (3)",
            "Layered Perov. 2 (3)",
            "Ortho. non-Perov. (1)",
            "Super-tet. non-Perov. (1)",
        ],
        color_threshold=5e-2,
        above_threshold_color="black",
        orientation="right",
        count_sort="ascending",
        leaf_font_size=8,
    )
    ax.tick_params(which="minor", axis="y", left=False)
    ax.grid(axis="x", which="both", color="grey", linestyle="--", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon$", size=12)
    ax.set_xlim(1e-3, 1e-1)

fig.savefig("../plots/batio3_dendrogram.pdf", bbox_inches="tight")
# %%
