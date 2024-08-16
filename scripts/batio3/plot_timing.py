# %%
from os import environ

N_THREADS = "1"
environ["OMP_NUM_THREADS"] = N_THREADS
environ["OPENBLAS_NUM_THREADS"] = N_THREADS
environ["MKL_NUM_THREADS"] = N_THREADS
environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS
environ["NUMEXPR_NUM_THREADS"] = N_THREADS

import time

from aiida import load_profile, orm, plugins
import matplotlib.pyplot as plt
import pandas as pd

import sorep

PwBaseWorkChain = plugins.WorkflowFactory("quantumespresso.pw.base")
PwBandsWorkChain = plugins.WorkflowFactory("quantumespresso.pw.bands")


load_profile("batio3")
STRUCTURE_GROUP_LABEL = "BaTiO3/structures/optimized"


INPUT_PROJECT = ["attributes.SYSTEM.occupations", "attributes.SYSTEM.smearing", "attributes.SYSTEM.degauss"]
OUTPUT_PROJECT = [
    "attributes.wfc_cutoff",
    "attributes.wfc_cutoff_units",
    "attributes.rho_cutoff",
    "attributes.rho_cutoff_units",
    "attributes.volume",
    "attributes.number_of_species",
    "attributes.number_of_atoms",
    "attributes.number_of_electrons",
    "attributes.number_of_spin_components",
    "attributes.number_of_bands",
    "attributes.number_of_k_points",
    "attributes.monkhorst_pack_grid",
    "attributes.monkhorst_pack_offset",
    "attributes.lsda",
    "attributes.do_magnetization",
    "attributes.fermi_energy",
    "attributes.fermi_energy_units",
    "attributes.wall_time_seconds",
    "attributes.dft_exchange_correlation",
    "attributes.creator_version",
]


# %%
def _query():
    qb = orm.QueryBuilder()
    qb.append(orm.Group, filters={"label": STRUCTURE_GROUP_LABEL}, tag="group")
    qb.append(
        orm.StructureData,
        with_group="group",
        project=[
            "*",
            "uuid",
            "extras.source.id",
            "extras.source.database",
            "extras.source.version",
            "extras.duplicates",
        ],
        tag="structure",
    )

    qb.append(PwBaseWorkChain, with_incoming="structure", project=["*", "uuid", "ctime"], tag="single_shot")
    qb.append(
        orm.Dict,
        with_outgoing="single_shot",
        edge_filters={"label": "pw__parameters"},
        filters={"attributes.ELECTRONS.electron_maxstep": 0},
        project=INPUT_PROJECT,
        tag="single_shot_input",
    )
    qb.append(
        orm.Dict,
        with_incoming="single_shot",
        edge_filters={"label": "output_parameters"},
        project=OUTPUT_PROJECT,
        tag="single_shot_output",
    )
    qb.append(
        orm.BandsData,
        with_incoming="single_shot",
        project=["*", "uuid", "attributes.labels", "attributes.label_numbers"],
        tag="single_shot_bands",
    )

    qb.append(PwBaseWorkChain, with_incoming="structure", project=["*", "uuid", "ctime"], tag="scf")
    qb.append(
        orm.Dict,
        with_outgoing="scf",
        edge_filters={"label": "pw__parameters"},
        filters={"attributes.ELECTRONS.electron_maxstep": {">": 0}},
        project=INPUT_PROJECT,
        tag="scf_input",
    )
    qb.append(
        orm.Dict,
        with_incoming="scf",
        edge_filters={"label": "output_parameters"},
        project=OUTPUT_PROJECT,
        tag="scf_output",
    )
    qb.append(
        orm.BandsData,
        with_incoming="scf",
        project=["*", "uuid", "attributes.labels", "attributes.label_numbers"],
        tag="scf_bands",
    )
    return qb.dict()


# %%
def get_nprocs(node):
    out = node.outputs.retrieved.get_object_content("aiida.out")
    for line in out.split("\n"):
        if "Parallel version" in line:
            return int(line.split()[-2])


qr = _query()
run_time = []
for r in qr:
    id_ = "|".join(
        [
            r["structure"]["extras.source.database"],
            r["structure"]["extras.source.version"],
            r["structure"]["extras.source.id"],
        ]
    )
    row = {"uuid": r["structure"]["uuid"], "id": id_, "n_atoms": r["scf_output"]["attributes.number_of_atoms"]}
    for calc_type in ("scf", "single_shot"):
        row[f"{calc_type}_creator_version"] = r[f"{calc_type}_output"]["attributes.creator_version"]
        row[f"{calc_type}_nkpoints"] = r[f"{calc_type}_output"]["attributes.number_of_k_points"]
        row[f"{calc_type}_nbands"] = r[f"{calc_type}_output"]["attributes.number_of_bands"]
        row[f"{calc_type}_nprocs"] = get_nprocs(r[calc_type]["*"])
        row[f"{calc_type}_wall_time_seconds"] = r[f"{calc_type}_output"]["attributes.wall_time_seconds"]
        row[f"{calc_type}_cpu_time_seconds"] = row[f"{calc_type}_nprocs"] * row[f"{calc_type}_wall_time_seconds"]
    run_time.append(row)
# %%
df = pd.DataFrame(run_time)
df = df[df["scf_nprocs"] == 1]
df["single_shot_speedup"] = df["single_shot_cpu_time_seconds"] / df["scf_cpu_time_seconds"]

# %%
KIN_BANDS_PARAMS = {
    "basis": "ano-ml-os",
    "operator": "int1e_kin",
    "kdensity": 0.15,  # 1/angstrom
    "eigen_solver": "eigh",
    "use_symmetries": True,
}

KIN_DOS_PARAMS = {
    "center": 0.0,
    "e_min": 0.0,
    "e_max": 180.0,
    "n_energies": 1024,
    "smearing_type": "gauss",
    "smearing_width": 0.8,
}


def time_kin(uuid):
    structure = orm.load_node(uuid)
    atoms = structure.get_ase()
    t0 = time.time()
    bands = sorep.spectra.pyscf.compute_one_electron_spectrum(atoms, **KIN_BANDS_PARAMS)
    # dos = sorep.features._dos_featurize(bands, **KIN_DOS_PARAMS)
    t1 = time.time()
    return t1 - t0


df["kin_cpu_time_seconds"] = df["uuid"].apply(time_kin)
df["kin_speedup"] = df["kin_cpu_time_seconds"] / df["scf_cpu_time_seconds"]

# %%
with plt.style.context("../../sorep.mplstyle"):
    fig, axes = plt.subplots(1, 2, width_ratios=[5, 1])

    xy_min = min(
        df["scf_cpu_time_seconds"].min(), df["single_shot_cpu_time_seconds"].min(), df["kin_cpu_time_seconds"].min()
    )
    xy_max = max(
        df["scf_cpu_time_seconds"].max(), df["single_shot_cpu_time_seconds"].max(), df["kin_cpu_time_seconds"].max()
    )

    axes[0].scatter(
        df["scf_cpu_time_seconds"],
        df["single_shot_cpu_time_seconds"],
        marker="o",
        c="#000080",
        label="Single-shot SOREP",
    )
    axes[0].scatter(
        df["scf_cpu_time_seconds"], df["kin_cpu_time_seconds"], marker="^", c="#aa0000", label="Kinetic SOREP"
    )
    axes[0].plot([1e-1, 3e3], [1e-1, 3e3], color="black", linestyle="--", linewidth=0.5)

    axes[0].set_aspect("equal")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(1e-1, 3e3)
    axes[0].set_ylim(1e-1, 3e3)
    axes[0].set_xlabel(r"$t_{\mathrm{SCF}}$ (core-second)")
    axes[0].set_ylabel(r"$t_{\mathrm{SOREP}}$ (core-second)")
    axes[0].tick_params(axis="x", which="minor", bottom=True, top=True)
    axes[0].legend()

    axes[1].boxplot(
        [df["single_shot_speedup"]],
        positions=[0],
        widths=[0.5],
        labels=["S.S."],
        sym=".",
        medianprops={"color": "#000080"},
        boxprops={"color": "k"},
        whiskerprops={"color": "k"},
        capprops={"color": "k"},
    )
    axes[1].boxplot(
        [df["kin_speedup"]],
        positions=[1],
        widths=[0.5],
        labels=["Kin."],
        sym=".",
        medianprops={"color": "#aa0000"},
        boxprops={"color": "k"},
        whiskerprops={"color": "k"},
        capprops={"color": "k"},
    )
    axes[1].tick_params(axis="x", which="minor", bottom=False, top=False)
    axes[1].set_ylim(1e-4, 1e-1)
    axes[1].set_xlim(-0.75, 1.75)
    axes[1].set_ylabel(r"$t_{\mathrm{SOREP}}~/~t_{\mathrm{SCF}}$")
    axes[1].set_yscale("log")
    axes[1].set_xticklabels(["S.-S.", "Kin."], rotation=90)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")

# fig.savefig("../../plots/batio3_timing.pdf", bbox_inches="tight")
fig
# %%
