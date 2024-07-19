# %%
import os
import pathlib as pl

from ase import Atoms
from ase.io.extxyz import write_extxyz
import numpy as np


# %%
def read_nomad_xyz(path: os.PathLike) -> Atoms:
    """Parse the old? ASE XYZ format used by the NOMAD2018 TCM dataset.

    Args:
        path (os.PathLike): Path to the XYZ file.

    Returns:
        Atoms: ASE Atoms object.
    """
    with open(path, "r", encoding="utf-8") as fp:
        lines: list[str] = fp.readlines()
    cell: list[list[float]] = []
    positions: list[list[float]] = []
    symbols: list[str] = []
    for line in lines:
        if line.startswith("lattice_vector"):
            cell.append([float(x) for x in line.split()[1:]])
        elif line.startswith("atom"):
            parts = line.split()
            symbols.append(parts[-1])
            positions.append([float(x) for x in parts[1:4]])
    return Atoms(cell=cell, symbols=symbols, positions=positions, pbc=True)


# %%
images_train: list[Atoms] = []
images_test: list[Atoms] = []
for xyz_path in pl.Path("../data/nomad2018tcm/").glob("**/geometry.xyz"):
    atoms: Atoms = read_nomad_xyz(xyz_path)
    train_test: str = xyz_path.parents[1].name
    id_: int = int(xyz_path.parents[0].name)
    atoms.info["material_id"] = id_
    if train_test == "train":
        images_train.append(atoms)
    else:
        images_test.append(atoms)
write_extxyz("../data/nomad2018tcm_structures_train.xyz", images_train)
write_extxyz("../data/nomad2018tcm_structures_test.xyz", images_test)
# %%
