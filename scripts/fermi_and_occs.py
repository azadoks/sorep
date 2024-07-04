# %%
import json
import os
import pathlib as pl

from ase import Atoms
from ase.io.extxyz import read_extxyz
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sorep

# %%
test_dirs = {
    "one_spin_insulator": None,
    "two_spin_insulator": None,
    "one_spin_metal": None,
    "two_spin_metal": None,
}
for dir_path in tqdm(list(pl.Path("../data/").glob("*/scf/0/"))):
    if all([value is not None for value in test_dirs.values()]):
        break
    material = sorep.MaterialData.from_dir(dir_path)
    n_spins = material.bands.n_spins
    is_insulating = material.bands.is_insulating()
    key = f"{'one' if n_spins == 1 else 'two'}_spin_{'insulator' if is_insulating else 'metal'}"
    if test_dirs[key] is None:
        test_dirs[key] = str(dir_path.resolve())

# %%
