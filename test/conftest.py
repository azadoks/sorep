"""Test configuration."""

from importlib.resources import files
import json
import test.resources  # pylint: disable=import-error

import numpy as np
import pytest


@pytest.fixture(scope="module", params=["one_spin_insulator", "two_spin_insulator", "one_spin_metal", "two_spin_metal"])
def bands_data(request):
    """Load band structure data from test resources."""
    dir_ = files(test.resources).joinpath(request.param)  # pylint: disable=no-member

    with open(dir_.joinpath("bands.npz"), "rb") as fp:
        arrays = dict(np.load(fp))
    with open(dir_.joinpath("metadata.json"), encoding="utf-8") as fp:
        metadata = json.load(fp)

    bands = arrays["bands"]
    if bands.ndim == 2:
        bands = np.expand_dims(bands, 0)

    weights = arrays["weights"]
    weights /= weights.sum()

    return {
        "bands": bands,
        "kpoints": arrays["kpoints"],
        "weights": weights,
        "occupations": arrays["occupations"],
        "fermi_energy": metadata["fermi_energy"],
        "n_electrons": metadata["number_of_electrons"],
        "smearing_type": metadata["smearing_type"],
        "smearing_width": metadata["degauss"],
    }
