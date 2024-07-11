"""Test configuration."""

from importlib.resources import files
import json
import test.resources  # pylint: disable=import-error

import jax
import pytest

import sorep

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", params=["one_spin_insulator", "two_spin_insulator", "one_spin_metal", "two_spin_metal"])
def bandstructure(request):
    """Load band structure data from test resources."""
    dir_ = files(test.resources).joinpath(request.param)  # pylint: disable=no-member
    with open(dir_.joinpath("metadata.json"), "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return (
        sorep.BandStructure.from_files(
            dir_.joinpath("bands.npz"), dir_.joinpath("structure.xyz"), dir_.joinpath("metadata.json")
        ),
        metadata["smearing_type"],
        metadata["degauss"],
    )
