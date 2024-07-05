"""Tests for the `sorep.fermi` module."""

import jax
import jax.numpy as jnp
import pytest  # pylint: disable=unused-import

from sorep.smearing import Cold, Delta, FermiDirac, Gaussian


@pytest.mark.parametrize("smearing_cls", [Delta, Gaussian, FermiDirac, Cold])
def test_occupation(smearing_cls):
    """Test `Smearing.occupation`."""
    smearing = smearing_cls(center=0.0, width=1.0)
    assert smearing.occupation(-100.0) == pytest.approx(1.0)
    assert smearing.occupation(100.0) == pytest.approx(0.0)


@pytest.mark.parametrize("smearing_cls", [Delta, Gaussian, FermiDirac, Cold])
def test_occupation_derivative(smearing_cls):
    """Test `Smearing.occupation_derivative`."""
    smearing = smearing_cls(center=0.0, width=1.0)
    x = jnp.linspace(-10.0, 10.0, 1_000)
    y = smearing.occupation_derivative(x)
    expected = -jax.vmap(jax.grad(smearing.occupation))(x)

    assert jnp.allclose(y, expected, atol=1e-8)


@pytest.mark.parametrize("smearing_cls", [Delta, Gaussian, FermiDirac, Cold])
def test_occupation_2nd_derivative(smearing_cls):
    """Test `Smearing.occupation_2nd_derivative`."""
    smearing = smearing_cls(center=0.0, width=1.0)
    x = jnp.linspace(-10.0, 10.0, 1_000)
    y = smearing.occupation_2nd_derivative(x)
    expected = -jax.vmap(jax.hessian(smearing.occupation))(x)

    assert jnp.allclose(y, expected, atol=1e-6)
