"""Compute occupations, number of electrons, and their derivatives."""

from .cython import compute_n_electrons, compute_n_electrons_2nd_derivative, compute_n_electrons_derivative
from .python import compute_n_electrons as compute_n_electrons_python
from .python import compute_n_electrons_2nd_derivative as compute_n_electrons_2nd_derivative_python
from .python import compute_n_electrons_derivative as compute_n_electrons_derivative_python
from .python import compute_occupations, compute_occupations_2nd_derivative, compute_occupations_derivative

__all__ = (
    "compute_n_electrons",
    "compute_n_electrons_derivative",
    "compute_n_electrons_2nd_derivative",
    "compute_occupations",
    "compute_occupations_derivative",
    "compute_occupations_2nd_derivative",
    "compute_n_electrons_python",
    "compute_n_electrons_derivative_python",
    "compute_n_electrons_2nd_derivative_python",
)
