"""Density of states functions."""

from .cython import smeared_dos
from .python import smeared_dos as smeared_dos_python  # pylint: disable=no-name-in-module

__all__ = ("smeared_dos_python", "smeared_dos")
