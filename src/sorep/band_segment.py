"""Class for a segment of a band structure along a path."""

import typing as ty

from findiff import FinDiff
import numpy as np
import numpy.typing as npt
import scipy as sp

from .constants import ANGSTROM_TO_BOHR, EV_TO_HARTREE, ROOM_TEMP_EV
from .prettify import unicode_kpoint_label
from .smearing import smearing_from_name


class BandPathSegment:
    # pylint: disable=too-many-arguments
    """Segment of a band structure along a path."""

    def __init__(
        self,
        bands: npt.ArrayLike,
        linear_k: ty.Optional[npt.ArrayLike],
        fermi_energy: ty.Optional[float] = None,
        start_label: ty.Optional[str] = None,
        stop_label: ty.Optional[str] = None,
        start_index: ty.Optional[int] = None,
        stop_index: ty.Optional[int] = None,
    ):
        self.bands = bands
        self.linear_k = linear_k
        self.fermi_energy = fermi_energy
        self.start_label = start_label
        self.stop_label = stop_label
        self.start_index = start_index
        self.stop_index = stop_index

    def __repr__(self) -> str:
        start_unicode = unicode_kpoint_label(self.start_label)
        stop_unicode = unicode_kpoint_label(self.stop_label)
        return f"BandPathSegment({start_unicode} \u2192 {stop_unicode}, n_kpoints={self.linear_k.shape[0]})"

    def compute_finite_diff_effective_mass(self, cbm: float, vbm: float, acc: int = 2, **_) -> dict[str, npt.ArrayLike]:
        """Compute the electron and hole effective masses using finite differences as the invserse of
        the band curvature about the CBM and VBM, respectively.

        Args:
            cbm (float): Conduction band minimum.
            vbm (float): Valence band maximum.
            acc (int, optional): Finite differences accuracy. Defaults to 2.

        Returns:
            dict[str, npt.ArrayLike]: Dictionary containing the electron and hole effective masses for each
            e-k point in the segment that `np.isclose` to the CBM and VBM, respectively.
        """
        dk = np.mean(np.diff(self.linear_k))
        assert np.allclose(np.diff(self.linear_k), dk)

        # Convert units to Hartree atomic so that the effective mass is in atomic units (electron masses)
        dk /= ANGSTROM_TO_BOHR
        cbm *= EV_TO_HARTREE
        vbm *= EV_TO_HARTREE
        bands = self.bands * EV_TO_HARTREE

        axis = 1  # axes are (spins, kpoints, bands)
        spacing = dk
        count = 2  # 2nd order derivative
        d2_dk2 = FinDiff(axis, spacing, count, acc=acc)
        try:
            bands_curvature = d2_dk2(bands)
            electron_effective_masses = 1 / bands_curvature[np.where(np.isclose(bands, cbm))]
            hole_effective_masses = 1 / bands_curvature[np.where(np.isclose(bands, vbm))]
        except IndexError:
            electron_effective_masses = np.array([])
            hole_effective_masses = np.array([])

        return {"electron": electron_effective_masses, "hole": hole_effective_masses}

    def compute_integral_line_effective_mass(
        self,
        cbm: float,
        vbm: float,
        smearing_type: ty.Optional[ty.Union[str, int]] = "fermi-dirac",
        smearing_width: float = ROOM_TEMP_EV,
        acc: int = 2,
        **_,
    ) -> dict[str, float]:
        # pylint: disable=too-many-locals
        """Compute the electron and hole line effective masses by weighting the band curvature computed
        at every e-k point using the requested occupation function and its distance from the CBM or VBM.

        Args:
            cbm (float): Conduction band minimum.
            vbm (float): Valence band maximum.
            smearing_type (ty.Optional[ty.Union[str, int]], optional): Type of smearing. Defaults to 'fermi-dirac'.
            smearing_width (float, optional): Smearing width. Defaults to 300 K.
            acc (int, optional): Finite differences accuracy. Defaults to 2.

        Returns:
            dict[str, npt.ArrayLike]: Dictionary containing the electron and hole effective masses for the segment.
        """
        dk = np.mean(np.diff(self.linear_k))
        assert np.allclose(np.diff(self.linear_k), dk)

        # Convert units to Hartree atomic so that the effective mass is in atomic units (electron masses)
        dk /= ANGSTROM_TO_BOHR
        cbm *= EV_TO_HARTREE
        vbm *= EV_TO_HARTREE
        bands = self.bands * EV_TO_HARTREE
        fermi_energy = self.fermi_energy * EV_TO_HARTREE
        smearing_width = smearing_width * EV_TO_HARTREE

        smearing_cls = smearing_from_name(smearing_type)
        axis = 0  # After reshaping below, axes are (kpoints, bands)
        spacing = dk
        count = 2  # 2nd order derivative
        d2_dk2 = FinDiff(axis, spacing, count, acc=acc)

        try:
            # Get all bands which are completely above the Fermi energy
            # from all spin channels and stack them together into a single
            # array of shape (n_kpoints, n_bands).
            conduction = np.hstack([bands_spin[:, np.all(bands_spin > fermi_energy, axis=0)] for bands_spin in bands])
            conduction_curvature = d2_dk2(conduction)
            conduction_occupations = smearing_cls(cbm, smearing_width).occupation(conduction)
            # Integrate over k (axis 0), sum over bands (axis 1)
            electron_effective_mass = (
                sp.integrate.simpson(y=conduction_occupations, dx=dk, axis=0).sum()
                / sp.integrate.simpson(y=(conduction_occupations * conduction_curvature), dx=dk, axis=0).sum()
            )

            valence = np.hstack([bands_spin[:, np.all(bands_spin < fermi_energy, axis=0)] for bands_spin in bands])
            valence_curvature = d2_dk2(valence)
            # Here, we need to invert the center and argument of the occupation function
            # to occupy down from the VBM into the valence as if we were occupying up
            # into the conduction from the CBM.
            # pylint: disable=invalid-unary-operand-type
            valence_occupations = smearing_cls(-vbm, smearing_width).occupation(-valence)
            hole_effective_mass = (
                sp.integrate.simpson(y=valence_occupations, dx=dk, axis=0).sum()
                / sp.integrate.simpson(y=(valence_occupations * valence_curvature), dx=dk, axis=0).sum()
            )
        except IndexError:
            electron_effective_mass = np.nan
            hole_effective_mass = np.nan

        return {"electron": electron_effective_mass, "hole": hole_effective_mass}
