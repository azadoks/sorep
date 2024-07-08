"""
Compute the bands screening quantities for TCMs:
- Band gap
- Electron effective mass
- Hole effective mass
"""
# %%
import pathlib as pl
import typing as ty

import findiff
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp

import sorep


# %%
def _get_linear_k(kpoints: npt.ArrayLike,
                  label_numbers: ty.Sequence[int]) -> npt.ArrayLike:
    # Compute distances between adjacent k-points
    distances = np.linalg.norm(np.diff(kpoints, axis=0), axis=1)
    # Set distance to zero when adjacent k-points are both labeled (likely a discontinuity)
    mask = np.array([
        i in label_numbers and i - 1 in label_numbers
        for i in range(1, kpoints.shape[0])
    ])
    distances[mask] = 0.0
    # Prepend 0 (the linear location of the first k-point)
    linear_k = np.concatenate([[0], np.cumsum(distances)])
    return linear_k


# %%
BOHR_TO_ANGSTROM = 0.52917720859
ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM
RY_TO_EV = 13.6056917253
EV_TO_RY = 1 / RY_TO_EV
HARTREE_TO_EV = 2 * RY_TO_EV
EV_TO_HARTREE = 1 / HARTREE_TO_EV
KELVIN_TO_EV = 8.61732814974056E-05
ROOM_TEMP = 300 * KELVIN_TO_EV

class BandPathSegment:
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
        start_unicode = sorep.prettify.unicode_kpoint_label(self.start_label)
        stop_unicode = sorep.prettify.unicode_kpoint_label(self.stop_label)
        return f'BandPathSegment({start_unicode} \u2192 {stop_unicode}, n_kpoints={self.linear_k.shape[0]})'

    def compute_finite_diff_effective_mass(
        self,
        cbm: float,
        vbm: float,
        acc: int = 2,
        **_
    ) -> dict[str, npt.ArrayLike]:
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

        d2_dk2 = findiff.FinDiff(1, dk, 2, acc=acc)
        try:
            bands_curvature = d2_dk2(bands)

            electron_effective_masses = 1 / bands_curvature[np.where(np.isclose(bands, cbm))]
            hole_effective_masses = 1 / bands_curvature[np.where(np.isclose(bands, vbm))]
        except IndexError:
            electron_effective_masses = np.array([])
            hole_effective_masses = np.array([])

        return {'electrons': electron_effective_masses, 'holes': hole_effective_masses}

    def compute_integral_line_effective_mass(
            self,
            cbm: float,
            vbm: float,
            smearing_type: ty.Optional[ty.Union[str, int]] = 'fermi-dirac',
            smearing_width: float = ROOM_TEMP,
            acc: int = 2,
            **_
        ) -> dict[str, npt.ArrayLike]:
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

        smearing_cls = sorep.smearing.smearing_from_name(smearing_type)
        d2_dk2 = findiff.FinDiff(1, dk, 2, acc=acc)

        try:
            conduction = np.array([bands_spin[:, np.all(bands_spin > fermi_energy, axis=0)] for bands_spin in bands])
            conduction_curvature = d2_dk2(conduction)
            conduction_occupations = smearing_cls(cbm, smearing_width).occupation(conduction)
            conduction_num = sp.integrate.simpson(y=conduction_occupations, dx=dk, axis=1).sum()
            conduction_denom = sp.integrate.simpson(y=(conduction_occupations * conduction_curvature), dx=dk, axis=1).sum()
            electron_effective_mass = conduction_num / conduction_denom

            valence = np.array([bands_spin[:, np.all(bands_spin < fermi_energy, axis=0)] for bands_spin in bands])
            valence_curvature = d2_dk2(valence)
            valence_occupations = smearing_cls(-vbm, smearing_width).occupation(-valence)
            valence_num = sp.integrate.simpson(y=valence_occupations, dx=dk, axis=1).sum()
            valence_denom = sp.integrate.simpson(y=(valence_occupations * valence_curvature), dx=dk, axis=1).sum()
            hole_effective_mass = valence_num / valence_denom
        except IndexError:
            electron_effective_mass = np.nan
            hole_effective_mass = np.nan

        return {'electron': np.array([electron_effective_mass]), 'hole': np.array([hole_effective_mass])}

# %%
def get_segments(bands: sorep.BandStructure) -> list[BandPathSegment]:
    linear_k = _get_linear_k(
        sorep.pbc.recip_frac_to_cart(bands.kpoints, bands.cell),
        bands.label_numbers)
    # Construct the segments
    segments = []
    for (i_from, i_to) in zip(range(len(bands.labels) - 1),
                              range(1, len(bands.labels))):
        ik_from = bands.label_numbers[i_from]
        ik_to = bands.label_numbers[i_to] + 1
        segment = BandPathSegment(
            bands=bands.bands[:, ik_from:ik_to].copy(),
            linear_k=linear_k[ik_from:ik_to].copy(),
            fermi_energy=bands.fermi_energy if bands.fermi_energy else None,
            start_label=bands.labels[i_from],
            stop_label=bands.labels[i_to],
            start_index = ik_from,
            stop_index = ik_to,
        )
        segments.append(segment)

    return segments


def plot_segments(material):
    segments = get_segments(material.bands)
    fig, ax = plt.subplots()
    xticks = [0]
    xtick_labels = [sorep.prettify.latex_kpoint_label(segments[0].start_label)]
    for seg in segments:
        if len(seg.linear_k) > 2:
            xticks.append(seg.linear_k[-1])
            xtick_labels.append(
                sorep.prettify.latex_kpoint_label(seg.stop_label))
            if seg.bands.shape[0] == 1:
                ax.plot(seg.linear_k, seg.bands[0], c='k')
            else:
                ax.plot(seg.linear_k, seg.bands[0], c='tab:blue')
                ax.plot(seg.linear_k, seg.bands[1], c='tab:red')
    for seg in segments[:-1]:
        ax.axvline(seg.linear_k[-1], c='grey', ls='-', alpha=0.5)

    if material.bands.is_insulating():
        linear_k = _get_linear_k(
            sorep.pbc.recip_frac_to_cart(material.bands.kpoints,
                                         material.bands.cell),
            material.bands.label_numbers)
        cbm_idx = material.bands.cbm_index
        ax.scatter([linear_k[cbm_idx[1]]], [material.bands.bands[cbm_idx]],
                   marker='o',
                   c='tab:purple',
                   s=30,
                   zorder=10,
                   linewidth=1,
                   edgecolors='k')
        vbm_idx = material.bands.vbm_index
        ax.scatter([linear_k[vbm_idx[1]]], [material.bands.bands[vbm_idx]],
                   marker='o',
                   c='tab:purple',
                   s=30,
                   zorder=10,
                   linewidth=1,
                   edgecolors='k')

    ax.set_xticks(xticks, labels=xtick_labels)
    ax.axhline(material.bands.fermi_energy, c='tab:green', ls='--')
    ax.set_ylim(material.bands.fermi_energy - 10,
                material.bands.fermi_energy + 10)
    ax.set_title(f"PwBandsWorkChain<{material.metadata['calculation_uuid']}>",
                 size=9)
    fig.suptitle(
        f'{sorep.prettify.unicode_chemical_formula(material.metadata["formula_hill"])}'
    )

    return fig, ax


# %%
# 5  ../data/mc3d/mpds|1.0.0|S1617807/bands  insulating w/ no spin
# 18 ../data/mc3d/mpds|1.0.0|S1712440/bands  insulating w/ collinear spin
material = sorep.MaterialData.from_dir(
    list(pl.Path('../data/mc3d/').glob('*/bands'))[18])
# material = sorep.MaterialData.from_dir('../data/mc3d/mpds|1.0.0|S303025/bands')

fig, ax = plot_segments(material)

# %%
segments = get_segments(material.bands)
int_eff_masses = []
fd_eff_masses = []
for (i, segment) in enumerate(segments):
    int_eff_masses.append(segment.compute_integral_line_effective_mass(material.bands.cbm, material.bands.vbm))
    fd_eff_masses.append(segment.compute_finite_diff_effective_mass(material.bands.cbm, material.bands.vbm))

{
    'int': int_eff_masses,
    'fd': fd_eff_masses
}
# %%
df = pd.read_json('../data/effective_mass.json')
df = df.dropna(axis='index')
df = df[np.abs(df.bands_cbm - df.bands_vbm) > 0.5 * EV_TO_HARTREE]
# %%
