"""
Compute the bands screening quantities for TCMs:
- Band gap
- Electron effective mass
- Hole effective mass
"""
# %%
import pathlib as pl
import re
import typing as ty

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import sorep


# %%
def _latex_k_label(label: str) -> str:
    """Prettify a k-point label by replacing keywords with LaTeX strings.

    Args:
        label (str): k-point label (e.g. "GAMMA")

    Returns:
        str: LaTeXified string (e.g. r"$\\Gamma$")
    """
    # Replace spelled-out greek letters with LaTeX symbols
    # yapf: disable
    label = (
        label
            .replace('GAMMA', r'$\Gamma$')
            .replace('DELTA', r'$\Delta$')
            .replace('LAMBDA', r'$\Lambda$')
            .replace('SIGMA', r'$\Sigma$')
    )
    # yapf: enable
    # Replace underscore-numerals with LaTeX subscripts
    label = re.sub(r'_(.?)', r'$_{\1}$', label)
    return label

def _unicode_k_label(label: str) -> str:
    """Prettify a k-point label by replacing keywords with Unicode characters.

    Args:
        label (str): k-point label(e.g. "GAMMA")

    Returns:
        str: Unicode string (e.g. "Γ")
    """
    # Replace spelled-out greek letters with Unicode characters
    # yapf: disable
    label = (
        label
            .replace('GAMMA', "Γ")
            .replace('DELTA', "Δ")
            .replace('LAMBDA', "Λ")
            .replace('SIGMA', "Σ")
            .replace('_0', '\u2080')
            .replace('_1', '\u2081')
            .replace('_2', '\u2082')
            .replace('_3', '\u2083')
            .replace('_4', '\u2084')
            .replace('_5', '\u2085')
            .replace('_6', '\u2086')
            .replace('_7', '\u2087')
            .replace('_8', '\u2088')
            .replace('_9', '\u2089')

    )
    return label

def _fix_labels(label_numbers: ty.Sequence[int], labels: ty.Sequence[str], n_kpoints: int) -> tuple[list[int], list[str]]:
    label_numbers = list(label_numbers)
    labels = list(labels)

    # If the first k-point does not have a label, give it a placeholder label, and
    # add its index to the indices so that it can start a segment.
    if label_numbers[0] != 0:
        label_numbers.insert(0, 0)
        labels.insert(0, '')
    # Ditto with the last k-point; it needs to end the last segment
    if label_numbers[-1] != n_kpoints - 1:
        label_numbers.append(n_kpoints- 1)
        labels.append('')

    return (label_numbers, labels)

def _get_linear_k(kpoints: npt.ArrayLike, label_numbers: ty.Sequence[int]) -> npt.ArrayLike:
    # Compute distances between adjacent k-points
    distances = np.linalg.norm(np.diff(kpoints, axis=0), axis=1)
    # Set distance to zero when adjacent k-points are both labeled (likely a discontinuity)
    mask = np.array([i in label_numbers and i - 1 in label_numbers for i in range(1, kpoints.shape[0])])
    distances[mask] = 0.0
    # Prepend 0 (the linear location of the first k-point)
    linear_k = np.concatenate([[0], np.cumsum(distances)])
    return linear_k
# %%
class BandPathSegment:
    """Segment of a band structure along a path."""
    def __init__(
        self,
        bands: npt.ArrayLike,
        kpoints: npt.ArrayLike,
        weights: npt.ArrayLike,
        occupations: ty.Optional[npt.ArrayLike] = None,
        linear_k: ty.Optional[npt.ArrayLike] = None,
        fermi_energy: ty.Optional[float] = None,
        start_label: ty.Optional[str] = None,
        stop_label: ty.Optional[str] = None,
    ):
        self.bands = bands
        self.kpoints = kpoints
        self.weights = weights
        self.occupations = occupations
        if linear_k is not None:
            self.linear_k = linear_k
        else:
            self.linear_k = np.concatenate([[0], np.linalg.norm(np.diff(kpoints, axis=0), axis=1)])
        self.fermi_energy = fermi_energy
        self.start_label = start_label
        self.stop_label = stop_label

    def __repr__(self) -> str:
        start_unicode = _unicode_k_label(self.start_label)
        stop_unicode = _unicode_k_label(self.stop_label)
        return f'BandPathSegment({start_unicode} \u2192 {stop_unicode}, n_kpoints={self.kpoints.shape[0]})'

# %%
def get_segments(bands: sorep.BandStructure) -> list[BandPathSegment]:
    label_numbers, labels = _fix_labels(bands.label_numbers, bands.labels, bands.n_kpoints)
    linear_k = _get_linear_k(bands.kpoints, bands.label_numbers)
    # Construct the segments
    segments = []
    for (i_from, i_to) in zip(range(len(labels) - 1), range(1, len(labels))):
        ik_from = label_numbers[i_from]
        ik_to = label_numbers[i_to] + 1
        segment = BandPathSegment(
            bands=bands.bands[:, ik_from:ik_to].copy(),
            kpoints=bands.kpoints[ik_from:ik_to].copy(),
            weights=bands.weights[ik_from:ik_to].copy(),
            occupations=bands.occupations[:, ik_from:ik_to].copy() if bands.occupations is not None else None,
            linear_k=linear_k[ik_from:ik_to].copy(),
            fermi_energy=bands.fermi_energy if bands.fermi_energy else None,
            start_label=labels[i_from],
            stop_label=labels[i_to],
        )
        segments.append(segment)

    return segments
# %%
segs = get_segments(material.bands)
segs
# %%
fig, ax = plt.subplots()
xticks = [0]
xtick_labels = [_latex_k_label(segs[0].start_label)]
for seg in segs:
    if len(seg.linear_k) > 2:
        xticks.append(seg.linear_k[-1])
        xtick_labels.append(_latex_k_label(seg.stop_label))
        ax.plot(seg.linear_k, seg.bands[0], c='k')
for seg in segs[:-1]:
    ax.axvline(seg.linear_k[-1], c='grey', ls='-', alpha=0.5)
ax.set_xlim(0, segs[-1].linear_k[-1])
ax.set_xticks(xticks, labels=xtick_labels)
ax.axhline(material.bands.fermi_energy, c='tab:red', ls='--')
ax.set_ylim(material.bands.fermi_energy - 10, material.bands.fermi_energy + 10)
# %%
