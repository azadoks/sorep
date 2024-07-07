"""
Compute the bands screening quantities for TCMs:
- Band gap
- Electron effective mass
- Hole effective mass
"""
import pathlib as pl

# %%
import typing as ty

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import sorep

# %%
material = sorep.MaterialData.from_dir(
    pl.Path('../data/mc3d/known_materials/').glob('*/bands/*').__next__())


# %%
class BandPathSegment:

    def __init__(
        self,
        bands: npt.ArrayLike,
        kpoints: npt.ArrayLike,
        weights: npt.ArrayLike,
        occupations: ty.Optional[npt.ArrayLike] = None,
        fermi_energy: ty.Optional[float] = None,
        start_label: ty.Optional[str] = None,
        stop_label: ty.Optional[str] = None,
    ):
        self.bands = bands
        self.kpoints = kpoints
        self.weights = weights
        self.occupations = occupations
        self.fermi_energy = fermi_energy
        self.start_label = start_label
        self.stop_label = stop_label

    def __repr__(self) -> str:
        return f'BandPathSegment({self.start_label} -> {self.stop_label}, n_kpoints={self.kpoints.shape[0]})'


# %%
def get_segments(bands: sorep.BandStructure) -> list[BandPathSegment]:
    label_numbers = bands.label_numbers.copy()
    labels = bands.labels.copy()

    # If the first k-point does not have a label, give it a placeholder label, and
    # add its index to the indices so that it can start a segment.
    if label_numbers[0] != 0:
        label_numbers.insert(0, 0)
        labels.insert(0, '')
    # Ditto with the last k-point; it needs to end the last segment
    if label_numbers[-1] != bands.n_kpoints - 1:
        label_numbers.append(bands.n_k_points - 1)
        labels.append('')
    n_labels = len(labels)

    # TODO: convert k-points to Cartesian coordinates on loading of BandStructure
    # TODO: construct linearized k-point path

    # Construct the segments
    segments = []
    for (i_from, i_to) in zip(range(n_labels - 1), range(1, n_labels)):
        ik_from = label_numbers[i_from]
        ik_to = label_numbers[i_to]  # This is meant to be inclusive; dealt with below
        segment = BandPathSegment(
            bands=bands.bands[:, ik_from:ik_to + 1].copy(),
            kpoints=bands.kpoints[ik_from:ik_to + 1].copy(),
            weights=bands.weights[ik_from:ik_to + 1].copy(),
            occupations=bands.occupations[:, ik_from:ik_to + 1].copy() if bands.occupations is not None else None,
            fermi_energy=bands.fermi_energy if bands.fermi_energy else None,
            start_label=labels[i_from],
            stop_label=labels[i_to],
        )
        segments.append(segment)

    return segments


# %%
get_segments(material.bands)
# %%
