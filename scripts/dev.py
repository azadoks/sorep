# %%
import pathlib as pl

from findiff import FinDiff
import numpy as np
import scipy as sp

import sorep
from sorep.constants import *
from sorep.smearing import smearing_from_name

# %%
material = sorep.MaterialData.from_dir(list(pl.Path("../data/mc3d/").glob('*/bands/'))[37])

segments = material.bands.path_segments
segment = segments[1]

cbm = material.bands.cbm
vbm = material.bands.vbm
smearing_type = 'fermi-dirac'
smearing_width = ROOM_TEMP_EV
acc = 2

dk = np.mean(np.diff(segment.linear_k))
assert np.allclose(np.diff(segment.linear_k), dk)

# Convert units to Hartree atomic so that the effective mass is in atomic units (electron masses)
dk /= ANGSTROM_TO_BOHR
cbm *= EV_TO_HARTREE
vbm *= EV_TO_HARTREE
bands = segment.bands * EV_TO_HARTREE
fermi_energy = segment.fermi_energy * EV_TO_HARTREE
smearing_width = smearing_width * EV_TO_HARTREE

smearing_cls = smearing_from_name(smearing_type)
d2_dk2 = FinDiff(0, dk, 2, acc=acc)

try:
    conduction = np.hstack([
        bands_spin[:, np.all(bands_spin > fermi_energy, axis=0)]
        for bands_spin in bands
    ])
    conduction_curvature = d2_dk2(conduction)
    conduction_occupations = smearing_cls(
        cbm, smearing_width).occupation(conduction)
    conduction_num = sp.integrate.simpson(y=conduction_occupations,
                                          dx=dk,
                                          axis=0).sum()
    conduction_denom = sp.integrate.simpson(y=(conduction_occupations *
                                                conduction_curvature),
                                            dx=dk,
                                            axis=0).sum()
    electron_effective_mass = conduction_num / conduction_denom

    valence = np.hstack([
        bands_spin[:, np.all(bands_spin < fermi_energy, axis=0)]
        for bands_spin in bands
    ])
    valence_curvature = d2_dk2(valence)
    valence_occupations = smearing_cls(-vbm, smearing_width).occupation(-valence)
    valence_num = sp.integrate.simpson(y=valence_occupations,
                                        dx=dk,
                                        axis=0).sum()
    valence_denom = sp.integrate.simpson(y=(valence_occupations *
                                            valence_curvature),
                                            dx=dk,
                                            axis=0).sum()
    hole_effective_mass = valence_num / valence_denom
except IndexError:
    electron_effective_mass = np.nan
    hole_effective_mass = np.nan

(electron_effective_mass, hole_effective_mass)
# %%
explicit_conduction = []
explicit_valence = []
for i_spin in range(bands.shape[0]):
    for i_band in range(bands.shape[2]):
        if np.all(bands[i_spin, :, i_band] > fermi_energy):
            explicit_conduction.append(bands[i_spin, :, i_band])
        if np.all(bands[i_spin, :, i_band] < fermi_energy):
            explicit_valence.append(bands[i_spin, :, i_band])
explicit_conduction = np.array(explicit_conduction).T
explicit_valence = np.array(explicit_valence).T

# %%
