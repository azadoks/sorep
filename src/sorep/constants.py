"""Various constants."""

__all__ = (
    "BOHR_TO_ANGSTROM",
    "ANGSTROM_TO_BOHR",
    "RY_TO_EV",
    "EV_TO_RY",
    "HARTREE_TO_EV",
    "EV_TO_HARTREE",
    "KELVIN_TO_EV",
    "EV_TO_KELVIN",
    "ROOM_TEMP_EV",
    "MAX_EXPONENT",
)

BOHR_TO_ANGSTROM = 0.52917720859
ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM
RY_TO_EV = 13.6056917253
EV_TO_RY = 1 / RY_TO_EV
HARTREE_TO_EV = 2 * RY_TO_EV
EV_TO_HARTREE = 1 / HARTREE_TO_EV
KELVIN_TO_EV = 8.61732814974056e-05
EV_TO_KELVIN = 1 / KELVIN_TO_EV
ROOM_TEMP_EV = 300 * KELVIN_TO_EV
MAX_EXPONENT = 200.0
