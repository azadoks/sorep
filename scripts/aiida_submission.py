#!/usr/bin/env runaiida
import math
import sys

from aiida import load_profile, orm
from aiida.plugins import WorkflowFactory
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_submission_controller import FromGroupSubmissionController

DRY_RUN = False
MAX_CONCURRENT = 2
CODE_LABEL = "pw-7.2-sorep@tpc50"
# PSEUDO_FAMILY_LABEL = "SSSP/1.3/PBE/efficiency"
PSEUDO_FAMILY_LABEL = "PseudoDojo/0.4/PBE/SR/standard/upf"

load_profile("sorep-tcm")


# Either use all of the pseudo-atomic wavefunctions for band initialization (if there are more)
# wfcs than there would be bands otherwise), or use all the wfcs + the minimum required number
# of random wavefunctions
# e.g. : a system w/ 43 electrons and 29 atomic wavefunctions
#      n_elec = 43
#      nbnd_fixed = 22
#      nbnd_smeared = 27  # ceil(1.2 * 22) === ceil(26.4)
#      nbnd_wfc = 27
# e.g. : the same system, but 25 atomic wavefunctions
#     ...
#     nbnd_wfc = 25  # will warn about possible non-zero occupations
# e.g. : the same system, but 21 atomic wavefunctions
#     ...
#     nbnd_wfc = 22  # will warn about 1 random wfc and about non-zero occupations
class SingleShotSubmissionController(FromGroupSubmissionController):
    def __init__(self, code_label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._code = orm.load_code(code_label)
        self._process_class = WorkflowFactory("quantumespresso.pw.base")
        self._pseudo_family = orm.load_group(PSEUDO_FAMILY_LABEL)

    def get_extra_unique_keys(self):
        """Return a tuple of the keys of the unique extras that will be used to uniquely identify your workchains."""
        return ("source.id",)

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """Return inputs and process class for the submission of this specific process."""
        structure = self.get_parent_node_from_extras(extras_values)
        assert isinstance(structure, orm.StructureData)
        pseudos = self._pseudo_family.get_pseudos(structure=structure)

        # Avoid call to create_kpoints_from_density, allowing much higher throughput by avoiding
        # `RecursionError`s and therefore process exceptions.
        # See https://github.com/aiidateam/aiida-core/issues/4876#issuecomment-1574344017
        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(structure)
        kpoints.set_kpoints_mesh_from_density(0.15)  # Taken from aiida-quantumespresso default protocol
        kpoints.store()

        n_elec = sum([pseudos[site.kind_name].z_valence for site in structure.sites])
        number_of_wfc_total = sum(
            [pseudos[site.kind_name].base.extras.all["number_of_wfc"] for site in structure.sites]
        )
        nbnd_fixed = math.ceil(n_elec / 2)  # No. of bands is no. of electrons / 2 for non-spin-polarized systems
        nbnd_smeared = max(
            nbnd_fixed + 4, math.ceil(1.2 * nbnd_fixed)
        )  # Taken from pw.x input documentation, smearing case
        # Use one of
        #  - QE default for smearing (max(nelec // 2 + 4, ceil(1.2 * nelec // 2)))
        #  - Total number of wfc (if it is < QE default for smearing and > nelec // 2)
        #  - nelec // 2 (if the total number of wfc is < nelec // 2)
        nbnd_wfc = max(min(number_of_wfc_total, nbnd_smeared), nbnd_fixed)
        # If random wavefunctions are required, warn the user
        if nbnd_wfc < nbnd_fixed:
            print(
                f"StructureData<{structure.pk}> {number_of_wfc_total:4d} atomic wfcs"
                f" available, but {nbnd_fixed:4d} bands are needed -> "
                f"{(nbnd_fixed - number_of_wfc_total):4d} random wfcs"
            )
        if nbnd_wfc < nbnd_smeared:
            print(
                f"StructureData<{structure.pk}> using fewer bands than recommended "
                f"{nbnd_wfc:4d} vs. {nbnd_smeared:4d}, which might result in non-zero "
                f"occupation of the highest band"
            )

        conv_thr_per_atom = 0.2e-9  # Taken from aiida-quantumespresso default protocol
        conv_thr = conv_thr_per_atom * len(structure.sites)  # Taken from aiida-quantumespresso default protocol
        diago_thr_init = conv_thr / n_elec / 10  # Taken from pw.x input documentation

        # We know that we're doing sketchy shit with the number of bands
        handler_overrides = orm.Dict(dict={"sanity_check_insufficient_bands": {"enabled": False}})

        overrides = {
            "clean_workdir": True,
            "handler_overrides": handler_overrides,
            "pseudo_family": self._pseudo_family.label,
            "kpoints": kpoints,
            "pw": {
                "metadata": {
                    "options": {
                        "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": 1},
                        "max_wallclock_seconds": 3600,  # 1 hour
                        "withmpi": False,
                    }
                },
                "parameters": {
                    "CONTROL": {
                        "tprnfor": False,  # don't compute forces
                        "tstress": False,  # don't comute stresses
                        "disk_io": "minimal",  # only write XML (>=7.3)
                        # "disk_io": "nowf",  # only write XML (<7.3)
                    },
                    "SYSTEM": {
                        "nbnd": nbnd_wfc,
                        "smearing": "gauss",
                    },
                    "ELECTRONS": {
                        "electron_maxstep": 0,
                        "diagonalization": "david",
                        "diago_thr_init": diago_thr_init,
                        "diago_full_acc": True,
                        "startingwfc": "atomic",
                        "startingpot": "atomic",
                        "scf_must_converge": False,
                    },
                },
            },
        }
        inputs = self._process_class.get_builder_from_protocol(
            self._code,
            structure,
            overrides=overrides,
            electronic_type=ElectronicType.METAL,
            spin_type=SpinType.NONE,
            initial_magnetic_moments=None,
            options=None,
        )
        return inputs


if __name__ == "__main__":
    STRUCTURES_GROUP_LABEL = "nomad2018tcm/test/structures"
    WORKFLOWS_GROUP_LABEL = "nomad2018tcm/test/workflows/single_shot"

    controller = SingleShotSubmissionController(
        parent_group_label=STRUCTURES_GROUP_LABEL,
        code_label=CODE_LABEL,
        group_label=WORKFLOWS_GROUP_LABEL,
        max_concurrent=MAX_CONCURRENT,
    )

    print("Already run    :", controller.num_already_run)
    print("Max concurrent :", controller.max_concurrent)
    print("Available slots:", controller.num_available_slots)
    print("Active slots   :", controller.num_active_slots)
    print("Still to run   :", controller.num_to_run)
    print()

    run_processes = controller.submit_new_batch(dry_run=DRY_RUN)
    for run_process_extras, run_process in run_processes.items():
        if run_process is None:
            print(f"{run_process_extras} --> To be run")
        else:
            print(f"{run_process_extras} --> PK = {run_process.pk}")

    print()
