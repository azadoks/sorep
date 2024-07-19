# %%
from aiida import load_profile, orm
from ase.io import read

load_profile("sorep-tcm")
# %%
SETS = ["train", "test"]
for set_ in SETS:
    images = read(f"../data/nomad2018tcm_structures_{set_}.xyz", ":")
    group = orm.Group(label=f"nomad2018tcm/{set_}/structures")
    group.store()
    structures = []
    for image in images:
        structure = orm.StructureData(ase=image)
        structure.store()
        structure.base.extras.set(
            "source", {"database": "nomad2018tcm", "version": set_, "id": str(image.info["material_id"])}
        )
        structures.append(structure)
    group.add_nodes(structures)

# %%c
