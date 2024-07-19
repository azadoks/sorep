# %%
import typing as ty

from aiida import load_profile, orm
from upf_tools import UPFDict

load_profile("sorep-tcm")


# %% Specify pseudo family
# %% Add number of pseudo-atomic wavefunctions extra
def _number_of_wfc(upf: UPFDict) -> int:
    return sum([2 * int(chi["l"]) + 1 for chi in upf["pswfc"]["chi"]])


def _number_of_proj(upf: UPFDict) -> int:
    return sum([2 * int(beta["angular_momentum"]) + 1 for beta in upf["nonlocal"]["beta"]])


def _pseudo_type(upf: UPFDict) -> str:
    header = upf["header"]
    if "pseudo_type" in header:
        return header["pseudo_type"]
    else:
        if header.get("is_paw") in (True, "T"):
            return "PAW"
        if header.get("is_ultrasoft") in (True, "T"):
            return "USPP"
        return "unknown"


def main(pseudo_family_label: str, getters: ty.Dict[str, ty.Callable[[UPFDict], ty.Any]]) -> None:
    qb = orm.QueryBuilder()
    qb.append(orm.Group, filters={"label": pseudo_family_label}, tag="g")
    qb.append(orm.Node, with_group="g", project="*")
    nodes: ty.List[ty.Any] = qb.all(flat=True)
    for node in nodes:
        upf: UPFDict = UPFDict.from_str(node.get_content())
        extras: ty.Dict[str, ty.Any] = {key: getter(upf) for key, getter in getters.items()}
        print(f"Adding {extras} to {node} ({node.element})")
        node.base.extras.set_many(extras)


# %%
PSEUDO_FAMILY_LABEL = "PseudoDojo/0.4/PBE/SR/standard/upf"
GETTERS = {
    "number_of_wfc": _number_of_wfc,
    "number_of_proj": _number_of_proj,
    "pseudo_type": _pseudo_type,
}

if __name__ == "__main__":
    main(PSEUDO_FAMILY_LABEL, GETTERS)
# %%
