# %%
import matplotlib.pyplot as plt

import sorep


# %%
def plot_segments(material):
    segments = material.bands.path_segments
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
        linear_k = material.bands.linear_k
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
