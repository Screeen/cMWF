import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.collections import PolyCollection
from src import utils as u
u.set_plot_options(use_tex=True)

# Open pickled image from file
common_path = r"/Users/giovannibologni/Documents/TU-Delft/Code-parent/CyclicBeamforming/src/figs/"
delta_path = Path(common_path) / "2025-07-15" / "16h53"
times_path = Path(common_path) / "2025-07-15" / "16h54"
fig_delta_path = delta_path / "figs_pkl" / "DeltaSI-SDR_vs_Noise_inharmonicity_percent_00.pkl"
fig_times_path = times_path / "figs_pkl" / "DeltaSI-SDR_vs_Noise_inharmonicity_percent_00.pkl"

cmvdr_delta_label = r"cMVDR ($\Delta$ strategy)"
cmvdr_times_label = r"cMVDR ($\times$ strategy)"

title_font_size = 9
font_size = 7
font_size_ticks_labels = 7
legend_font_size = 6

with open(fig_delta_path, 'rb') as f:
    fig_delta = pickle.load(f)

with open(fig_times_path, 'rb') as f:
    fig_times = pickle.load(f)

# fig_delta: has two lines. leave "MVDR" untouched, and rename the other line to "cMVDR (Delta)"
# fig_times: has two lines. remove "MVDR" and rename the other line to "cMVDR (Times)"
# Rename lines in fig_delta
for line in fig_delta.axes[0].get_lines():
    if line.get_label() == "MVDR":
        continue  # Leave MVDR untouched
    elif line.get_label() == "cMVDR (prop.)":
        line.set_label(cmvdr_delta_label)  # Rename cMVDR (prop.) to cMVDR (Delta)
    else:
        print(f"Unexpected label in fig_delta: {line.get_label()}")

# Rename lines in fig_times
for line in fig_times.axes[0].get_lines():
    if line.get_label() == "MVDR":
        line.remove()  # Remove MVDR line
    elif line.get_label() == "cMVDR (prop.)":
        line.set_label(cmvdr_times_label)  # Rename cMVDR (prop.) to cMVDR (Times)
    else:
        print(f"Unexpected label in fig_times: {line.get_label()}")

# Create new figure with same size and DPI
# x_size = u.get_plot_width_double_column_latex() / 2  # two figs should fit one column (4 figs in one row)
#             fig = plt.figure(figsize=(x_size, 0.8 * x_size), dpi=150, constrained_layout=True)

x_size = u.get_plot_width_double_column_latex() / 2  # two figs should fit one column (4 figs in one row)
new_fig = plt.figure(figsize=(x_size, 0.8 * x_size), dpi=fig_delta.dpi, constrained_layout=True)
new_ax = new_fig.add_subplot(1, 1, 1)

# Make a new figure with the same axes (they are the same in fig_delta and fig_times)
# but including the lines from both figures
for fig in [fig_delta, fig_times]:
    ax = fig.axes[0]

    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            verts = collection.get_paths()[0].vertices
            x_vals = verts[:, 0]
            y_vals = verts[:, 1]

            new_ax.fill_between(x_vals, y_vals,
                                facecolor=collection.get_facecolor()[0],
                                alpha=collection.get_alpha(),
                                linewidth=0)

    for line in ax.get_lines():
        label = line.get_label()

        # Default style
        plot_kwargs = dict(
            linestyle=line.get_linestyle(),
            color=line.get_color(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            markerfacecolor=line.get_markerfacecolor(),
            markeredgecolor=line.get_markeredgecolor(),
            markeredgewidth=line.get_markeredgewidth(),
            alpha=line.get_alpha(),
        )

        # Modify style for specific lines
        if label == cmvdr_times_label:
            plot_kwargs["marker"] = 'o'
            plot_kwargs["color"] = 'green'
            plot_kwargs["markerfacecolor"] = 'none'
            plot_kwargs["markeredgecolor"] = 'green'

        new_ax.plot(*line.get_data(), label=label, **plot_kwargs)

# Copy full tick and grid styling from ref_ax
ref_ax = fig_delta.axes[0]

# Copy title, labels, limits
new_ax.set_title(ref_ax.get_title())
new_ax.set_xlabel(ref_ax.get_xlabel())
new_ax.set_ylabel(ref_ax.get_ylabel())

# Copy grid and ticks
new_ax.grid(ref_ax.axison)  # Copy grid state
new_ax.tick_params(axis='both', which='major')  # Copy tick parameters

# Match spine linewidths and visibility
for spine_name in ref_ax.spines:
    new_ax.spines[spine_name].set_linewidth(ref_ax.spines[spine_name].get_linewidth())
    new_ax.spines[spine_name].set_visible(ref_ax.spines[spine_name].get_visible())

# Assume ref_ax = fig_delta.axes[0] (as both figures share same layout)
ref_ax = fig_delta.axes[0]

# Copy xscale, symlog, and linthresh if present
xscale = ref_ax.get_xscale()
if xscale == 'symlog':
    # Get linthresh from locator
    locator = ref_ax.xaxis.get_major_locator()
    linthresh = getattr(locator, 'linthresh', 1e-2)  # fallback value
    new_ax.set_xscale('symlog', linthresh=linthresh)
else:
    new_ax.set_xscale(xscale)

# Copy tick positions and labels
new_ax.set_xticks(ref_ax.get_xticks())
new_ax.set_xticklabels([tick.get_text() for tick in ref_ax.get_xticklabels()], fontsize=font_size_ticks_labels)
new_ax.set_yticks(ref_ax.get_yticks())
new_ax.set_yticklabels([tick.get_text() for tick in ref_ax.get_yticklabels()], fontsize=font_size_ticks_labels)

# Copy minor locators (especially for grids)
new_ax.xaxis.set_minor_locator(ref_ax.xaxis.get_minor_locator())
new_ax.yaxis.set_minor_locator(ref_ax.yaxis.get_minor_locator())

# Copy grid config
new_ax.grid(True, which='both')
new_ax.grid(which='major', color='#CCCCCC')
new_ax.xaxis.grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=0.3)
new_ax.yaxis.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.3)

# Copy tick parameters
new_ax.tick_params(axis='both', labelsize=7, pad=2)

# Flip x-axis if needed
if ref_ax.xaxis_inverted():
    new_ax.invert_xaxis()

new_ax.set_xlim(ref_ax.get_xlim())
new_ax.set_ylim(ref_ax.get_ylim())

# Match legend font size and placement
new_fig.legend(fontsize=legend_font_size, handletextpad=0.4, borderaxespad=0.3, ncols=3,
               columnspacing=1.2, loc='outside upper center')

# Match axis label font sizes
new_ax.set_xlabel(ref_ax.get_xlabel(), fontsize=font_size, labelpad=2)
new_ax.set_ylabel(ref_ax.get_ylabel(), fontsize=font_size, labelpad=2)

# Match title
new_ax.set_title(ref_ax.get_title(), fontsize=title_font_size)

# No tight layout as it is already constrained
new_fig.show()

# Save the new figure
new_fig.savefig(delta_path / "DeltaSI-SDR_vs_Noise_inharmonicity_percent_00_combined_legend.pdf",
                dpi=fig_delta.dpi, bbox_inches='tight', pad_inches=0.1)




