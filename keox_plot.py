"""Figure: the #1488 KEOX case against the NEXRAD Level 3 VAD product."""

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
    }
)

d = np.load(os.path.join(HERE, "KEOX_results.npz"))
z = d["z"] / 1000.0


def mae(key):
    return np.nanmean(np.abs(d[key] - d["l3"]))


def bias(key):
    return np.nanmean(d[key] - d["l3"])


fig, ax = plt.subplots(figsize=(7.2, 7.6))
ax.plot(d["l3"], z, color=INK, lw=2.4, marker="s", ms=5, mec=SURF, mew=1, label="NEXRAD Level 3 VAD")
ax.plot(d["browning_p90"], z, color=INK2, lw=1.8, ls=(0, (4, 3)),
        label=f"vad_browning  (bias {bias('browning_p90'):+.0f} kt, MAE {mae('browning_p90'):.0f} kt)")
ax.plot(d["old_p90"], z, color=ORANGE, lw=2, marker="o", ms=5, mec=SURF, mew=1,
        label=f"vad_michelson, original  (bias {bias('old_p90'):+.0f} kt, MAE {mae('old_p90'):.0f} kt)")
ax.plot(d["new_p90"], z, color=BLUE, lw=2, marker="o", ms=5, mec=SURF, mew=1,
        label=f"vad_michelson, fixed, valid_ray_min=16  (bias {bias('new_p90'):+.0f} kt, MAE {mae('new_p90'):.0f} kt)")
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_xlim(0, 115)
ax.set_ylim(0, 8)
ax.set_xlabel("Wind speed (kt)")
ax.set_ylabel("Height (km)")
handles, labels = ax.get_legend_handles_labels()
order = [0, 3, 2, 1]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=1, frameon=False, fontsize=9)
fig.suptitle("The #1488 case: fixed vad_michelson removes the low bias", x=0.03, ha="left",
             fontsize=13, color=INK, weight="bold")
fig.text(0.03, 0.915,
         "KEOX 2023-11-21 2030Z, reporter's data and script (gate filter, dealiasing,\n"
         "90th percentile over sweeps). Bias and MAE against the Level 3 product, 22 heights.\n"
         "Fixed Michelson (default valid_ray_min=16) is no longer biased low; one height (1.2 km) still spikes.",
         fontsize=9.5, color=INK2, va="top")
fig.tight_layout(rect=(0, 0.14, 1, 0.86))
fig.savefig(os.path.join(HERE, "vad_1488_keox_l3.png"), dpi=160)
print("ok")
