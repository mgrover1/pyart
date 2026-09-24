"""Figures for the vad_michelson masked-gate fix (#1488)."""

import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CASES = [
    ("KLIX", "KLIX  2005-08-28 18:01Z\nHurricane Katrina outer bands"),
    ("KLBB", "KLBB  2016-06-01 15:00Z\nLubbock, TX"),
]

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


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# Figure 1: wind speed profiles
fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
for ax, (key, title) in zip(axes, CASES):
    d = np.load(os.path.join(HERE, f"{key}_results.npz"))
    z = d["z"] / 1000.0
    ax.plot(d["browning_speed"], z, color=INK2, lw=2, ls=(0, (4, 3)), label="vad_browning (reference)")
    ax.plot(d["old_speed"], z, color=ORANGE, lw=2, marker="o", ms=5, mec=SURF, mew=1, label="vad_michelson, original")
    ax.plot(d["new_speed"], z, color=BLUE, lw=2, marker="o", ms=5, mec=SURF, mew=1, label="vad_michelson, fixed")
    style(ax)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_xlim(left=0)
axes[0].set_ylabel("Height above radar (km)")
axes[0].set_ylim(0, 6.1)
handles, labels = axes[0].get_legend_handles_labels()
order = [2, 1, 0]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3, frameon=False, fontsize=9.5)
fig.suptitle(
    "Fixed vad_michelson vs vad_browning on two more NEXRAD cases",
    x=0.06, ha="left", fontsize=13, color=INK, weight="bold",
)
fig.text(
    0.06, 0.915,
    "Real NEXRAD Level 2 volumes, gate-filtered and dealiased as in #1488. Median over velocity sweeps.",
    fontsize=9.5, color=INK2,
)
fig.tight_layout(rect=(0, 0.07, 1, 0.925))
fig.savefig(os.path.join(HERE, "vad_1488_profiles.png"), dpi=160)

# Figure 2: per-gate bias against the share of valid rays
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
bins = np.linspace(0.05, 1.0, 11)
for ax, (key, title) in zip(axes, CASES):
    d = np.load(os.path.join(HERE, f"{key}_results.npz"))
    x, y = d["gate_valid"], d["gate_ratio"]
    k = np.isfinite(y) & (np.abs(y) < 5)
    x, y = x[k], y[k]
    ax.plot([0, 1], [0, 1], color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.annotate("ratio = valid share", xy=(0.9, 0.9), xytext=(0.93, 0.62), color=INK2, fontsize=9, ha="right", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.scatter(x, y, s=9, color=BLUE, alpha=0.3, lw=0)
    idx = np.digitize(x, bins)
    cx = [np.median(x[idx == i]) for i in range(1, len(bins)) if (idx == i).sum() >= 10]
    cy = [np.median(y[idx == i]) for i in range(1, len(bins)) if (idx == i).sum() >= 10]
    ax.plot(cx, cy, color=INK, lw=0, marker="o", ms=7, mec=SURF, mew=1.5, label="bin median")
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.03, 0.97, f"{k.sum():,} range gates   r = {r:.2f}", transform=ax.transAxes, va="top", color=INK2, fontsize=9.5)
    style(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.3)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("Share of rays with valid velocity at that range")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0].set_ylabel("Original speed / fixed speed")
axes[0].legend(loc="lower right", frameon=False, fontsize=9)
fig.suptitle(
    "The original code scaled wind speed by the share of valid gates",
    x=0.06, ha="left", fontsize=13, color=INK, weight="bold",
)
fig.text(
    0.06, 0.905,
    "Each dot is one range gate of one sweep. 30% valid gates gave about 30% of the true speed.",
    fontsize=9.5, color=INK2,
)
fig.tight_layout(rect=(0, 0, 1, 0.915))
fig.savefig(os.path.join(HERE, "vad_1488_bias_vs_coverage.png"), dpi=160)
print("ok")
