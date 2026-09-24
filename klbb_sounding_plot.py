"""KLBB 2016-06-01 15Z VADs against the bracketing AMA and MAF soundings."""
import json, os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, INK, INK2, GRID, SURF, BAND = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#dcdad4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
d = np.load(os.path.join(HERE, "KLBB_results.npz")); z = d["z"]

def sounding(fn):
    p = json.load(open(os.path.join(HERE, fn)))["profiles"][0]["profile"]
    h = np.array([l["hght"] for l in p], float)
    s = np.array([l["sknt"] if l["sknt"] is not None else np.nan for l in p], float)
    ok = np.isfinite(h) & np.isfinite(s); h, s = h[ok], s[ok] * 0.514444
    agl = h - h[0]; o = np.argsort(agl)
    return np.interp(z, agl[o], s[o], left=np.nan, right=np.nan)

snd = np.array([sounding(f) for f in ("raob_KAMA.json", "raob_KAMA_00z.json", "raob_KMAF.json", "raob_KMAF_00z.json")])
lo, hi, mean = snd.min(0), snd.max(0), snd.mean(0)

def stats(k):
    e = d[k] - mean
    return np.nanmean(e), np.nanmean(np.abs(e))

zk = z / 1000
fig, ax = plt.subplots(figsize=(7.2, 7.4))
ax.fill_betweenx(zk, lo, hi, color=BAND, lw=0, label="Soundings, range of 4 (AMA, MAF; 12Z and 00Z)")
ax.plot(mean, zk, color=INK, lw=2.2, label="Soundings, mean")
b, m = stats("new_speed")
ax.plot(d["new_speed"], zk, color=BLUE, lw=2, marker="o", ms=5, mec=SURF, mew=1, label=f"vad_michelson, this PR  (bias {b:+.1f}, MAE {m:.1f} m/s)")
b, m = stats("browning_speed")
ax.plot(d["browning_speed"], zk, color=INK2, lw=1.8, ls=(0, (4, 3)), label=f"vad_browning  (bias {b:+.1f}, MAE {m:.1f} m/s)")
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlim(0, 15); ax.set_ylim(0, 6.1)
ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Height above ground (km)")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.07, 0.0), frameon=False, fontsize=9)
fig.suptitle("KLBB: this PR's vad_michelson is closer to the soundings", x=0.03, ha="left", fontsize=13, color=INK, weight="bold")
fig.text(0.03, 0.925, "KLBB 2016-06-01 15Z, median over velocity sweeps. Soundings are 170-180 km away\n"
         "and 3-9 hours from the radar time, so they are a rough reference. Bias and MAE vs their mean.",
         fontsize=9.5, color=INK2, va="top")
fig.tight_layout(rect=(0, 0.13, 1, 0.87))
fig.savefig(os.path.join(HERE, "vad_1488_klbb_soundings.png"), dpi=160)
print("ok")
