"""Rerun the #1488 reproduction (KEOX 2023-11-21 2030Z) with original and fixed vad_michelson."""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYART_QUIET"] = "1"
import pyart  # noqa: E402
from pyart.retrieve import vad as vad_new  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vad_old  # noqa: E402

l3 = np.load(os.path.join(HERE, "keox", "l3", "keox.npz"), allow_pickle=False)
radar = pyart.io.read(os.path.join(HERE, "keox", "l2", "KEOX20231121_203035_V06.ar2v"))

# Verbatim preprocessing from the #1488 script
gatefilter = pyart.filters.GateFilter(radar)
gatefilter.exclude_transition()
gatefilter.exclude_invalid("velocity")
gatefilter.exclude_invalid("reflectivity")
gatefilter.exclude_outside("reflectivity", 0, 80)
cor_vel = pyart.correct.dealias_region_based(radar, gatefilter=gatefilter)
radar.add_field("corr_velocity", cor_vel, True)


def quiet(fn, *a, **k):
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        return fn(*a, **k)
    finally:
        sys.stdout.close()
        sys.stdout = stdout


def calculate_vad(fn, zlevels, percentile=90):
    # Same per-sweep loop and percentile aggregation as the #1488 script
    u_all, v_all = [], []
    for idx in range(radar.nsweeps):
        one = radar.extract_sweeps([idx])
        try:
            vad = quiet(fn, one, "corr_velocity", z_want=zlevels)
        except ValueError:
            continue
        u_all.append(np.ma.filled(vad.u_wind, np.nan).astype(float))
        v_all.append(np.ma.filled(vad.v_wind, np.nan).astype(float))
    u = np.nanpercentile(np.array(u_all), percentile, axis=0)
    v = np.nanpercentile(np.array(v_all), percentile, axis=0)
    return np.hypot(u, v)


z = l3["height"]
KT = 1.94384
res = {"z": z, "l3": l3["speed"]}
for pct in (90, 50):
    for name, fn in (
        ("old", vad_old.vad_michelson),
        ("new", vad_new.vad_michelson),
        ("browning", vad_new.vad_browning),
    ):
        res[f"{name}_p{pct}"] = calculate_vad(fn, z, pct) * KT
        err = res[f"{name}_p{pct}"] - l3["speed"]
        print(f"p{pct} {name:9s} bias {np.nanmean(err):6.1f} kt  MAE {np.nanmean(np.abs(err)):5.1f} kt")
print("l3 ", res["l3"])
np.savez(os.path.join(HERE, "KEOX_results.npz"), **res)
