"""Compute original vs fixed vad_michelson (and vad_browning) on real NEXRAD data."""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYART_QUIET"] = "1"
import pyart  # noqa: E402
from pyart.retrieve import vad as vad_new  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vad_old  # noqa: E402

NAME = sys.argv[1]
FILE = os.path.expanduser("~/Library/Caches/open-radar-data/" + NAME)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), NAME.split("_")[0][:4] + "_results.npz")


def quiet(fn, *a, **k):
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        return fn(*a, **k)
    finally:
        sys.stdout.close()
        sys.stdout = stdout


radar = pyart.io.read(FILE)
print("read", radar.nsweeps, "sweeps", list(radar.fields))

# Same preprocessing as the #1488 reproduction script
gf = pyart.filters.GateFilter(radar)
gf.exclude_transition()
gf.exclude_invalid("velocity")
gf.exclude_invalid("reflectivity")
gf.exclude_outside("reflectivity", 0, 80)
cor = pyart.correct.dealias_region_based(radar, gatefilter=gf)
cor["data"] = np.ma.masked_where(gf.gate_excluded, cor["data"])
radar.add_field("corr_velocity", cor, True)

zlevels = np.arange(250.0, 6001.0, 250.0)
res = {"z": zlevels, "alt": float(radar.altitude["data"][0])}

# Only sweeps with velocity data (NEXRAD split cuts: surveillance sweeps have none)
sweeps = [
    i
    for i in range(radar.nsweeps)
    if radar.get_field(i, "corr_velocity").count() > 0
]
res["sweeps"] = np.array(sweeps)
res["elev"] = radar.fixed_angle["data"][sweeps]

for name, fn in (
    ("old", vad_old.vad_michelson),
    ("new", vad_new.vad_michelson),
    ("browning", vad_new.vad_browning),
):
    us, vs = [], []
    for i in sweeps:
        one = radar.extract_sweeps([i])
        try:
            v = quiet(fn, one, "corr_velocity", z_want=zlevels)
        except ValueError:
            continue
        us.append(np.ma.filled(v.u_wind, np.nan).astype(float))
        vs.append(np.ma.filled(v.v_wind, np.nan).astype(float))
    u = np.nanmedian(np.array(us), axis=0)
    v = np.nanmedian(np.array(vs), axis=0)
    res[f"{name}_speed"] = np.hypot(u, v)
    res[f"{name}_dir"] = np.rad2deg(np.arctan2(-u, -v)) % 360
    print(name, np.round(res[f"{name}_speed"], 1))

# Per-gate comparison: speed ratio old/new against the share of valid rays
fracs, ratios, elevs = [], [], []
for i in sweeps:
    s, e = radar.get_start_end(i)
    e = e + 1
    if (e - s) % 2:
        e -= 1
    vel = radar.fields["corr_velocity"]["data"][s:e]
    az = radar.azimuth["data"][s:e]
    el = radar.fixed_angle["data"][i]
    with np.errstate(all="ignore"):
        sp_old, _ = vad_old._vad_calculation_m(vel, az, el)
        sp_new = vad_new._vad_calculation_m(vel, az, el)[0]
    valid = 1.0 - np.ma.getmaskarray(vel).mean(axis=0)
    sp_old = np.ravel(np.ma.filled(sp_old, np.nan))
    ok = (valid >= 0.05) & np.isfinite(sp_old) & np.isfinite(sp_new) & (sp_new > 3.0)
    fracs.append(valid[ok])
    ratios.append(sp_old[ok] / sp_new[ok])
    elevs.append(np.full(ok.sum(), el))
res["gate_valid"] = np.concatenate(fracs)
res["gate_ratio"] = np.concatenate(ratios)
res["gate_elev"] = np.concatenate(elevs)
print("gates", res["gate_valid"].size)

# Lowest velocity sweep for a context map
i0 = sweeps[0]
res["ppi_sweep"] = i0
np.savez(OUT, **res)
print("saved", OUT)
