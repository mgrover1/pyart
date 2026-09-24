"""Experiment: does a minimum-valid-rays rule (as in vad_browning) remove the Michelson noise?"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore"); os.environ["PYART_QUIET"] = "1"
import pyart
from pyart.retrieve import vad as vad_new

HERE = os.path.dirname(os.path.abspath(__file__))
l3 = np.load(os.path.join(HERE, "keox", "l3", "keox.npz"), allow_pickle=False)
radar = pyart.io.read(os.path.join(HERE, "keox", "l2", "KEOX20231121_203035_V06.ar2v"))
gf = pyart.filters.GateFilter(radar)
gf.exclude_transition(); gf.exclude_invalid("velocity"); gf.exclude_invalid("reflectivity")
gf.exclude_outside("reflectivity", 0, 80)
radar.add_field("corr_velocity", pyart.correct.dealias_region_based(radar, gatefilter=gf), True)
orig = vad_new._vad_calculation_m

def with_min(nmin):
    def calc(velocity_field, azimuth, elevation):
        speed, angle = orig(velocity_field, azimuth, elevation)
        valid = np.ma.count(np.ma.masked_invalid(np.ma.asarray(velocity_field, dtype=float)), axis=0)
        speed = np.where(valid < nmin, np.nan, speed); angle = np.where(valid < nmin, np.nan, angle)
        return speed, angle
    return calc

def run(pct=90):
    us, vs = [], []
    for i in range(radar.nsweeps):
        one = radar.extract_sweeps([i])
        so = sys.stdout; sys.stdout = open(os.devnull, "w")
        try:
            v = vad_new.vad_michelson(one, "corr_velocity", z_want=l3["height"])
        except ValueError:
            continue
        finally:
            sys.stdout.close(); sys.stdout = so
        us.append(np.ma.filled(v.u_wind, np.nan).astype(float)); vs.append(np.ma.filled(v.v_wind, np.nan).astype(float))
    return np.hypot(np.nanpercentile(us, pct, axis=0), np.nanpercentile(vs, pct, axis=0)) * 1.94384

def rough(x): return np.nanmean(np.abs(np.diff(x))) / np.nanmean(x)
res = {}
for nmin in (0, 16, 50, 100):
    vad_new._vad_calculation_m = with_min(nmin) if nmin else orig
    s = run(); e = s - l3["speed"]; res[f"min{nmin}"] = s
    print(f"min valid rays {nmin:3d}: bias {np.nanmean(e):+5.1f} kt  MAE {np.nanmean(np.abs(e)):5.1f} kt  jump/mean {rough(s):.2f}  heights with data {np.isfinite(s).sum()}/22")
np.savez(os.path.join(HERE, "KEOX_minrays.npz"), z=l3["height"], l3=l3["speed"], **res)
