"""Same minimum-valid-rays experiment on KLIX and KLBB, with vad_browning as the reference."""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore"); os.environ["PYART_QUIET"] = "1"
import pyart
from pyart.retrieve import vad as vad_new

orig = vad_new._vad_calculation_m
def with_min(nmin):
    def calc(vf, az, el):
        s, a = orig(vf, az, el)
        n = np.ma.count(np.ma.masked_invalid(np.ma.asarray(vf, dtype=float)), axis=0)
        return np.where(n < nmin, np.nan, s), np.where(n < nmin, np.nan, a)
    return calc

def rough(x): return np.nanmean(np.abs(np.diff(x))) / np.nanmean(x)
for name in ("KLIX20050828_180149.gz", "KLBB20160601_150025_V06"):
    radar = pyart.io.read(os.path.expanduser("~/Library/Caches/open-radar-data/" + name))
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_transition(); gf.exclude_invalid("velocity"); gf.exclude_invalid("reflectivity")
    gf.exclude_outside("reflectivity", 0, 80)
    cor = pyart.correct.dealias_region_based(radar, gatefilter=gf)
    cor["data"] = np.ma.masked_where(gf.gate_excluded, cor["data"])
    radar.add_field("corr_velocity", cor, True)
    z = np.arange(250.0, 4001.0, 250.0)
    sweeps = [i for i in range(radar.nsweeps) if radar.get_field(i, "corr_velocity").count() > 0]
    def run(fn):
        us, vs = [], []
        for i in sweeps:
            so = sys.stdout; sys.stdout = open(os.devnull, "w")
            try:
                v = fn(radar.extract_sweeps([i]), "corr_velocity", z_want=z)
            except ValueError:
                continue
            finally:
                sys.stdout.close(); sys.stdout = so
            us.append(np.ma.filled(v.u_wind, np.nan).astype(float)); vs.append(np.ma.filled(v.v_wind, np.nan).astype(float))
        return np.hypot(np.nanmedian(us, axis=0), np.nanmedian(vs, axis=0))
    ref = run(vad_new.vad_browning)
    print(name[:4], f"browning jump/mean {rough(ref):.2f}")
    for nmin in (0, 16, 50):
        vad_new._vad_calculation_m = with_min(nmin) if nmin else orig
        s = run(vad_new.vad_michelson); vad_new._vad_calculation_m = orig
        e = s - ref
        print(f"  min {nmin:2d}: vs browning bias {np.nanmean(e):+5.2f} m/s  MAE {np.nanmean(np.abs(e)):4.2f} m/s  jump/mean {rough(s):.2f}  heights {np.isfinite(s).sum()}/{z.size}")
