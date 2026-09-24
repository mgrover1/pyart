"""Which part of the new fit makes KLBB read faster than vad_browning?"""
import os, warnings
import numpy as np
warnings.filterwarnings("ignore"); os.environ["PYART_QUIET"] = "1"
import pyart
from pyart.retrieve import vad as vad
from open_radar_data import DATASETS

r = pyart.io.read(DATASETS.fetch("KLBB20160601_150025_V06"))
gf = pyart.filters.GateFilter(r); gf.exclude_transition(); gf.exclude_invalid("velocity"); gf.exclude_invalid("reflectivity"); gf.exclude_outside("reflectivity", 0, 80)
r.add_field("cv", pyart.correct.dealias_region_based(r, gatefilter=gf), True)
z = np.arange(250.0, 6001.0, 250.0)
sweeps = [i for i in range(r.nsweeps) if r.get_field(i, "cv").count() > 0]
orig_mean = vad._interval_mean

def prof(fn, **kw):
    us, vs = [], []
    for i in sweeps:
        v = fn(r.extract_sweeps([i]), "cv", z_want=z, **kw)
        us.append(np.ma.filled(v.u_wind, np.nan)); vs.append(np.ma.filled(v.v_wind, np.nan))
    return np.hypot(np.nanmedian(us, 0), np.nanmedian(vs, 0))

ref = prof(vad.vad_browning)
def show(name, s): e = s - ref; print(f"{name:40s} bias {np.nanmean(e):+.2f}  MAE {np.nanmean(np.abs(e)):.2f}  heights {np.isfinite(s).sum()}")
show("default (reject 2 m/s, weighted)", prof(vad.vad_michelson))
show("no rejection, weighted", prof(vad.vad_michelson, max_speed_error=None))
vad._interval_mean = lambda d, c, w, weights=None: orig_mean(d, c, w)
show("reject 2 m/s, equal weights", prof(vad.vad_michelson))
show("no rejection, equal weights", prof(vad.vad_michelson, max_speed_error=None))
vad._interval_mean = orig_mean
