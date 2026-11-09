"""iono_b_lib -- helpers for the NISAR Frequency-B (5 MHz) dispersive/ionosphere
workflow.  Stage 0 (A/B parameter table) and Stage 1 (stack-wise registration
of B onto the B master using A's alignment).

Design (see NISAR_freqB_ionosphere_strategy.md, section 6):
  * B is resampled ONCE per date, to the B master, with the C `resamp` mode 5
    (same Doppler-aware resampler as A), using offset grids DERIVED FROM A's
    r.grd/a.grd -- not re-estimated on the noisy 5 MHz band.
  * The A range offset is converted to a slant-range change (metres), then to
    B pixels:
        Delta R = (nA_r - nA_m) + rA * dA
        rB      = (Delta R - (nB_r - nB_m)) / dB
    evaluated at the same ground point (A grid sampled at the A pixel having
    the B master pixel's slant range).  The azimuth grid is reused unchanged
    (A and B share the zero-Doppler time axis; checked via PRF / clock start).
  * resamp mode 5 only reads the grid DATA and its increments (node k <-> master
    pixel k*inc), so the B grid keeps A's rows and rescales the x increment by
    dA/dB so that node ix is at the same slant range as A's node ix.

Everything here is pure numpy except `build_b_grids`, which needs netCDF4 via
gmt_grd_io.  Untested on real B data at the time of writing (sandbox has none).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom_align_lib import read_prm_value, prm_dims  # noqa: E402

C = 299792458.0


def slc_params(prm):
    """Geometry/timing of one SLC from its PRM."""
    def f(key, required=True):
        v = read_prm_value(prm, key)
        if v is None:
            if required:
                raise ValueError(f"{prm}: missing {key}")
            return float("nan")
        return float(v)
    nx, ny = prm_dims(prm)
    fs = f("rng_samp_rate")
    wl = f("radar_wavelength")
    return {
        "prm": prm, "nx": nx, "ny": ny,
        "near_range": f("near_range"),
        "fs": fs, "dr": C / (2.0 * fs),
        "prf": f("PRF"),
        "clock_start": f("SC_clock_start", required=False),
        "wavelength": wl, "freq": C / wl,
        "bandwidth": f("chirp_slope", required=False),   # informational only
    }


def azimuth_pixel_m(prm):
    """Azimuth line spacing on the ground (m), vel/sqrt(1+ht/RE)/PRF, as
    make_gaussian_filter computes it."""
    vel = float(read_prm_value(prm, "SC_vel"))
    ht = float(read_prm_value(prm, "SC_height"))
    re_ = float(read_prm_value(prm, "earth_radius"))
    prf = float(read_prm_value(prm, "PRF"))
    return vel / np.sqrt(1.0 + ht / re_) / prf


def check_ab_timing(a, b, max_lines=0.1):
    """Azimuth grid reuse is only valid when A and B have the same line timing.
    Raise ValueError if PRF differs or the first-line time differs by more
    than `max_lines` lines.  (SC_clock_start is YYYYDDD.fraction in a double,
    ~20 us resolution, hence the 0.1-line default.)"""
    if abs(a["prf"] - b["prf"]) > 1e-6 * a["prf"]:
        raise ValueError(f"A/B PRF differ: {a['prf']} vs {b['prf']} "
                         f"({a['prm']}, {b['prm']})")
    if not (np.isnan(a["clock_start"]) or np.isnan(b["clock_start"])):
        dt = (a["clock_start"] - b["clock_start"]) * 86400.0
        # SC_clock_start is a day-of-year float; sub-line precision needs the
        # difference in seconds
        if abs(dt) * a["prf"] > max_lines:
            raise ValueError(
                f"A/B first-line times differ by {dt * 1e3:.4f} ms "
                f"({dt * a['prf']:.3f} lines): azimuth grids cannot be reused "
                f"unchanged ({a['prm']}, {b['prm']})")
    if a["ny"] != b["ny"]:
        raise ValueError(f"A/B azimuth line counts differ: {a['ny']} vs {b['ny']}")


def b_range_grid(rA, incA_x, mA, aA, mB, aB):
    """B range-offset array (B pixels) from A's (A pixels).

    rA      2-D array (rows = azimuth nodes, cols = range nodes) in A pixels
    incA_x  A grid x increment (A pixels per node)
    mA, aA  slc_params of A master / A repeat;  mB, aB likewise for B.
    Node ix of the output is at the same slant range as node ix of rA, so the
    output has the same shape.  A is sampled at the fractional column
    ix + (nB_m - nA_m)/(dA*incA) (linear, linear extrapolation at the ends)."""
    dA, dB = mA["dr"], mB["dr"]
    ny, nx = rA.shape
    shift = (mB["near_range"] - mA["near_range"]) / (dA * incA_x)
    col = np.arange(nx, dtype=float) + shift
    out = np.empty_like(rA, dtype=np.float64)
    for i in range(ny):
        row = rA[i].astype(np.float64)
        v = np.interp(col, np.arange(nx), row)
        # np.interp clamps; extend linearly beyond the ends
        lo, hi = col < 0, col > nx - 1
        if lo.any() and nx > 1:
            v[lo] = row[0] + (row[1] - row[0]) * col[lo]
        if hi.any() and nx > 1:
            v[hi] = row[-1] + (row[-1] - row[-2]) * (col[hi] - (nx - 1))
        out[i] = v
    d_near_A = aA["near_range"] - mA["near_range"]
    d_near_B = aB["near_range"] - mB["near_range"]
    delta_r = d_near_A + out * dA                  # metres
    return (delta_r - d_near_B) / dB


def build_b_grids(mA_prm, aA_prm, mB_prm, aB_prm, r_grd, a_grd,
                  out_r="r.grd", out_a="a.grd"):
    """Write B-geometry r.grd/a.grd from A's grids (see module docstring).
    Returns a small dict of diagnostics (mean/min/max range offset in B px)."""
    from gmt_grd_io import read_gmt_grd, write_gmt_grd
    mA, aA = slc_params(mA_prm), slc_params(aA_prm)
    mB, aB = slc_params(mB_prm), slc_params(aB_prm)
    check_ab_timing(mA, mB)
    check_ab_timing(aA, aB)
    r, x, y, info = read_gmt_grd(r_grd)
    az, xa, ya, _ = read_gmt_grd(a_grd)
    if r.shape != az.shape:
        raise ValueError(f"{r_grd} and {a_grd} shapes differ: {r.shape} vs {az.shape}")
    incx = info["x_inc"]
    rB = b_range_grid(r, incx, mA, aA, mB, aB)
    scale = mA["dr"] / mB["dr"]
    xB = x * scale
    desc = "B-band offsets derived from A grids (iono_b_lib)"
    write_gmt_grd(out_r, rB.astype(np.float32), xB, y,
                  node_offset=info["node_offset"], title="rB", description=desc)
    write_gmt_grd(out_a, az.astype(np.float32), xB, y,
                  node_offset=info["node_offset"], title="aB", description=desc)
    return {"r_mean": float(np.nanmean(rB)), "r_min": float(np.nanmin(rB)),
            "r_max": float(np.nanmax(rB)), "inc_x_B": float(incx * scale),
            "scale": scale}


def b_decimation(x_inc_a, y_inc_a, d_a, d_b, tol=0.02):
    """Decimation for the B-band `filter` call so that B's output cells have the
    same ground size as A's existing products.

    x_inc_a, y_inc_a  increments of A's phasefilt.grd (pixels of the A SLC)
    d_a, d_b          A and B range pixel sizes (m)
    Returns dict(range_dec, azimuth_dec, exact, note).  range_dec is in B
    pixels.  `exact` is False when A's range step is not an integer number of
    B pixels (or is finer than one B pixel): B is then as close as possible /
    coarser and A must be coarsened to B's grid before comparison.
    `filter`'s explicit-decimation form needs azimuth_dec odd or a multiple of 4
    (PRF > 1000: az_lks = 4 otherwise it divides it into a non-integer)."""
    rb = x_inc_a * d_a / d_b
    rd = max(1, int(round(rb)))
    exact = rb >= 1 and abs(rb - round(rb)) <= tol * max(rb, 1.0)
    note = "" if exact else (
        f"A range step {x_inc_a:g} px = {rb:.3f} B px: not an integer >= 1; "
        f"B uses {rd} px ({rd * d_b:.1f} m vs A's {x_inc_a * d_a:.1f} m)")
    az = int(round(y_inc_a))
    if abs(y_inc_a - az) > 1e-6 or az < 1:
        raise ValueError(f"A azimuth increment {y_inc_a} is not a positive integer")
    if az % 2 == 0 and az % 4 != 0:
        raise ValueError(
            f"A azimuth increment {az} is even but not a multiple of 4: filter's "
            f"explicit-decimation form would compute a non-integer azimuth step")
    return {"range_dec": rd, "azimuth_dec": az, "exact": exact, "note": note}


def ab_table(stems, raw_dir="raw"):
    """Rows (dict) of A/B parameters for each A stem (NSR_<date>A); B stem is
    the same with trailing 'B'.  Missing B files give a row with b=None."""
    rows = []
    for s in stems:
        if not s.endswith("A"):
            raise ValueError(f"expected A stem ending in 'A', got {s}")
        pa = os.path.join(raw_dir, f"{s}.PRM")
        pb = os.path.join(raw_dir, f"{s[:-1]}B.PRM")
        row = {"stem": s, "a": slc_params(pa) if os.path.isfile(pa) else None,
               "b": slc_params(pb) if os.path.isfile(pb) else None}
        if row["a"] and row["b"]:
            try:
                check_ab_timing(row["a"], row["b"])
                row["timing"] = "ok"
            except ValueError as e:
                row["timing"] = str(e)
        else:
            row["timing"] = "missing"
        rows.append(row)
    return rows


def format_table(rows):
    hdr = (f"{'stem':<16}{'fA(GHz)':>9}{'fB(GHz)':>9}{'dA(m)':>8}{'dB(m)':>8}"
           f"{'PRF_A':>9}{'PRF_B':>9}{'nr_A(m)':>13}{'nr_B(m)':>13}"
           f"{'nxA':>7}{'nxB':>7}{'nyA':>7}{'nyB':>7}  timing")
    out = [hdr]
    for r in rows:
        a, b = r["a"], r["b"]
        out.append(
            f"{r['stem']:<16}{(format(a['freq']/1e9,'9.4f') if a else '        -')}"
            f"{(format(b['freq']/1e9,'9.4f') if b else '        -')}"
            f"{(format(a['dr'],'8.3f') if a else '       -')}"
            f"{(format(b['dr'],'8.3f') if b else '       -')}"
            f"{(format(a['prf'],'9.2f') if a else '        -')}"
            f"{(format(b['prf'],'9.2f') if b else '        -')}"
            f"{(format(a['near_range'],'13.2f') if a else '            -')}"
            f"{(format(b['near_range'],'13.2f') if b else '            -')}"
            f"{(a['nx'] if a else '-'):>7}{(b['nx'] if b else '-'):>7}"
            f"{(a['ny'] if a else '-'):>7}{(b['ny'] if b else '-'):>7}  {r['timing']}")
    return "\n".join(out)
