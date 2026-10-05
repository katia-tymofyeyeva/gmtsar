"""geom_align_lib -- geometric (DEM + orbit) alignment model for NISAR, refined
by a low-order fit to cross-correlation offsets.

Added 2026-10-05. Shared by:
  * fitoffset_ra_geo      (drop-in replacement for `fitoffset_ra 10 10 ...`)
  * diag_geom_vs_xcorr    (read-only diagnostic that motivated this)
  * align_batch_nsr, p2p_stages (select the method via align_method())

Method (see docs/PATHWAY_FORWARD.md, 2026-10-05 entries):
  1. xcorr_py runs exactly as before (constant seed from SAT_baseline_py,
     40x40 grid) and gives MEASURED offsets.
  2. A decimated DEM is projected into the master and the repeat (raw PRM,
     rshift = ashift = 0) with SAT_llt2rat_py; (repeat - master) pixel
     coordinates of the same ground point are the GEOMETRIC offsets, fitted
     with a cubic polynomial in master (range, azimuth) pixels -- the
     align_tops recipe for Sentinel-1.
  3. A low-order CORRECTION (constant + plane by default) is robustly fitted
     to measured - geometric at the high-SNR xcorr points; points that
     disagree with geometry by > outlier_px are excluded.
  4. geometry + correction is evaluated on a regular lattice over the whole
     scene -- including areas where correlation fails -- and run through
     the same `gmt surface ... -rp -I64/64 -T.3` + FLIPUD steps
     fitoffset_ra uses, so r.grd / a.grd have exactly the conventions
     resamp_py mode 5 already expects.

Sign convention (verified in source on both sides): offset = (position in
repeat) - (position in master) of the same feature, in master pixels.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys

import numpy as np

DEFAULT_METHOD = "geo"
VALID_METHODS = ("geo", "xcorr")
ENV_METHOD = "GMTSAR_NSR_ALIGN_METHOD"


# ------------------------------------------------------- method select ---
def config_value(path, key):
    """Value (3rd token) of the first config line whose FIRST token is
    `key` ("key = value" format), or None."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == key:
                return parts[2]
    return None


def align_method(config_path=None):
    """'geo' (default) or 'xcorr' (the previous cross-correlation-only
    method). Precedence: env GMTSAR_NSR_ALIGN_METHOD, then `align_method`
    in the config file, then the default."""
    v = os.environ.get(ENV_METHOD)
    src = f"env {ENV_METHOD}"
    if v is None:
        v = config_value(config_path, "align_method")
        src = f"config key align_method in {config_path}"
    if v is None:
        return DEFAULT_METHOD
    v = v.strip().lower()
    if v not in VALID_METHODS:
        sys.exit(f"align_method must be one of {VALID_METHODS}, got {v!r} (from {src})")
    return v


# ----------------------------------------------------------------- PRM ---
def read_prm_value(path, key):
    """First whitespace token after `key =` in a PRM, or None.

    LAST occurrence wins, like the pipeline's own grep_value / get_prm:
    SAT_baseline_py APPENDS its results (rshift, ashift, ...) to the end
    of the repeat PRM, below the raw PRM's own `rshift = 0` line."""
    found = None
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                toks = v.split()
                if toks:
                    found = toks[0]
    return found


def prm_dims(path):
    """(num_rng_bins, num_patches * num_valid_az)."""
    nr = read_prm_value(path, "num_rng_bins")
    npch = read_prm_value(path, "num_patches")
    nva = read_prm_value(path, "num_valid_az")
    if nr is None or npch is None or nva is None:
        raise ValueError(f"{path}: missing num_rng_bins/num_patches/num_valid_az")
    return int(float(nr)), int(float(npch)) * int(float(nva))


# --------------------------------------------------------- polynomials ---
def poly_exponents(order):
    """All (i, j) with i + j <= order; term = u**i * v**j."""
    return [(i, d - i) for d in range(order + 1) for i in range(d + 1)]


def _design(u, v, order):
    return np.column_stack([u ** i * v ** j for i, j in poly_exponents(order)])


def _norm(x, y, dims):
    return 2.0 * np.asarray(x, float) / dims[0] - 1.0, 2.0 * np.asarray(y, float) / dims[1] - 1.0


def fit_poly(x, y, z, dims, order):
    u, v = _norm(x, y, dims)
    coef, *_ = np.linalg.lstsq(_design(u, v, order), np.asarray(z, float), rcond=None)
    return coef


def eval_poly(coef, x, y, dims, order):
    u, v = _norm(x, y, dims)
    return _design(u, v, order) @ coef


def fit_poly_robust(x, y, z, dims, order, n_iter=4, k=3.0):
    """Least squares with iterative k-sigma (MAD) clipping. Returns
    (coef, kept_mask)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    keep = np.ones(z.shape, bool)
    n_terms = len(poly_exponents(order))
    coef = None
    for _ in range(n_iter):
        if keep.sum() < n_terms:
            break
        coef = fit_poly(x[keep], y[keep], z[keep], dims, order)
        res = z - eval_poly(coef, x, y, dims, order)
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826
        if mad == 0:
            break
        new_keep = np.abs(res) <= k * mad
        if new_keep.sum() < n_terms or (new_keep == keep).all():
            break
        keep = new_keep
    if coef is None:
        coef = fit_poly(x, y, z, dims, order)
    return coef, keep


# ----------------------------------------------------------- statistics ---
def robust_stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    med = float(np.median(v))
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": med,
        "std": float(v.std()),
        "rsigma": float(np.median(np.abs(v - med)) * 1.4826),
        "maxabs": float(np.abs(v).max()),
    }


def block_medians(x, y, z, dims, nb=4):
    """nb x nb grid of block medians (row 0 = lowest azimuth)."""
    out = np.full((nb, nb), np.nan)
    ix = np.clip((np.asarray(x) / dims[0] * nb).astype(int), 0, nb - 1)
    iy = np.clip((np.asarray(y) / dims[1] * nb).astype(int), 0, nb - 1)
    z = np.asarray(z, float)
    for j in range(nb):
        for i in range(nb):
            m = (ix == i) & (iy == j)
            if m.any():
                out[j, i] = np.median(z[m])
    return out


def block_counts(x, y, dims, nb=8):
    ix = np.clip((np.asarray(x) / dims[0] * nb).astype(int), 0, nb - 1)
    iy = np.clip((np.asarray(y) / dims[1] * nb).astype(int), 0, nb - 1)
    cnt = np.zeros((nb, nb), int)
    np.add.at(cnt, (iy, ix), 1)
    return cnt


# ------------------------------------------------------------ xcorr I/O ---
def load_xcorr(path):
    """freq_xcorr.dat rows: x  range_offset  y  azimuth_offset  SNR."""
    d = np.loadtxt(path, ndmin=2)
    if d.shape[1] < 5:
        raise ValueError(f"{path}: expected 5 columns, got {d.shape[1]}")
    return d[:, :5]


# ------------------------------------------------------------- geometry ---
def decimate_dem(data, x, y, target_points, nan_height=0.0):
    """Stride-subsample a (ny, nx) DEM to roughly target_points cells.
    Returns ((N,3) array of lon, lat, h), stride.

    NaN cells (typically ocean / no-data) are KEPT with height nan_height
    (default 0 m) so geometry is defined over the whole DEM extent rather
    than silently missing exactly where cross-correlation also fails
    (open water). Offsets are insensitive to height: about 0.045 px per km
    for B_perp ~ 90 m, and the cross-correlation correction absorbs the
    smooth part of any error."""
    ny, nx = data.shape
    stride = max(1, int(np.ceil(np.sqrt(ny * nx / float(target_points)))))
    sub = np.array(np.asarray(data)[::stride, ::stride], dtype=float)
    sub[~np.isfinite(sub)] = nan_height
    X, Y = np.meshgrid(np.asarray(x)[::stride], np.asarray(y)[::stride])
    return np.column_stack([X.ravel(), Y.ravel(), sub.ravel()]), stride


def geometric_offsets(m_rat, a_rat, m_dims, a_dims):
    """From SAT_llt2rat rows (range_pix, azi_pix, ...) for the same ground
    points in master and repeat: keep points inside BOTH images (as
    align_tops does) and return (r_m, a_m, dr, da) with
    dr = r_repeat - r_master, da = a_repeat - a_master."""
    r_m, a_m = m_rat[:, 0], m_rat[:, 1]
    r_a, a_a = a_rat[:, 0], a_rat[:, 1]
    ok = (np.isfinite(r_m) & np.isfinite(a_m) & np.isfinite(r_a) & np.isfinite(a_a)
          & (r_m > 0) & (r_m < m_dims[0]) & (a_m > 0) & (a_m < m_dims[1])
          & (r_a > 0) & (r_a < a_dims[0]) & (a_a > 0) & (a_a < a_dims[1]))
    return r_m[ok], a_m[ok], (r_a - r_m)[ok], (a_a - a_m)[ok]


def run_llt2rat(prm_path, pts, precise):
    """Pipe lon/lat/h rows through SAT_llt2rat_py (ASCII in/out, no bounds
    filtering, so master and repeat outputs stay row-aligned)."""
    buf = io.StringIO()
    np.savetxt(buf, pts, fmt="%.9f %.9f %.3f")
    res = subprocess.run(
        ["SAT_llt2rat_py", os.path.abspath(prm_path), str(precise)],
        input=buf.getvalue(), capture_output=True, text=True, check=True,
    )
    return np.loadtxt(io.StringIO(res.stdout), ndmin=2)


def read_dem(path):
    from gmt_grd_io import read_gmt_grd
    data, x, y, _info = read_gmt_grd(path)
    return data, x, y


def geometric_model(master_prm, aligned_raw_prm, dem_path, dem_points=250000,
                    precise=1, order=3, min_points=50, max_empty_blocks=4):
    """Geometric offset model for one master/repeat pair.

    aligned_raw_prm must be the repeat's PRM BEFORE SAT_baseline_py appended
    its rshift/ashift (SAT_llt2rat_py subtracts those from its output).
    Returns a dict: coef_r, coef_a, dims, n_points, fit_rms, span,
    empty_blocks. Exits with a clear message if the DEM does not cover the
    scene."""
    m_dims = prm_dims(master_prm)
    a_dims = prm_dims(aligned_raw_prm)
    data, x, y = read_dem(dem_path)
    pts, stride = decimate_dem(data, x, y, dem_points)
    m_rat = run_llt2rat(master_prm, pts, precise)
    a_rat = run_llt2rat(aligned_raw_prm, pts, precise)
    if m_rat.shape[0] != a_rat.shape[0] or m_rat.shape[0] != len(pts):
        sys.exit("geom_align_lib: SAT_llt2rat_py row counts differ between master "
                 f"({m_rat.shape[0]}), repeat ({a_rat.shape[0]}) and input ({len(pts)}); "
                 "cannot pair points.")
    r_m, a_m, dr, da = geometric_offsets(m_rat, a_rat, m_dims, a_dims)
    if r_m.size < min_points:
        sys.exit(f"geom_align_lib: only {r_m.size} DEM points fall inside both images "
                 f"-- does {dem_path} cover the scene? (set align_method = xcorr to "
                 f"use cross-correlation only)")
    cnt = block_counts(r_m, a_m, m_dims, nb=8)
    n_empty = int((cnt < 3).sum())
    if n_empty > max_empty_blocks:
        sys.exit(f"geom_align_lib: DEM coverage of the scene is incomplete: {n_empty} of "
                 f"64 blocks have fewer than 3 DEM points (limit {max_empty_blocks}). "
                 f"The geometric model would extrapolate over them. Use a DEM that "
                 f"covers the whole scene, or set align_method = xcorr.")
    coef_r = fit_poly(r_m, a_m, dr, m_dims, order)
    coef_a = fit_poly(r_m, a_m, da, m_dims, order)
    rms = (float(np.sqrt(np.mean((dr - eval_poly(coef_r, r_m, a_m, m_dims, order)) ** 2))),
           float(np.sqrt(np.mean((da - eval_poly(coef_a, r_m, a_m, m_dims, order)) ** 2))))
    return {
        "coef_r": coef_r, "coef_a": coef_a, "dims": m_dims, "order": order,
        "n_points": int(r_m.size), "fit_rms": rms, "stride": stride,
        "span": (float(dr.min()), float(dr.max()), float(da.min()), float(da.max())),
        "empty_blocks": n_empty,
        # raw points, kept for the diagnostic
        "pts": (r_m, a_m, dr, da),
    }


def fit_alignment_grids(master, aligned, config_path=None, dem_path="../topo/dem.grd",
                        snr=20, run_cmd=None):
    """Shared entry point for align_batch_nsr and p2p_stages (cwd = SLC/,
    containing {master}.PRM, {aligned}.PRM0 (raw), freq_xcorr.dat, amp*.grd).
    Produces r.grd / a.grd by the configured method; returns the method used.

    geo (default): requires dem_path; a missing DEM is FATAL (no silent
    fallback to cross-correlation only -- the user asked for a clear error).
    xcorr: the previous `fitoffset_ra 10 10 freq_xcorr.dat <snr>`."""
    if run_cmd is None:
        from gmtsar_lib import run as run_cmd  # noqa: WPS433
    method = align_method(config_path)
    if method == "xcorr":
        print("ALIGN METHOD: xcorr (cross-correlation only, 10-term polynomial)")
        run_cmd(f"fitoffset_ra 10 10 freq_xcorr.dat {snr}")
        return method
    if not os.path.isfile(dem_path):
        sys.exit(f"{os.path.basename(sys.argv[0])}: FATAL: NISAR alignment now uses the DEM "
                 f"(geometric model refined by cross-correlation), but {dem_path} was not "
                 f"found (cwd {os.getcwd()}). Create topo/dem.grd first (make_dem / "
                 f"your own DEM covering the scene), or set `align_method = xcorr` in the "
                 f"config (or env {ENV_METHOD}=xcorr) to use cross-correlation only.")
    print(f"ALIGN METHOD: geo (DEM geometry + xcorr correction), DEM {dem_path}")
    geo_fitoffset(f"{master}.PRM", f"{aligned}.PRM0", dem_path, "freq_xcorr.dat",
                  snr=float(snr), run_cmd=run_cmd)
    return method


# ----------------------------------------------- correction + grid build ---
def fit_correction(x, y, res_r, res_a, dims, order=1, outlier_px=0.5,
                   max_outlier_frac=0.5, max_correction_px=2.0):
    """Robust low-order fit of measured - geometric residuals.

    Points with |res| > outlier_px in either axis are excluded first; if
    more than max_outlier_frac of the points are excluded, or the fitted
    correction exceeds max_correction_px anywhere in the scene, geometry and
    correlation disagree too much to trust either silently -> sys.exit.
    Returns dict(coef_r, coef_a, order, n_used, n_outliers, max_corr)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    res_r, res_a = np.asarray(res_r, float), np.asarray(res_a, float)
    bad = (np.abs(res_r) > outlier_px) | (np.abs(res_a) > outlier_px)
    n_out = int(bad.sum())
    if n_out > max_outlier_frac * x.size:
        sys.exit(f"geom_align_lib: {n_out} of {x.size} cross-correlation points disagree "
                 f"with the geometric model by more than {outlier_px} px -- geometry and "
                 f"correlation are inconsistent (wrong DEM/orbit, or bad correlation). "
                 f"Inspect with diag_geom_vs_xcorr, or set align_method = xcorr.")
    good = ~bad
    n_terms = len(poly_exponents(order))
    if good.sum() < max(8, n_terms):
        sys.exit(f"geom_align_lib: only {int(good.sum())} usable points after outlier "
                 f"rejection (need >= {max(8, n_terms)}). Try a lower SNR cutoff.")
    coef_r, _ = fit_poly_robust(x[good], y[good], res_r[good], dims, order)
    coef_a, _ = fit_poly_robust(x[good], y[good], res_a[good], dims, order)
    cx = np.array([0.0, dims[0], 0.0, dims[0], dims[0] / 2.0])
    cy = np.array([0.0, 0.0, dims[1], dims[1], dims[1] / 2.0])
    max_corr = float(max(np.abs(eval_poly(coef_r, cx, cy, dims, order)).max(),
                         np.abs(eval_poly(coef_a, cx, cy, dims, order)).max()))
    if max_corr > max_correction_px:
        sys.exit(f"geom_align_lib: the cross-correlation correction reaches {max_corr:.2f} px "
                 f"(limit {max_correction_px}) -- far larger than expected for a working "
                 f"geometric model. Inspect with diag_geom_vs_xcorr, or set "
                 f"align_method = xcorr.")
    return {"coef_r": coef_r, "coef_a": coef_a, "order": order,
            "n_used": int(good.sum()), "n_outliers": n_out, "max_corr": max_corr}


def lattice_model(geo, corr, region, spacing=256.0):
    """Evaluate geometry + correction on a regular lattice covering `region`
    = (w, e, s, n). Returns (xyz_r, xyz_a): (N,3) arrays of x, y, value."""
    w, e, s, n = region
    xs = np.linspace(w, e, max(2, int(np.ceil((e - w) / spacing)) + 1))
    ys = np.linspace(s, n, max(2, int(np.ceil((n - s) / spacing)) + 1))
    X, Y = np.meshgrid(xs, ys)
    x, y = X.ravel(), Y.ravel()
    dims, go, co = geo["dims"], geo["order"], corr["order"]
    zr = eval_poly(geo["coef_r"], x, y, dims, go) + eval_poly(corr["coef_r"], x, y, dims, co)
    za = eval_poly(geo["coef_a"], x, y, dims, go) + eval_poly(corr["coef_a"], x, y, dims, co)
    return np.column_stack([x, y, zr]), np.column_stack([x, y, za])


def parse_region(text):
    """'-R0/53316/0/54720' (as printed by `gmt grdinfo ... -I-`) -> floats."""
    m = re.search(r"-R\s*([-+0-9.eE]+)/([-+0-9.eE]+)/([-+0-9.eE]+)/([-+0-9.eE]+)", text)
    if not m:
        raise ValueError(f"cannot parse a -R region from {text!r}")
    return tuple(float(g) for g in m.groups())


def grid_region(cmd="gmt grdinfo amp*.grd -I-"):
    """Region of the master amplitude grid, same source fitoffset_ra uses."""
    out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True).stdout
    return parse_region(out)


def geo_fitoffset(master_prm, aligned_raw_prm, dem_path, xcorr_dat, snr=20.0,
                  corr_order=1, run_cmd=None, region=None, spacing=256.0,
                  **geo_kwargs):
    """Replacement for `fitoffset_ra`: writes r.grd / a.grd in cwd.

    run_cmd: callable taking a shell command string (default: subprocess
    via gmtsar_lib.run). region: (w, e, s, n) of the amplitude grid
    (default: asked from `gmt grdinfo amp*.grd -I-`, which requires the
    slc2amp output the pipeline already creates)."""
    if run_cmd is None:
        from gmtsar_lib import run as run_cmd  # noqa: WPS433
    xc = load_xcorr(xcorr_dat)
    sel = xc[xc[:, 4] > snr]
    if sel.shape[0] < 8:
        sys.exit(f" FAILED - not enough points ({sel.shape[0]}/{xc.shape[0]}). Try lower SNR.")

    geo = geometric_model(master_prm, aligned_raw_prm, dem_path, **geo_kwargs)
    dims = geo["dims"]
    x, dr_m, y, da_m = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
    res_r = dr_m - eval_poly(geo["coef_r"], x, y, dims, geo["order"])
    res_a = da_m - eval_poly(geo["coef_a"], x, y, dims, geo["order"])
    corr = fit_correction(x, y, res_r, res_a, dims, order=corr_order)

    print(f"GEO ALIGN: {geo['n_points']} DEM points (stride {geo['stride']}), geometric "
          f"cubic fit rms range {geo['fit_rms'][0]:.4f} / azimuth {geo['fit_rms'][1]:.4f} px; "
          f"offsets span range [{geo['span'][0]:+.2f}, {geo['span'][1]:+.2f}], "
          f"azimuth [{geo['span'][2]:+.2f}, {geo['span'][3]:+.2f}] px")
    print(f"GEO ALIGN: {corr['n_used']} of {sel.shape[0]} xcorr points (SNR > {snr}) used "
          f"for the order-{corr_order} correction; {corr['n_outliers']} rejected as "
          f"outliers (> 0.5 px from geometry); max |correction| in scene "
          f"{corr['max_corr']:.3f} px")
    print(f"GEO ALIGN: correction coefficients (normalized coords, terms "
          f"{poly_exponents(corr_order)}): range {np.round(corr['coef_r'], 4).tolist()}, "
          f"azimuth {np.round(corr['coef_a'], 4).tolist()}")

    if region is None:
        region = grid_region()
    xyz_r, xyz_a = lattice_model(geo, corr, region, spacing=spacing)
    np.savetxt("r_model.xyz", xyz_r, fmt="%f %f %f")
    np.savetxt("a_model.xyz", xyz_a, fmt="%f %f %f")

    # Same tail as fitoffset_ra: gmt surface on the amp grid region, then FLIPUD.
    run_cmd("gmt surface r_model.xyz `gmt grdinfo amp*.grd -I-` -rp -I64/64 -T.3 -Gr_tmp.grd")
    run_cmd("gmt surface a_model.xyz `gmt grdinfo amp*.grd -I-` -rp -I64/64 -T.3 -Ga_tmp.grd")
    try:
        import gmt_grdmath_py as _gm  # type: ignore
        use_py = os.environ.get("GMTSAR_GRDMATH_PY", "1") == "1"
    except ImportError:
        _gm, use_py = None, False
    if use_py:
        _gm.grdmath1("FLIPUD", "r_tmp.grd", "r.grd", ctx="fitoffset_ra_geo:FLIPUD_r")
        _gm.grdmath1("FLIPUD", "a_tmp.grd", "a.grd", ctx="fitoffset_ra_geo:FLIPUD_a")
    else:
        run_cmd("gmt grdmath r_tmp.grd FLIPUD = r.grd")
        run_cmd("gmt grdmath a_tmp.grd FLIPUD = a.grd")
    run_cmd("rm -f r_model.xyz a_model.xyz r_tmp.grd a_tmp.grd")
    return geo, corr
