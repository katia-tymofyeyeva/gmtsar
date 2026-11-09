"""iono_est_lib -- per-pair raw dispersive (ionospheric) phase estimate from the
two NISAR bands (Stage 2, step 2).  Pure numpy.

Model (phase in radians, r = fB/fA > 1, A = main band, B = 5 MHz band):
    phi_A = I + N               I: ionosphere seen by A,  N: non-dispersive
    phi_B = I/r + r*N
Eliminating N:
    phi_B - r*phi_A = -I * (r^2 - 1) / r
so   I = -psi * r / (r^2 - 1),   psi = wrap(phi_B - r*phi_A_unwrapped).
Only phi_A has to be unwrapped (coarse scale); psi is only ~9 % of I and never
wraps for |I| < ~36 rad.  A 2*pi unwrapping slip k in phi_A shifts I by about
k * 2*pi * r/(r+1) ~= 3.2 rad (about pi): loud, and detectable as a residual in
the network inversion.

No spatial filtering happens here: these are raw estimates with a variance and
a validity mask, for the per-date network inversion (Stage 3).  The input
phases must be plain multilooks (phase.grd / realfilt+imagfilt), never
Goldstein `phasefilt.grd`.
"""
import numpy as np


def box_columns(a, ratio, ncols, strict=False, chunk=512):
    """Exact box average along columns onto a grid whose cells are `ratio`
    input cells wide (ratio may be fractional): output cell j averages the input
    over [j*ratio, (j+1)*ratio) in cell units, with fractional end cells
    weighted by overlap.  Both grids start at input column 0 (shared pixel 0).

    strict=False: NaN counts as 0.  strict=True: an output cell is NaN if any
    input cell overlapping it (with nonzero weight) is NaN (for unwrapped phase:
    never average across a masked gap).  Raises ValueError if ncols*ratio
    exceeds the number of input columns."""
    a = np.asarray(a, dtype=np.float32)
    ny, nx = a.shape
    if ncols * ratio > nx + 1e-6:
        raise ValueError(f"need {ncols * ratio:.3f} input columns for {ncols} cells, have {nx}")
    edges = np.arange(ncols + 1, dtype=np.float64) * ratio
    i0 = np.minimum(np.floor(edges).astype(np.int64), nx - 1)
    fr = edges - i0
    out = np.empty((ny, ncols), dtype=np.float32)
    for s in range(0, ny, chunk):
        blk = a[s:s + chunk]
        nan = ~np.isfinite(blk)
        c = np.zeros((blk.shape[0], nx + 1), dtype=np.float64)
        np.cumsum(np.where(nan, 0.0, blk), axis=1, out=c[:, 1:])
        F = c[:, i0] + fr[None, :] * (c[:, i0 + 1] - c[:, i0])
        val = (F[:, 1:] - F[:, :-1]) / ratio
        if strict and nan.any():
            n = np.zeros((blk.shape[0], nx + 1), dtype=np.float64)
            np.cumsum(nan, axis=1, out=n[:, 1:])
            G = n[:, i0] + fr[None, :] * (n[:, i0 + 1] - n[:, i0])
            val = np.where((G[:, 1:] - G[:, :-1]) > 1e-6, np.nan, val)
        out[s:s + chunk] = val
    return out


def coarsen_range(a, ratio, ncols):
    """Box average of `ratio` (may be fractional) columns, NaN counted as 0."""
    return box_columns(a, float(ratio), ncols, strict=False)


def coarsen_range_strict(a, ratio, ncols):
    """Box average of `ratio` columns; NaN if any contributing cell is NaN."""
    return box_columns(a, float(ratio), ncols, strict=True)


def unwrap_from_reference(phase, unw_ref_fine, ratio):
    """phi_unw = Phi + wrap(phase - Phi), where Phi is an EXISTING unwrapped
    phase on a grid `ratio` (fractional ok) times finer in range than `phase`
    (A's own unwrap.grd on its 12.5 m grid, box-averaged onto the B grid).  Cells
    where Phi is NaN (masked in the original unwrap) stay NaN.  Row order and
    grid origin must match.  Returns (phi_unw, inconsistent): `inconsistent` is
    the fraction of valid cells where |wrap(phase - Phi)| > 2 rad."""
    phi = coarsen_range_strict(unw_ref_fine, ratio, phase.shape[1])
    if phi.shape[0] != phase.shape[0]:
        raise ValueError(f"row count differs: reference {phi.shape[0]}, phase {phase.shape[0]}")
    d = wrap(phase - phi)
    ok = np.isfinite(d)
    bad = float(np.mean(np.abs(d[ok]) > 2.0)) if ok.any() else float("nan")
    return (phi + d).astype(np.float32), bad


def complex_on_coarse_grid(real, imag, amp1, amp2, ratio, ncols):
    """Phase and coherence of an already-multilooked complex interferogram
    box-averaged by `ratio` input cells per output cell (fractional ok) in range,
    from `filter`'s own outputs.

    Same formula `filter` uses for coherence, amp / sqrt(amp1*amp2), with
    amp = hypot(real, imag); real/imag/amp1/amp2 are averaged over the cells
    being merged.  Returns (phase, coh), phase in (-pi, pi]; cells with no
    signal are NaN.  Row order is whatever the inputs have (the caller applies
    the same FLIPUD `filter` applies to phase.grd/corr.grd)."""
    re = coarsen_range(real, ratio, ncols)
    im = coarsen_range(imag, ratio, ncols)
    p1 = coarsen_range(amp1, ratio, ncols)
    p2 = coarsen_range(amp2, ratio, ncols)
    amp = np.hypot(re, im)
    den = np.sqrt(p1 * p2)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.where(den > 0, amp / den, np.nan)
        phase = np.where(amp > 0, np.arctan2(im, re), np.nan)
    coh = np.where(np.isfinite(coh), np.clip(coh, 0.0, 1.0), np.nan)
    return phase.astype(np.float32), coh.astype(np.float32)


def block_mean_phasor(phase, coh, b):
    """Coarsen wrapped `phase` by b x b blocks with a coherence-weighted complex
    mean: z = coh * exp(i*phase).  Returns (phase_c, coh_c, nrow_c, ncol_c);
    coh_c = |sum z| / sum coh (1 = all cells agree).  Blocks without signal are
    NaN.  Edge rows/columns that do not fill a block are dropped."""
    ny, nx = phase.shape
    nr, nc = ny // b, nx // b
    if nr < 1 or nc < 1:
        raise ValueError(f"block {b} larger than grid {phase.shape}")
    ph = np.nan_to_num(phase[:nr * b, :nc * b].astype(np.float32))
    cg = np.nan_to_num(coh[:nr * b, :nc * b].astype(np.float32))
    zr = (cg * np.cos(ph)).reshape(nr, b, nc, b).sum(axis=(1, 3), dtype=np.float64)
    zi = (cg * np.sin(ph)).reshape(nr, b, nc, b).sum(axis=(1, 3), dtype=np.float64)
    w = cg.reshape(nr, b, nc, b).sum(axis=(1, 3), dtype=np.float64)
    amp = np.hypot(zr, zi)
    with np.errstate(divide="ignore", invalid="ignore"):
        pc = np.where(amp > 0, np.arctan2(zi, zr), np.nan)
        cc = np.where(w > 0, amp / w, np.nan)
    return pc.astype(np.float32), cc.astype(np.float32), nr, nc


def fill_nearest(a, max_dist):
    """Fill NaNs by growing the finite region inward `max_dist` cells: each pass
    gives a NaN cell the mean of its finite 4-neighbours.  Cells farther than
    `max_dist` from any finite value stay NaN.  (numpy only; the unwrapped
    surface is smooth, so a neighbour mean is as good as the nearest value.)"""
    out = np.array(a, dtype=np.float32, copy=True)
    for _ in range(int(np.ceil(max_dist))):
        bad = ~np.isfinite(out)
        if not bad.any() or bad.all():
            break
        pad = np.pad(out, 1, constant_values=np.nan)
        nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        fin = np.isfinite(nb)
        cnt = fin.sum(axis=0)
        s = np.where(fin, nb, 0.0).sum(axis=0)
        fill = bad & (cnt > 0)
        out[fill] = s[fill] / cnt[fill]
    return out.astype(a.dtype if np.issubdtype(np.asarray(a).dtype, np.floating) else np.float32)


def upsample_linear(zc, b, shape):
    """Linear interpolation of a block-mean grid `zc` back to `shape`: coarse
    cell j is centred at fine index j*b + (b-1)/2; positions outside the
    centres clamp.  NaNs in `zc` propagate to nearby fine cells."""
    nr, nc = zc.shape
    ny, nx = shape

    def idx_w(n, ncoarse):
        t = (np.arange(n) - (b - 1) / 2.0) / b
        t = np.clip(t, 0, ncoarse - 1)
        i0 = np.minimum(np.floor(t).astype(int), max(ncoarse - 2, 0))
        i1 = np.minimum(i0 + 1, ncoarse - 1)
        return i0, i1, (t - i0).astype(np.float32)

    j0, j1, wx = idx_w(nx, nc)
    i0, i1, wy = idx_w(ny, nr)
    zc = zc.astype(np.float32)
    cols = (1 - wx)[None, :] * zc[:, j0] + wx[None, :] * zc[:, j1]      # (nr, nx)
    out = np.empty((ny, nx), dtype=np.float32)
    step = 512
    for s in range(0, ny, step):
        e = min(s + step, ny)
        out[s:e] = (1 - wy[s:e, None]) * cols[i0[s:e]] + wy[s:e, None] * cols[i1[s:e]]
    return out


def unwrap_via_coarse(phase, unw_coarse, b, fill_cells=2):
    """phi_unw = Phi + wrap(phase - Phi), Phi = `unw_coarse` interpolated to the
    full grid.  Restores cell-level phase without unwrapping it (valid while
    |phase - Phi| < pi).  Coarse NaNs within `fill_cells` coarse cells are filled
    from the nearest unwrapped cell; beyond that the fine cells are NaN."""
    phi = upsample_linear(fill_nearest(unw_coarse, fill_cells), b, phase.shape)
    return phi + wrap(phase - phi)


def band_ratio(f_a, f_b):
    """(r, a, b) with I = a*phi_A + b*phi_B (a = r^2/(r^2-1), b = -r/(r^2-1))."""
    r = f_b / f_a
    if r <= 1.0:
        raise ValueError(f"expected fB > fA, got r = {r}")
    return r, r * r / (r * r - 1.0), -r / (r * r - 1.0)


def wrap(x):
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def n_eff_gaussian(wavelength_m, az_pixel_m, rng_pixel_m):
    """Approximate number of independent single looks in one Gaussian-filtered
    cell.  `filter`'s Gaussian has sigma = wavelength/5.3 (ground metres); a
    Gaussian's noise-equivalent area is 4*pi*sigma^2, so
    N ~= 4*pi*sigma^2 / (az_pixel * rng_pixel).  Ignores the small 15x3 prefilter
    and any pixel oversampling, so it is an estimate (tune against data)."""
    sigma = wavelength_m / 5.3
    return max(1.0, 4.0 * np.pi * sigma * sigma / (az_pixel_m * rng_pixel_m))


def phase_sigma(coh, n_eff, coh_floor=0.05, coh_ceil=0.995):
    """Cramer-Rao phase standard deviation (rad) for coherence `coh` and
    `n_eff` independent looks: sqrt((1-g^2) / (2 N g^2))."""
    g = np.clip(np.asarray(coh, dtype=float), coh_floor, coh_ceil)
    return np.sqrt((1.0 - g * g) / (2.0 * n_eff * g * g))


def dispersive_estimate(phi_a_unw, phi_b, coh_a, coh_b, r, n_eff_a, n_eff_b=None,
                        coh_min=0.2):
    """Raw dispersive phase I (rad, seen by band A) with its standard deviation.

    phi_a_unw  unwrapped A-band phase (rad), same grid as the rest
    phi_b      wrapped B-band phase (rad)
    coh_a/b    coherence of each band, in [0, 1]
    r          fB/fA
    n_eff_a/b  effective looks per cell (n_eff_b defaults to n_eff_a)
    coh_min    cells where either coherence is below this are invalid (NaN)

    Returns dict(I, sigma, valid, psi).  I and sigma are NaN where invalid."""
    n_eff_b = n_eff_a if n_eff_b is None else n_eff_b
    k = r / (r * r - 1.0)
    psi = wrap(phi_b - r * phi_a_unw)
    I = -psi * k
    sig_a = phase_sigma(coh_a, n_eff_a)
    sig_b = phase_sigma(coh_b, n_eff_b)
    sigma = k * np.sqrt(sig_b ** 2 + (r * sig_a) ** 2)
    valid = (np.isfinite(phi_a_unw) & np.isfinite(phi_b)
             & (np.asarray(coh_a) >= coh_min) & (np.asarray(coh_b) >= coh_min))
    I = np.where(valid, I, np.nan)
    sigma = np.where(valid, sigma, np.nan)
    return {"I": I, "sigma": sigma, "valid": valid, "psi": np.where(valid, psi, np.nan)}


def slip_shift(r, k=1):
    """Shift in I (rad) caused by k 2*pi unwrapping slips in phi_A."""
    return k * 2.0 * np.pi * r / (r + 1.0)
