"""iono_net_lib -- from raw per-pair dispersive estimates to per-date ionospheric
screens (Stages 3-4).  Pure numpy.

  block_reduce      raw per-pair I (cell grid) -> robust inverse-variance means on
                    a coarser grid (sigma corrected for correlated cells)
  solve_network     weighted least-squares inversion of the pair network into
                    per-date screens (master = 0), per-pair constants (datum),
                    IRLS down-weighting of unwrapping slips, residual QC
  filter_screen     one linear filter per date: weighted quadratic trend ->
                    outlier rejection -> gap-aware normalized Gaussian ->
                    shrink to the trend where there is no data

Closure: the correction applied to pair (i, j) is F(S_i) - F(S_j) with F applied
to each date separately, so for any triplet the corrections sum to zero and the
corrected closure phase equals the raw closure exactly (tested).
"""
import numpy as np


# ----------------------------------------------------------- block reduce ---
def block_reduce(I, sigma, by, bx, n_indep_per_cell=1.0, k_out=4.0, min_valid=3):
    """Robust inverse-variance mean of I over by x bx blocks.

    I, sigma         (ny, nx) raw estimate and per-cell sigma (NaN invalid)
    n_indep_per_cell independent looks' worth of cells per cell: a block of
                     by*bx cells holds only about by*bx*n_indep_per_cell
                     independent cells when neighbouring cells overlap
                     (Gaussian multilook), so sigma_block = sigma_cell /
                     sqrt(n_indep) with n_indep = min(valid, by*bx*n_indep_per_cell)
    Outlier cells (|I - block median| > k_out * max(sigma, block MAD sigma)),
    e.g. unwrapping slips, are dropped.  Blocks with fewer than `min_valid`
    inliers are NaN.  Partial edge blocks are dropped.
    Returns (I_b, sigma_b, n_inliers)."""
    ny, nx = I.shape
    nr, nc = ny // by, nx // bx
    if nr < 1 or nc < 1:
        raise ValueError(f"block {by}x{bx} larger than grid {I.shape}")
    a = I[:nr * by, :nc * bx].reshape(nr, by, nc, bx).transpose(0, 2, 1, 3).reshape(nr, nc, -1)
    s = sigma[:nr * by, :nc * bx].reshape(nr, by, nc, bx).transpose(0, 2, 1, 3).reshape(nr, nc, -1)
    bad = ~(np.isfinite(a) & np.isfinite(s) & (s > 0))
    a = np.where(bad, np.nan, a)
    s = np.where(bad, np.nan, s)
    import warnings
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(a, axis=-1)
            res = a - med[..., None]
            mad = 1.4826 * np.nanmedian(np.abs(res), axis=-1)
        thr = k_out * np.maximum(s, mad[..., None])
        keep = np.isfinite(a) & (np.abs(res) <= thr)
        w = np.where(keep, 1.0 / s ** 2, 0.0)
        sw = w.sum(axis=-1)
        n = keep.sum(axis=-1)
        Ib = np.where(sw > 0, (np.where(keep, a, 0.0) * w).sum(axis=-1) / sw, np.nan)
        # typical per-cell sigma is sqrt(n / sw) (harmonic); the block mean has
        # that sigma over sqrt(number of INDEPENDENT cells)
        n_ind = np.maximum(1.0, np.minimum(n, by * bx * n_indep_per_cell))
        sb = np.sqrt(n / sw) / np.sqrt(n_ind)
        ok = n >= min_valid
    return (np.where(ok, Ib, np.nan).astype(np.float32),
            np.where(ok, sb, np.nan).astype(np.float32), n.astype(np.int32))


# ---------------------------------------------------------------- network ---
def _design(pairs, n_dates, master):
    cols = [d for d in range(n_dates) if d != master]
    pos = {d: k for k, d in enumerate(cols)}
    A = np.zeros((len(pairs), len(cols)))
    for p, (i, j) in enumerate(pairs):          # I_p = S_i - S_j + c_p
        if i != master:
            A[p, pos[i]] += 1.0
        if j != master:
            A[p, pos[j]] -= 1.0
    return A, cols


def check_connected(pairs, n_dates):
    parent = list(range(n_dates))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in pairs:
        parent[find(i)] = find(j)
    roots = {find(d) for d in range(n_dates)}
    return len(roots) == 1


def solve_network(pairs, I, sig, n_dates, master, n_iter=3, huber_k=2.5,
                  chunk=40000, ridge=1e-6):
    """Per-pixel weighted least squares  I_p = S_i - S_j + c_p,  S_master = 0.

    pairs   list of (i, j) date indices, one per pair (ref, rep)
    I, sig  (P, ny, nx) raw estimates and sigmas (NaN = invalid)
    c_p     one spatially constant datum per pair, estimated (weighted mean
            residual) because each pair's unwrapping fixes it only up to a
            constant; this removes the spatial mean of the screens (datum is
            unobservable anyway).  Huber IRLS down-weights isolated bad cells.
    Returns dict: S (D, ny, nx), sigma_S (D, ny, nx), c (P,), resid (P, ny, nx)
    standardized residuals, qc (per-pair dict).  Dates with no valid pair at a
    pixel get S = NaN there."""
    P, ny, nx = I.shape
    if P != len(pairs):
        raise ValueError("pairs and I disagree")
    if not check_connected(pairs, n_dates):
        raise ValueError("pair network is not connected: some dates cannot be tied to the master")
    A, cols = _design(pairs, n_dates, master)
    K = len(cols)
    Aout = (A[:, :, None] * A[:, None, :]).reshape(P, K * K)

    valid = np.isfinite(I) & np.isfinite(sig) & (sig > 0)
    w0 = np.where(valid, 1.0 / np.where(valid, sig, 1.0) ** 2, 0.0).reshape(P, -1)
    y = np.where(valid, I, 0.0).reshape(P, -1).astype(np.float64)
    npix = y.shape[1]

    # initial datum: weighted median per pair
    c = np.zeros(P)
    for p in range(P):
        v = valid[p].ravel()
        if v.any():
            c[p] = _wmedian(y[p, v], w0[p, v])
    wt = w0.copy()
    S = np.zeros((K, npix))
    covd = np.zeros((K, npix))
    for it in range(n_iter):
        for s in range(0, npix, chunk):
            e = min(s + chunk, npix)
            W = wt[:, s:e]                                   # (P, n)
            N = (W.T @ Aout).reshape(-1, K, K)
            tr = np.trace(N, axis1=1, axis2=2) / K
            N = N + (ridge * (tr + 1e-30))[:, None, None] * np.eye(K)[None]
            rhs = np.einsum("pk,pn,pn->nk", A, W, y[:, s:e] - c[:, None])
            S[:, s:e] = np.linalg.solve(N, rhs[:, :, None])[:, :, 0].T
            if it == n_iter - 1:
                covd[:, s:e] = np.diagonal(np.linalg.inv(N), axis1=1, axis2=2).T
        pred = A @ S                                          # (P, npix)
        r = y - c[:, None] - pred
        # update the pair datum (weighted mean residual), then Huber weights
        for p in range(P):
            sw = w0[p].sum()
            if sw > 0:
                c[p] += (w0[p] * r[p]).sum() / sw
        r = y - c[:, None] - pred
        z = r * np.sqrt(w0)
        hub = np.where(np.abs(z) > huber_k, huber_k / np.maximum(np.abs(z), 1e-12), 1.0)
        wt = w0 * hub

    # observability: a date needs at least one valid pair at the pixel
    cnt = np.zeros((K, npix))
    for k in range(K):
        use = np.nonzero(A[:, k] != 0)[0]
        cnt[k] = (w0[use] > 0).sum(axis=0)
    Sfull = np.full((n_dates, npix), np.nan, dtype=np.float32)
    sigfull = np.full((n_dates, npix), np.nan, dtype=np.float32)
    Sfull[master] = 0.0
    sigfull[master] = 0.0
    for k, d in enumerate(cols):
        okk = cnt[k] > 0
        Sfull[d, okk] = S[k, okk]
        sigfull[d, okk] = np.sqrt(np.maximum(covd[k, okk], 0.0))
    pred = A @ S
    r = y - c[:, None] - pred
    z = np.where(w0 > 0, r * np.sqrt(w0), np.nan)
    qc = []
    for p in range(P):
        zp = z[p][np.isfinite(z[p])]
        qc.append({"pair": pairs[p],
                   "n": int(zp.size),
                   "rms_z": float(np.sqrt(np.mean(zp ** 2))) if zp.size else float("nan"),
                   "frac_z_gt3": float(np.mean(np.abs(zp) > 3)) if zp.size else float("nan"),
                   "datum": float(c[p])})
    return {"S": Sfull.reshape(n_dates, ny, nx), "sigma_S": sigfull.reshape(n_dates, ny, nx),
            "c": c, "resid": z.reshape(P, ny, nx).astype(np.float32), "qc": qc}


def _wmedian(v, w):
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


# ----------------------------------------------------------------- filter ---
def _gauss_kernel(sig_cells, trunc=4.0):
    n = max(1, int(np.ceil(trunc * sig_cells)))
    x = np.arange(-n, n + 1, dtype=np.float64)
    g = np.exp(-0.5 * (x / max(sig_cells, 1e-6)) ** 2)
    return g / g.sum()


def _conv2_sep(a, ky, kx):
    """Zero-padded separable convolution (same size) via FFT."""
    ny, nx = a.shape
    py, px = len(ky) // 2, len(kx) // 2
    fy, fx = ny + 2 * py, nx + 2 * px
    Ny = 1 << int(np.ceil(np.log2(fy + len(ky))))
    Nx = 1 << int(np.ceil(np.log2(fx + len(kx))))
    A = np.fft.rfft2(a, s=(Ny, Nx))
    KY = np.fft.fft(ky, n=Ny)[:, None]
    KX = np.fft.rfft(kx, n=Nx)[None, :]
    out = np.fft.irfft2(A * KY * KX, s=(Ny, Nx))
    return out[py:py + ny, px:px + nx]


def _trend_design(ny, nx, order):
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    x = 2 * x / max(nx - 1, 1) - 1
    y = 2 * y / max(ny - 1, 1) - 1
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    return np.stack([x ** i * y ** j for i, j in terms], axis=-1)   # (ny, nx, T)


def filter_screen(S, sigma, sig_y_cells, sig_x_cells, order=2, k_out=5.0,
                  prior_frac=0.02, n_trend_iter=3):
    """One date's screen -> filtered screen + formal sigma.

    1. weighted polynomial trend of `order` (quadratic default) by LS with
       weights 1/sigma^2, iterating outlier rejection (> k_out * robust sigma);
    2. residual = S - trend on inliers;
    3. normalized Gaussian: num = G*(w r), den = G*w, residual_f = num/(den+eps),
       eps = prior_frac * median(w): data-free regions fall back to the trend
       and gaps are filled from neighbours without smearing data across them;
       zero padding is correct at the edges (num and den padded alike);
    4. S_f = trend + residual_f everywhere (trend extrapolated over gaps).
    Returns (S_f, sigma_f, mask_used)  where sigma_f is the formal sigma of the
    normalized convolution (ignores trend-fit uncertainty)."""
    ny, nx = S.shape
    ok = np.isfinite(S) & np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < 20:
        raise ValueError("too few valid cells to filter")
    w = np.where(ok, 1.0 / np.where(ok, sigma, 1.0) ** 2, 0.0)
    T = _trend_design(ny, nx, order)
    Tf = T.reshape(-1, T.shape[-1])
    inl = ok.copy()
    for _ in range(n_trend_iter):
        wv = np.where(inl, w, 0.0).ravel()
        Aw = Tf * wv[:, None]
        coef = np.linalg.lstsq(Tf.T @ Aw, Aw.T @ np.where(ok, S, 0.0).ravel(), rcond=None)[0]
        trend = (Tf @ coef).reshape(ny, nx)
        res = np.where(ok, S - trend, 0.0)
        rs = res[inl]
        sig_rob = 1.4826 * np.median(np.abs(rs - np.median(rs)))
        inl = ok & (np.abs(res) <= k_out * np.maximum(sigma, sig_rob))
    wv = np.where(inl, w, 0.0)
    ky, kx = _gauss_kernel(sig_y_cells), _gauss_kernel(sig_x_cells)
    num = _conv2_sep(wv * np.where(inl, res, 0.0), ky, kx)
    den = _conv2_sep(wv, ky, kx)
    eps = prior_frac * np.median(w[inl])
    resf = num / (den + eps)
    den2 = _conv2_sep(wv, ky ** 2, kx ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_f = np.sqrt(den2) / (den + eps)
    return (trend + resf).astype(np.float32), sig_f.astype(np.float32), inl


# ------------------------------------------------------------ interpolate ---
def interp_to_grid(zc, xc, yc, xt, yt, chunk=512):
    """Separable bilinear interpolation of a coarse grid onto target
    coordinates.  zc is (len(yc), len(xc)); xc, yc ascending; xt, yt are the
    target column / row coordinates (any order).  Positions outside the coarse
    centres take the edge value; NaNs propagate to the cells that use them.
    Coordinates (not block indices) are used so grids that are not an exact
    multiple of each other (A cells vs B cells, ratio 7.998) line up."""
    zc = np.asarray(zc, dtype=np.float32)
    xc = np.asarray(xc, dtype=np.float64)
    yc = np.asarray(yc, dtype=np.float64)

    def idx_w(c, t):
        t = np.clip(np.asarray(t, dtype=np.float64), c[0], c[-1])
        i1 = np.clip(np.searchsorted(c, t, side="right"), 1, len(c) - 1)
        i0 = i1 - 1
        w = (t - c[i0]) / (c[i1] - c[i0])
        return i0, i1, w.astype(np.float32)

    j0, j1, wx = idx_w(xc, xt)
    i0, i1, wy = idx_w(yc, yt)
    cols = (1 - wx)[None, :] * zc[:, j0] + wx[None, :] * zc[:, j1]
    out = np.empty((len(yt), len(xt)), dtype=np.float32)
    for s in range(0, len(yt), chunk):
        e = min(s + chunk, len(yt))
        out[s:e] = (1 - wy[s:e, None]) * cols[i0[s:e]] + wy[s:e, None] * cols[i1[s:e]]
    return out
