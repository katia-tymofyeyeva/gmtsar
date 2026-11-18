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
  reject_outliers   the trend + outlier-rejection part of filter_screen on its
                    own, memory-bounded for full-resolution grids (used by the
                    SBAS-like full-resolution inversion, which does not filter)

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


def _normal_eq(ki, kj, W, Y, K):
    """Normal equations of I_p = S_i - S_j (gauge column removed) for a batch
    of pixels, assembled pair by pair from the two nonzeros of each design
    row (4 updates per pair instead of a dense P x K x K product).
    ki, kj: column of each pair's dates (-1 = master); W, Y: (P, n).
    Returns N (n, K, K), rhs (n, K)."""
    n = W.shape[1]
    N = np.zeros((n, K, K))
    rhs = np.zeros((n, K))
    for p in range(W.shape[0]):
        w = W[p]
        wy = w * Y[p]
        a, b = ki[p], kj[p]
        if a >= 0:
            N[:, a, a] += w
            rhs[:, a] += wy
        if b >= 0:
            N[:, b, b] += w
            rhs[:, b] -= wy
        if a >= 0 and b >= 0:
            N[:, a, b] -= w
            N[:, b, a] -= w
    return N, rhs


def _regularize(N, ridge):
    K = N.shape[-1]
    tr = np.trace(N, axis1=1, axis2=2) / K
    N += (ridge * (tr + 1e-30))[:, None, None] * np.eye(K)[None]
    return N


def _predict(S, ki, kj):
    """(P, n) predictions S_i - S_j from S (K, n), master = 0."""
    Sx = np.vstack([S, np.zeros((1, S.shape[1]))])      # index -1 -> master row of zeros
    return Sx[ki] - Sx[kj]


def solve_network(pairs, I, sig, n_dates, master, n_iter=3, huber_k=2.5,
                  chunk=None, ridge=1e-6, datum=None, require_connected=False, exact_sigma=True):
    """Per-pixel weighted least squares  I_p = S_i - S_j + c_p,  S_master = 0.

    pairs   list of (i, j) date indices, one per pair (ref, rep)
    I, sig  (P, ny, nx) raw estimates and sigmas (NaN = invalid); weights are
            1/sig^2 (sig comes from both bands' coherence, Cramer-Rao)
    c_p     one spatially constant datum per pair, estimated (weighted mean
            residual) because each pair's unwrapping fixes it only up to a
            constant; this removes the spatial mean of the screens (datum is
            unobservable anyway).  Huber IRLS down-weights isolated bad cells.
    datum   optional (P,) pair constants to hold FIXED (no datum update); used
            when the image is inverted in row chunks, with the datum estimated
            beforehand on a subsample of the whole image.  Pixels are then
            independent; Huber IRLS still runs per pixel, and after the first
            solve only pixels whose weights changed are solved again.
    require_connected  if True, a date gets S = NaN at a pixel unless it is
            tied to the master through pairs that are valid AT THAT PIXEL
            (otherwise only the small ridge term would pin its value).
    chunk   pixels per batched solve (default: about 200 MB of K x K matrices)
    exact_sigma  (fixed datum only) False: skip the batched matrix inverse and
            approximate sigma_S by 1/sqrt(diag N) (a lower bound; ~2.5x faster
            for ~40 dates, more for more dates).  S itself is unchanged.
    Returns dict: S (D, ny, nx), sigma_S (D, ny, nx), c (P,), resid (P, ny, nx)
    standardized residuals, qc (per-pair dict).  Dates with no valid pair at a
    pixel get S = NaN there."""
    P, ny, nx = I.shape
    if P != len(pairs):
        raise ValueError("pairs and I disagree")
    if not check_connected(pairs, n_dates):
        raise ValueError("pair network is not connected: some dates cannot be tied to the master")
    cols = [d for d in range(n_dates) if d != master]
    pos = {d: k for k, d in enumerate(cols)}
    K = len(cols)
    ki = np.array([pos.get(i, -1) for i, _ in pairs])
    kj = np.array([pos.get(j, -1) for _, j in pairs])
    if chunk is None:
        chunk = int(max(1000, min(200000, 2.5e7 // (K * K))))

    valid = np.isfinite(I) & np.isfinite(sig) & (sig > 0)
    w0 = np.where(valid, 1.0 / np.where(valid, sig, 1.0) ** 2, 0.0)
    w0 = w0.reshape(P, -1)
    y = np.where(valid, I, 0.0).reshape(P, -1).astype(np.float64)
    npix = y.shape[1]
    S = np.zeros((K, npix))
    covd = np.zeros((K, npix))

    fixed = datum is not None
    if fixed:
        c = np.asarray(datum, dtype=np.float64).copy()
        if c.shape != (P,):
            raise ValueError("datum must have one value per pair")
        # pixels independent.  Per chunk: LU solve for all pixels, then after
        # each Huber update re-solve only the pixels whose weights changed;
        # the final pass uses inv(N) (needed for sigma) and gives S with it.
        # (numpy's batched inv costs ~4 LU solves, so it is done once.)
        for s in range(0, npix, chunk):
            e = min(s + chunk, npix)
            W = w0[:, s:e]
            Y = y[:, s:e] - c[:, None]
            N, rhs = _normal_eq(ki, kj, W, Y, K)
            N = _regularize(N, ridge)
            if n_iter <= 1:
                Ninv = np.linalg.inv(N)
                S[:, s:e] = np.einsum("nij,nj->in", Ninv, rhs)
                covd[:, s:e] = np.diagonal(Ninv, axis1=1, axis2=2).T
                continue
            Sc = np.linalg.solve(N, rhs[:, :, None])[:, :, 0].T
            wt = W
            for it in range(1, n_iter):
                z = (Y - _predict(Sc, ki, kj)) * np.sqrt(W)
                hub = np.where(np.abs(z) > huber_k, huber_k / np.maximum(np.abs(z), 1e-12), 1.0)
                wn = W * hub
                ch = np.nonzero(np.any(wn != wt, axis=0))[0]
                wt = wn
                if it < n_iter - 1:
                    if ch.size:
                        Nc, rc = _normal_eq(ki, kj, wn[:, ch], Y[:, ch], K)
                        N[ch] = _regularize(Nc, ridge)
                        Sc[:, ch] = np.linalg.solve(N[ch], rc[:, :, None])[:, :, 0].T
                elif exact_sigma:
                    # final weights: rebuild changed pixels, invert everything once
                    if ch.size:
                        Nc, _ = _normal_eq(ki, kj, wn[:, ch], Y[:, ch], K)
                        N[ch] = _regularize(Nc, ridge)
                    _, rhs = _normal_eq(ki, kj, wn, Y, K)
                    Ninv = np.linalg.inv(N)
                    Sc = np.einsum("nij,nj->in", Ninv, rhs)
                    covd[:, s:e] = np.diagonal(Ninv, axis1=1, axis2=2).T
                    del Ninv
                else:
                    if ch.size:
                        Nc, rc = _normal_eq(ki, kj, wn[:, ch], Y[:, ch], K)
                        N[ch] = _regularize(Nc, ridge)
                        Sc[:, ch] = np.linalg.solve(N[ch], rc[:, :, None])[:, :, 0].T
                    covd[:, s:e] = 1.0 / np.diagonal(N, axis1=1, axis2=2).T
            S[:, s:e] = Sc
            del N
    else:
        # initial datum: weighted median per pair
        c = np.zeros(P)
        for p in range(P):
            v = valid[p].ravel()
            if v.any():
                c[p] = _wmedian(y[p, v], w0[p, v])
        wt = w0.copy()
        for it in range(n_iter):
            for s in range(0, npix, chunk):
                e = min(s + chunk, npix)
                N, rhs = _normal_eq(ki, kj, wt[:, s:e], y[:, s:e] - c[:, None], K)
                N = _regularize(N, ridge)
                if it == n_iter - 1:
                    Ninv = np.linalg.inv(N)
                    S[:, s:e] = np.einsum("nij,nj->in", Ninv, rhs)
                    covd[:, s:e] = np.diagonal(Ninv, axis1=1, axis2=2).T
                    del Ninv
                else:
                    S[:, s:e] = np.linalg.solve(N, rhs[:, :, None])[:, :, 0].T
                del N
            pred = _predict(S, ki, kj)
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
    vp = w0 > 0
    cnt = np.zeros((K, npix))
    for p in range(P):
        if ki[p] >= 0:
            cnt[ki[p]] += vp[p]
        if kj[p] >= 0:
            cnt[kj[p]] += vp[p]
    if require_connected:
        # dates reachable from the master through pairs valid at each pixel
        reach = np.zeros((n_dates, npix), dtype=bool)
        reach[master] = True
        for _ in range(n_dates):
            new = reach.copy()
            for p, (i, j) in enumerate(pairs):
                new[i] |= vp[p] & reach[j]
                new[j] |= vp[p] & reach[i]
            if np.array_equal(new, reach):
                break
            reach = new
        for k, d in enumerate(cols):
            cnt[k] = np.where(reach[d], cnt[k], 0)
    Sfull = np.full((n_dates, npix), np.nan, dtype=np.float32)
    sigfull = np.full((n_dates, npix), np.nan, dtype=np.float32)
    Sfull[master] = 0.0
    sigfull[master] = 0.0
    for k, d in enumerate(cols):
        okk = cnt[k] > 0
        Sfull[d, okk] = S[k, okk]
        sigfull[d, okk] = np.sqrt(np.maximum(covd[k, okk], 0.0))
    r = y - c[:, None] - _predict(S, ki, kj)
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


def _conv1d(a, k, axis, max_bytes=2e8):
    """Zero-padded 'same' convolution of a 2-D array with kernel k along one
    axis, by 1-D FFTs over slabs of the other axis (bounded memory)."""
    a = np.asarray(a, dtype=np.float64)
    if axis == 0:
        return _conv1d(a.T, k, 1, max_bytes).T
    ny, nx = a.shape
    pk = len(k) // 2
    Nf = 1 << int(np.ceil(np.log2(nx + len(k))))
    Kf = np.fft.rfft(k, n=Nf)[None, :]
    out = np.empty((ny, nx), dtype=np.float64)
    rows = max(1, int(max_bytes // (Nf * 16 * 3)))
    for r0 in range(0, ny, rows):
        r1 = min(r0 + rows, ny)
        out[r0:r1] = np.fft.irfft(np.fft.rfft(a[r0:r1], n=Nf, axis=1) * Kf, n=Nf, axis=1)[:, pk:pk + nx]
    return out


def _conv2_sep(a, ky, kx):
    """Zero-padded separable convolution (same size): along x, then along y,
    each with 1-D FFTs in slabs, so full-resolution grids fit in memory."""
    return _conv1d(_conv1d(a, kx, 1), ky, 0)


def _trend_design(ny, nx, order):
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    x = 2 * x / max(nx - 1, 1) - 1
    y = 2 * y / max(ny - 1, 1) - 1
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    return np.stack([x ** i * y ** j for i, j in terms], axis=-1)   # (ny, nx, T)


def _trend_terms(order):
    return [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]


def _trend_rows(r0, r1, ny, nx, terms):
    """Trend design rows r0:r1 as (n, T) float64."""
    y, x = np.mgrid[r0:r1, 0:nx].astype(np.float64)
    x = 2 * x / max(nx - 1, 1) - 1
    y = 2 * y / max(ny - 1, 1) - 1
    return np.stack([(x ** i * y ** j).ravel() for i, j in terms], axis=-1)


def reject_outliers(S, sigma, order=2, k_out=5.0, n_iter=3, rows=256):
    """Outlier rejection against a weighted polynomial trend (the first stage of
    filter_screen): fit a trend of `order` by LS with weights 1/sigma^2, flag
    cells with |S - trend| > k_out * max(sigma, robust sigma of the residuals),
    refit on the inliers, n_iter times.  Computed in row blocks so it runs on
    full-resolution grids without building the (ny, nx, T) design array.
    Returns (inliers, trend, residual)  (residual = S - trend, 0 where invalid)."""
    S = np.asarray(S)
    sigma = np.asarray(sigma)
    ny, nx = S.shape
    ok = np.isfinite(S) & np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < 20:
        raise ValueError("too few valid cells for the outlier test")
    w = np.where(ok, 1.0 / np.where(ok, sigma, 1.0) ** 2, 0.0)
    Sz = np.where(ok, S, 0.0)
    terms = _trend_terms(order)
    T = len(terms)
    inl = ok.copy()
    trend = np.zeros((ny, nx))
    res = np.zeros((ny, nx))
    for _ in range(n_iter):
        N = np.zeros((T, T))
        rhs = np.zeros(T)
        for r0 in range(0, ny, rows):
            r1 = min(r0 + rows, ny)
            G = _trend_rows(r0, r1, ny, nx, terms)
            wv = np.where(inl[r0:r1], w[r0:r1], 0.0).ravel()
            Gw = G * wv[:, None]
            N += G.T @ Gw
            rhs += Gw.T @ Sz[r0:r1].ravel()
        coef = np.linalg.lstsq(N, rhs, rcond=None)[0]
        for r0 in range(0, ny, rows):
            r1 = min(r0 + rows, ny)
            trend[r0:r1] = (_trend_rows(r0, r1, ny, nx, terms) @ coef).reshape(r1 - r0, nx)
        res = np.where(ok, S - trend, 0.0)
        rs = res[inl]
        sig_rob = 1.4826 * np.median(np.abs(rs - np.median(rs)))
        inl = ok & (np.abs(res) <= k_out * np.maximum(sigma, sig_rob))
    return inl, trend, res


def filter_screen(S, sigma, sig_y_cells, sig_x_cells, order=2, k_out=5.0,
                  prior_frac=0.02, n_trend_iter=3, posterior_sigma=False):
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
    normalized convolution (ignores trend-fit uncertainty).
    posterior_sigma=True (used when the filtered field is itself an input to a
    weighted inversion): sigma_f also includes the uncertainty of falling back
    to the trend, sqrt(den2/den^2 + (s_prior * eps/den)^2) with s_prior the
    robust spread of the residuals about the trend, so filled cells far from
    any data get about the residual spread instead of a near-zero sigma."""
    ny, nx = S.shape
    ok = np.isfinite(S) & np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < 20:
        raise ValueError("too few valid cells to filter")
    w = np.where(ok, 1.0 / np.where(ok, sigma, 1.0) ** 2, 0.0)
    inl, trend, res = reject_outliers(S, sigma, order=order, k_out=k_out, n_iter=n_trend_iter)
    wv = np.where(inl, w, 0.0)
    eps = prior_frac * np.median(w[inl])
    del w
    ky, kx = _gauss_kernel(sig_y_cells), _gauss_kernel(sig_x_cells)
    num = _conv2_sep(wv * np.where(inl, res, 0.0), ky, kx)
    if posterior_sigma:
        rr = res[inl]
        s_prior = 1.4826 * float(np.median(np.abs(rr - np.median(rr))))
    del res
    den = _conv2_sep(wv, ky, kx) + eps
    out = (trend + num / den).astype(np.float32)
    del num, trend
    # FFT round-off can leave tiny negative values far from any data
    den2 = np.maximum(_conv2_sep(wv, ky ** 2, kx ** 2), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if posterior_sigma:
            sig_f = np.sqrt(den2 / den ** 2 + (s_prior * eps / den) ** 2).astype(np.float32)
        else:
            sig_f = (np.sqrt(den2) / den).astype(np.float32)
    return out, sig_f, inl


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


# ------------------------------------------------- row-chunk worker (sbas) ---
def invert_rows(task):
    """Invert rows r0:r1 of disk-backed stacks with a fixed datum and write the
    result into disk-backed outputs.  One task per process-pool job.
    task = (r0, r1, pairs, n_dates, master, datum, paths) with paths
    {"I", "sig", "S_raw", "S_sig"} (.npy).  Returns (r0, r1, acc) where acc is
    (P, 3): n, sum of z^2, count |z| > 3 per pair."""
    r0, r1, pairs, n_dates, master, datum, paths = task[:7]
    exact = task[7] if len(task) > 7 else True
    stI = np.load(paths["I"], mmap_mode="r")
    stS = np.load(paths["sig"], mmap_mode="r")
    res = solve_network(pairs, np.asarray(stI[:, r0:r1]), np.asarray(stS[:, r0:r1]), n_dates, master,
                        datum=datum, require_connected=True, exact_sigma=exact)
    S_raw = np.load(paths["S_raw"], mmap_mode="r+")
    S_sig = np.load(paths["S_sig"], mmap_mode="r+")
    S_raw[:, r0:r1] = res["S"]
    S_sig[:, r0:r1] = res["sigma_S"]
    S_raw.flush()
    S_sig.flush()
    acc = np.zeros((len(pairs), 3))
    for p, q in enumerate(res["qc"]):
        if q["n"]:
            acc[p] = (q["n"], q["rms_z"] ** 2 * q["n"], q["frac_z_gt3"] * q["n"])
    return r0, r1, acc
