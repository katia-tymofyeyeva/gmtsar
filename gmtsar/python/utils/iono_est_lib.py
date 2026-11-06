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
