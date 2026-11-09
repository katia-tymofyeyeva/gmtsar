#!/usr/bin/env python3
"""test_iono_est_lib -- synthetic tests for utils/iono_est_lib.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

import iono_est_lib as ie  # noqa: E402

FA, FB = 1.2390e9, 1.2935e9
R = FB / FA


def screen(n=120):
    y, x = np.mgrid[0:n, 0:n] / n
    return 4.0 * np.sin(2 * np.pi * x * 0.8) * np.cos(2 * np.pi * y * 0.6) + 2.0 * x - 1.0 * y


class TestCoarsen(unittest.TestCase):
    def test_columns_and_centering(self):
        a = np.arange(12, dtype=float).reshape(2, 6)
        c = ie.coarsen_range(a, 2, 3)
        self.assertEqual(c.shape, (2, 3))
        self.assertTrue(np.allclose(c[0], [0.5, 2.5, 4.5]))    # pairs (0,1),(2,3),(4,5)
        c = ie.coarsen_range(np.arange(14, dtype=float).reshape(2, 7), 2, 3)   # extra column dropped
        self.assertEqual(c.shape, (2, 3))
        with self.assertRaises(ValueError):
            ie.coarsen_range(a, 2, 4)

    def test_phase_and_coherence(self):
        # fully coherent smooth field: coarsened phase = field at cell centres, coherence 1
        ny, nx, f = 4, 40, 2
        x = np.arange(nx)
        ph = 0.05 * x[None, :] + np.zeros((ny, 1))
        amp = np.full((ny, nx), 3.0)
        re, im = amp * np.cos(ph), amp * np.sin(ph)
        p, c = ie.complex_on_coarse_grid(re, im, amp, amp, f, nx // f)
        centre = 0.05 * (np.arange(nx // f) * f + 0.5)
        self.assertLess(np.abs(p[0] - centre).max(), 1e-3)
        self.assertTrue(np.all(c > 0.99) and np.all(c <= 1.0))

    def test_incoherent_and_empty(self):
        rng = np.random.default_rng(0)
        re, im = rng.normal(size=(50, 400)), rng.normal(size=(50, 400))
        amp = np.ones((50, 400))
        _, c = ie.complex_on_coarse_grid(re, im, amp * 2, amp * 2, 2, 200)
        self.assertLess(np.nanmean(c), 0.5)
        z = np.zeros((2, 4))
        p, c = ie.complex_on_coarse_grid(z, z, z, z, 2, 2)
        self.assertTrue(np.isnan(p).all() and np.isnan(c).all())


class TestCoarseUnwrap(unittest.TestCase):
    def test_block_mean_and_upsample(self):
        b, ny, nx = 8, 64, 96
        yy, xx = np.mgrid[0:ny, 0:nx]
        truth = 0.1 * xx + 0.04 * yy                     # wraps ~1.5 times across the grid
        coh = np.full((ny, nx), 0.9)
        pc, cc, nr, nc = ie.block_mean_phasor(ie.wrap(truth), coh, b)
        self.assertEqual((nr, nc), (8, 12))
        self.assertTrue(np.all(cc > 0.9))
        # unwrap the coarse ramp by hand (what snaphu would return) and rebuild
        yc, xc = np.mgrid[0:nr, 0:nc]
        coarse_truth = 0.1 * (xc * b + (b - 1) / 2) + 0.04 * (yc * b + (b - 1) / 2)
        self.assertLess(np.abs(ie.wrap(pc - coarse_truth)).max(), 1e-3)
        unw = ie.unwrap_via_coarse(ie.wrap(truth), coarse_truth.astype(np.float32), b)
        self.assertLess(np.abs(unw - truth).max(), 5e-2)

    def test_noisy_cells_survive(self):
        rng = np.random.default_rng(3)
        b, ny, nx = 8, 128, 128
        yy, xx = np.mgrid[0:ny, 0:nx]
        truth = 0.25 * xx + 0.05 * yy
        noisy = ie.wrap(truth + rng.normal(0, 0.8, truth.shape))
        coh = np.full(truth.shape, 0.5)
        pc, cc, nr, nc = ie.block_mean_phasor(noisy, coh, b)
        yc, xc = np.mgrid[0:nr, 0:nc]
        ct = 0.25 * (xc * b + 3.5) + 0.05 * (yc * b + 3.5)
        self.assertLess(np.abs(ie.wrap(pc - ct)).max(), 0.5)    # averaging beats the noise
        unw = ie.unwrap_via_coarse(noisy, ct.astype(np.float32), b)
        err = unw - truth
        self.assertLess(np.abs(err.mean()), 0.1)                 # no 2*pi bias
        self.assertGreater(np.mean(np.abs(err) < 3.0), 0.999)    # nearly no 2*pi slips

    def test_nan_fill_and_dropped_edge(self):
        z = np.arange(36, dtype=np.float32).reshape(6, 6)
        z[2:4, 2:4] = np.nan
        f = ie.fill_nearest(z, 2)
        self.assertTrue(np.isfinite(f).all())
        z2 = np.full((6, 6), np.nan, np.float32)
        z2[0, 0] = 1.0
        self.assertTrue(np.isnan(ie.fill_nearest(z2, 2)[5, 5]))
        with self.assertRaises(ValueError):
            ie.block_mean_phasor(np.zeros((4, 4)), np.ones((4, 4)), 8)


class TestReference(unittest.TestCase):
    def test_reuse_recovers_unwrapped(self):
        rng = np.random.default_rng(5)
        ny, nxb, f = 40, 50, 2
        xb = np.arange(nxb)
        truth_b = 0.4 * xb[None, :] + 0.05 * np.arange(ny)[:, None]       # unwrapped, B grid
        noisy = ie.wrap(truth_b + rng.normal(0, 0.5, truth_b.shape))       # plain multilook
        # A's own unwrap on a 2x finer grid (+2 extra columns like the real grids)
        xa = (np.arange(nxb * f + 2) + 0.5) / f - 0.5
        ref = 0.4 * xa[None, :] + 0.05 * np.arange(ny)[:, None] + 2 * np.pi      # reference one cycle off
        phi, bad = ie.unwrap_from_reference(noisy, ref, f)
        self.assertAlmostEqual(float(np.mean(phi - truth_b)), 2 * np.pi, delta=0.05)   # keeps the reference's cycle
        self.assertLess(np.abs((phi - truth_b) - 2 * np.pi).max(), 2.5)              # cell noise kept, no slips
        self.assertLess(bad, 0.01)

    def test_masked_stays_nan_and_flags_bad_reference(self):
        ny, nxb, f = 10, 20, 2
        noisy = np.zeros((ny, nxb), dtype=np.float32)
        ref = np.zeros((ny, nxb * f), dtype=np.float32)
        ref[3, 10:12] = np.nan                                # masked in A's unwrap
        phi, bad = ie.unwrap_from_reference(noisy, ref, f)
        self.assertTrue(np.isnan(phi[3, 5]))
        self.assertTrue(np.isfinite(phi[0, 0]))
        ref2 = ref.copy()
        ref2[:5] = 2.5                                        # reference disagrees with phase
        _, bad2 = ie.unwrap_from_reference(noisy, ref2, f)
        self.assertGreater(bad2, 0.4)
        with self.assertRaises(ValueError):
            ie.unwrap_from_reference(noisy, ref[:5], f)


class TestBoxColumns(unittest.TestCase):
    def test_fractional_ratio_exact(self):
        # linear ramp: the box average over [j*r,(j+1)*r) is the ramp at the centre
        r, n, ncols = 1.9995, 4000, 1900
        x = np.arange(n, dtype=np.float64)
        a = np.tile((0.01 * (x + 0.5)).astype(np.float32), (3, 1))
        o = ie.box_columns(a, r, ncols)
        centres = 0.01 * (np.arange(ncols) + 0.5) * r
        self.assertLess(np.abs(o[0] - centres).max(), 1e-3)
        # integer pairing at ratio 2 would be off by 0.01*0.0005*j: ~0.01 rad at the far end
        o2 = ie.box_columns(a, 2.0, ncols)
        self.assertGreater(np.abs(o2[0] - centres).max(), 0.005)

    def test_matches_old_pairing_for_ratio_2(self):
        a = np.arange(12, dtype=float).reshape(2, 6)
        self.assertTrue(np.allclose(ie.box_columns(a, 2.0, 3)[0], [0.5, 2.5, 4.5]))

    def test_strict_nan(self):
        a = np.ones((1, 12), dtype=np.float32)
        a[0, 5] = np.nan
        o = ie.box_columns(a, 2.5, 4, strict=True)       # cells cover [0,2.5) [2.5,5) [5,7.5) [7.5,10)
        self.assertTrue(np.isnan(o[0, 2]) and np.isfinite(o[0, [0, 1, 3]]).all())
        o = ie.box_columns(a, 2.5, 4, strict=False)
        self.assertTrue(np.isfinite(o).all())
        with self.assertRaises(ValueError):
            ie.box_columns(a, 2.5, 5)


class TestRatio(unittest.TestCase):
    def test_gains(self):
        r, a, b = ie.band_ratio(1.239, 1.2935)
        self.assertAlmostEqual(a, 12.12, delta=0.02)
        self.assertAlmostEqual(b, -11.61, delta=0.02)
        self.assertAlmostEqual(a + b * r, 0.0 + (a + b * r), places=9)
        # I = a*phiA + b*phiB with phiA = I+N, phiB = I/r + rN must give I and cancel N
        self.assertAlmostEqual(a + b / r, 1.0, places=9)
        self.assertAlmostEqual(a + b * r, 0.0, places=9)
        with self.assertRaises(ValueError):
            ie.band_ratio(1.3, 1.2)


class TestEstimate(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.I = screen()
        # large non-dispersive phase: many cycles (deformation/atmosphere/topo residual)
        y, x = np.mgrid[0:120, 0:120]
        self.N = 40.0 * (x / 120.0) + 15.0 * np.sin(y / 17.0)
        self.coh = np.full(self.I.shape, 0.8)

    def make(self, sig_a=0.0, sig_b=0.0):
        pa = self.I + self.N + self.rng.normal(0, sig_a, self.I.shape) if sig_a else self.I + self.N
        pb = self.I / R + R * self.N
        if sig_b:
            pb = pb + self.rng.normal(0, sig_b, self.I.shape)
        return pa, ie.wrap(pb)

    def test_noise_free_recovery(self):
        pa, pb = self.make()
        out = ie.dispersive_estimate(pa, pb, self.coh, self.coh, R, 100.0)
        self.assertLess(np.abs(out["I"] - self.I).max(), 1e-9)
        self.assertTrue(out["valid"].all())

    def test_noise_matches_predicted_sigma(self):
        sa, sb = 0.05, 0.10
        pa, pb = self.make(sa, sb)
        # build coh/n_eff giving exactly sa, sb is awkward: check the propagation
        # formula directly instead
        out = ie.dispersive_estimate(pa, pb, self.coh, self.coh, R, 100.0)
        err = out["I"] - self.I
        k = R / (R * R - 1)
        expect = k * np.sqrt(sb ** 2 + (R * sa) ** 2)
        self.assertAlmostEqual(err.std(), expect, delta=0.05 * expect)

    def test_sigma_formula(self):
        s = ie.phase_sigma(0.8, 100.0)
        self.assertAlmostEqual(float(s), np.sqrt((1 - 0.64) / (2 * 100 * 0.64)), places=9)
        out = ie.dispersive_estimate(np.zeros((2, 2)), np.zeros((2, 2)),
                                     np.full((2, 2), 0.8), np.full((2, 2), 0.8), R, 100.0)
        k = R / (R * R - 1)
        self.assertAlmostEqual(float(out["sigma"][0, 0]),
                               float(k * s * np.sqrt(1 + R * R)), places=9)

    def test_slip_shift(self):
        pa, pb = self.make()
        pa_slip = pa.copy()
        pa_slip[40:80, 40:80] += 2 * np.pi          # a 2*pi unwrapping slip
        out = ie.dispersive_estimate(pa_slip, pb, self.coh, self.coh, R, 100.0)
        d = (out["I"] - self.I)[50:70, 50:70]
        self.assertAlmostEqual(abs(d.mean()), ie.slip_shift(R), delta=1e-6)
        self.assertAlmostEqual(ie.slip_shift(R), 3.21, delta=0.01)
        self.assertLess(np.abs((out["I"] - self.I)[:30, :30]).max(), 1e-9)

    def test_mask(self):
        pa, pb = self.make()
        coh_b = self.coh.copy()
        coh_b[:10] = 0.1
        pa[5, 5] = np.nan
        out = ie.dispersive_estimate(pa, pb, self.coh, coh_b, R, 100.0, coh_min=0.2)
        self.assertFalse(out["valid"][:10].any())
        self.assertTrue(np.isnan(out["I"][5, 5]) and np.isnan(out["sigma"][0, 0]))
        self.assertTrue(out["valid"][20:, :].all())

    def test_n_eff(self):
        n = ie.n_eff_gaussian(200.0, 4.2, 35.0)
        self.assertGreater(n, 50)
        self.assertLess(n, 500)
        self.assertEqual(ie.n_eff_gaussian(1.0, 10.0, 10.0), 1.0)


if __name__ == "__main__":
    unittest.main()
