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
