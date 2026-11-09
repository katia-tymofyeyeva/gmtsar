#!/usr/bin/env python3
"""test_iono_net_lib -- synthetic tests for utils/iono_net_lib.py: block
reduction, network inversion, per-date filtering and the closure property."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

import iono_net_lib as nl  # noqa: E402


def smooth_screen(ny, nx, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    f = np.zeros((ny, nx))
    for _ in range(3):
        kx, ky = rng.uniform(0.5, 2.0) * np.pi / nx, rng.uniform(0.5, 2.0) * np.pi / ny
        f += rng.uniform(0.5, 2.0) * np.sin(kx * x + rng.uniform(0, 6)) * np.cos(ky * y + rng.uniform(0, 6))
    return f + rng.uniform(-0.02, 0.02) * x + rng.uniform(-0.02, 0.02) * y


# network: 6 dates (master = 2), consecutive + skip pairs
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (1, 3), (2, 4), (3, 5), (0, 3)]
D, MASTER = 6, 2


def make_data(ny=30, nx=34, sigma=0.3, seed=0, gaps=False):
    rng = np.random.default_rng(seed)
    S = np.stack([smooth_screen(ny, nx, 10 + d) for d in range(D)])
    S = S - S[MASTER]                                   # master = 0
    I = np.empty((len(PAIRS), ny, nx))
    sg = np.full((len(PAIRS), ny, nx), sigma)
    datum = rng.uniform(-3, 3, len(PAIRS))
    for p, (i, j) in enumerate(PAIRS):
        I[p] = S[i] - S[j] + datum[p] + rng.normal(0, sigma, (ny, nx))
    if gaps:
        I[:, 10:16, 12:20] = np.nan
    return S, I, sg, datum


class TestBlockReduce(unittest.TestCase):
    def test_mean_and_sigma(self):
        rng = np.random.default_rng(1)
        I = 2.0 + rng.normal(0, 0.5, (64, 64))
        s = np.full((64, 64), 0.5)
        Ib, sb, n = nl.block_reduce(I, s, 8, 8, n_indep_per_cell=1.0)
        self.assertEqual(Ib.shape, (8, 8))
        self.assertAlmostEqual(float(np.nanmean(Ib)), 2.0, delta=0.05)
        self.assertAlmostEqual(float(np.nanmean(sb)), 0.5 / 8, delta=1e-3)
        self.assertAlmostEqual(float(np.std(Ib)), 0.5 / 8, delta=0.02)
        # correlated cells: only 1/4 of the cells are independent -> sigma doubles
        _, sb2, _ = nl.block_reduce(I, s, 8, 8, n_indep_per_cell=0.25)
        self.assertAlmostEqual(float(np.nanmean(sb2)), 0.5 / 4, delta=1e-3)

    def test_outliers_and_nan(self):
        I = np.zeros((16, 16))
        I[:2, :2] = 3.2                                  # a few slip cells in block (0,0)
        I[8:, 8:] = np.nan                               # empty block
        s = np.full((16, 16), 0.3)
        Ib, sb, n = nl.block_reduce(I, s, 8, 8)
        self.assertAlmostEqual(float(Ib[0, 0]), 0.0, places=6)
        self.assertEqual(int(n[0, 0]), 60)
        self.assertTrue(np.isnan(Ib[1, 1]))
        with self.assertRaises(ValueError):
            nl.block_reduce(I, s, 32, 32)


class TestNetwork(unittest.TestCase):
    def test_recovers_screens_and_datum(self):
        S, I, sg, datum = make_data()
        out = nl.solve_network(PAIRS, I, sg, D, MASTER)
        err = out["S"] - S
        # screens are determined up to a constant per date (datum), compare without it
        err = err - np.nanmean(err, axis=(1, 2), keepdims=True)
        self.assertLess(np.sqrt(np.nanmean(err ** 2)), 0.3)         # better than raw sigma
        self.assertTrue(np.all(out["S"][MASTER] == 0))
        self.assertLess(abs(float(np.nanmean(out["resid"]))), 0.05)
        qc = out["qc"]
        self.assertEqual(len(qc), len(PAIRS))
        # LS absorbs degrees of freedom: mean z^2 should be (P - K)/P, K = D - 1 unknown screens
        z2 = np.mean([q["rms_z"] ** 2 for q in qc])
        self.assertAlmostEqual(float(z2), (len(PAIRS) - (D - 1)) / len(PAIRS), delta=0.05)

    def test_slip_flagged_and_downweighted(self):
        S, I, sg, _ = make_data(seed=2)
        I2 = I.copy()
        I2[3, 5:25, 5:25] += 3.2                                    # slip region in one pair
        clean = nl.solve_network(PAIRS, I, sg, D, MASTER)
        bad = nl.solve_network(PAIRS, I2, sg, D, MASTER)
        qc = bad["qc"]
        worst = max(range(len(PAIRS)), key=lambda p: qc[p]["frac_z_gt3"])
        self.assertEqual(worst, 3)
        self.assertGreater(qc[3]["frac_z_gt3"], 0.1)
        # the slip leaks little into the screens thanks to redundancy + Huber weights
        d = np.nanmax(np.abs(bad["S"] - clean["S"]))
        self.assertLess(d, 2.0)

    def test_gaps_and_disconnected(self):
        S, I, sg, _ = make_data(gaps=True)
        out = nl.solve_network(PAIRS, I, sg, D, MASTER)
        self.assertTrue(np.isnan(out["S"][0, 12, 14]))              # no data anywhere there
        self.assertTrue(np.isfinite(out["S"][0, 0, 0]))
        with self.assertRaises(ValueError):
            nl.solve_network([(0, 1), (3, 4)], I[:2], sg[:2], D, MASTER)


class TestFilter(unittest.TestCase):
    def test_denoises_fills_gap_no_edge_bias(self):
        ny, nx = 60, 70
        S0 = smooth_screen(ny, nx, 3) * 2.0
        rng = np.random.default_rng(4)
        sig = np.full((ny, nx), 0.6)
        S = S0 + rng.normal(0, 0.6, (ny, nx))
        S[20:30, 25:40] = np.nan                                     # gap
        Sf, sf, inl = nl.filter_screen(S, sig, 4.0, 4.0)
        e = Sf - S0
        self.assertLess(np.sqrt(np.mean(e[~np.isnan(S)] ** 2)), 0.25)    # well below 0.6 raw
        self.assertLess(np.sqrt(np.mean(e[20:30, 25:40] ** 2)), 0.5)       # gap filled from neighbours
        # edges are not pulled toward zero by the zero padding
        edge = np.r_[e[0, :], e[-1, :], e[:, 0], e[:, -1]]
        self.assertLess(abs(edge.mean()), 0.15)
        self.assertTrue(np.all(np.isfinite(sf)))

    def test_outlier_rejected_and_far_gap_falls_back_to_trend(self):
        ny, nx = 50, 50
        y, x = np.mgrid[0:ny, 0:nx].astype(float)
        S0 = 0.05 * x + 0.02 * y
        S = S0.copy()
        S[10, 10] = 50.0                                             # a wild cell
        S[:, 35:] = np.nan                                           # large data-free region
        sig = np.full((ny, nx), 0.1)
        Sf, _, inl = nl.filter_screen(S, sig, 2.0, 2.0)
        self.assertFalse(inl[10, 10])
        self.assertLess(abs(Sf[10, 10] - S0[10, 10]), 0.1)
        # far from data the screen is the (extrapolated) trend: linear field -> small error
        self.assertLess(abs(Sf[25, 48] - S0[25, 48]), 0.3)

    def test_closure_preserved(self):
        # raw pair phases with arbitrary content; correction F(S_i) - F(S_j) per date
        S, I, sg, _ = make_data(seed=5)
        out = nl.solve_network(PAIRS, I, sg, D, MASTER)
        F = np.stack([nl.filter_screen(out["S"][d], out["sigma_S"][d], 3.0, 3.0)[0]
                      if d != MASTER else np.zeros(out["S"].shape[1:], np.float32)
                      for d in range(D)])
        rng = np.random.default_rng(6)
        phi = {pr: rng.normal(0, 5, F.shape[1:]) for pr in PAIRS}           # arbitrary unwrapped pair phases
        corr = {pr: phi[pr] - (F[pr[0]] - F[pr[1]]) for pr in PAIRS}
        # triplets that close in the network: (0,1),(1,2),(0,2) and (1,2),(2,3),(1,3)
        for (a, b, c) in (((0, 1), (1, 2), (0, 2)), ((1, 2), (2, 3), (1, 3)), ((2, 3), (3, 4), (2, 4))):
            raw = phi[a] + phi[b] - phi[c]
            new = corr[a] + corr[b] - corr[c]
            self.assertLess(np.abs(raw - new).max(), 1e-4)      # float32 round-off only


class TestInterp(unittest.TestCase):
    def test_linear_field_exact_and_nonuniform_ratio(self):
        xc = (np.arange(40) + 0.5) * 16.0 * 7.998          # coarse centres, A px
        yc = (np.arange(30) + 0.5) * 16.0 * 8.0
        zc = (0.001 * xc[None, :] + 0.002 * yc[:, None]).astype(np.float32)
        xt = 2.0 + 4.0 * np.arange(4000)                    # A cell centres
        yt = 4.0 + 8.0 * np.arange(1000)
        o = nl.interp_to_grid(zc, xc, yc, xt, yt)
        inside_x = (xt >= xc[0]) & (xt <= xc[-1])
        inside_y = (yt >= yc[0]) & (yt <= yc[-1])
        expect = 0.001 * xt[None, :] + 0.002 * yt[:, None]
        self.assertLess(np.abs(o - expect)[np.ix_(inside_y, inside_x)].max(), 1e-3)
        # outside the centres: edge value, no extrapolated blow-up
        self.assertTrue(np.isfinite(o).all())
        self.assertAlmostEqual(float(o[0, 0]), float(zc[0, 0]), places=5)

    def test_nan_propagates_locally(self):
        xc = np.arange(10.0)
        yc = np.arange(10.0)
        zc = np.zeros((10, 10), np.float32)
        zc[5, 5] = np.nan
        o = nl.interp_to_grid(zc, xc, yc, np.arange(0, 9.5, 0.5), np.arange(0, 9.5, 0.5))
        self.assertTrue(np.isnan(o[10, 10]))
        self.assertTrue(np.isfinite(o[0, 0]) and np.isfinite(o[18, 18]))


if __name__ == "__main__":
    unittest.main()
