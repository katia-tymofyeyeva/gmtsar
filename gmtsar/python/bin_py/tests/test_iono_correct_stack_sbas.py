#!/usr/bin/env python3
"""iono_correct_stack --method sbas: filter each pair's raw estimate (outlier
rejection, gap filling), then SBAS-like full-resolution inversion (parallel row
chunks), no filter afterwards; step 5 unchanged.  Same synthetic stack and
in-memory grid stub as test_iono_correct_stack, plus: an unwrapping-slip patch
in one pair, a gross outlier on one date, and a hole where date 3 has no valid
pair (filled by the per-pair filter)."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import test_iono_correct_stack as base   # installs the gmt_grd_io stub, loads the driver
from test_iono_correct_stack import (_write, _read, STORE, CMDS, ics, ie, DATES, PAIRS, MASTER,
                                     NXB, NY, NXA, RATIO, FS_A, FS_B, write_prm, screen_fn)

SLIP_PAIR, SLIP = 1, (slice(10, 20), slice(30, 45))        # pair (1,2): +2pi*r/(r+1) slip
OUT_DATE, OUT_PIX = 2, (50, 100)                            # gross outlier on date 2
HOLE_DATE, HOLE = 3, (slice(70, 80), slice(10, 25))         # no valid pair for date 3 here
ARGS = ["--datum-stride", "2", "--sigma-km", "0.3"]


class TestSbas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        STORE.clear()
        CMDS.clear()
        cls.root = tempfile.mkdtemp()
        cls.old = os.getcwd()
        os.chdir(cls.root)
        rng = np.random.default_rng(1)
        for d in ("SLC", "SLC_B", "topo"):
            os.makedirs(d)
        Path("topo/trans.dat").write_bytes(b"x")
        clocks = [2026001.5, 2026013.5, 2026025.5, 2026037.5, 2026049.5]
        for d, stem in enumerate(DATES):
            write_prm(f"SLC/{stem}.PRM", FS_A, clocks[d], NXA * 4)
        write_prm(f"SLC_B/{DATES[MASTER][:-1]}B.PRM", FS_B, clocks[0], NXB)
        Path("config.txt").write_text(f"master_image = {DATES[MASTER]}\nthreshold_geocode = 0.1\n"
                                      "filter_wavelength = 200\niono_method = sbas\n")
        Path("intf.in").write_text("\n".join(f"{DATES[i]}:{DATES[j]}" for i, j in PAIRS) + "\n")
        xB = np.arange(NXB) + 0.5
        y = 4.0 + 8.0 * np.arange(NY)
        cls.xB, cls.y = xB, y
        cls.truth = np.stack([screen_fn((xB * RATIO)[None, :], y[:, None], d) for d in range(len(DATES))])
        xA = 2.0 + 4.0 * np.arange(NXA)
        cls.datum = rng.uniform(-3, 3, len(PAIRS))
        cls.sub, cls.raw = [], {}
        r = 1.2935 / 1.2390
        for p, (i, j) in enumerate(PAIRS):
            sub = f"{int(clocks[i])}_{int(clocks[j])}"
            cls.sub.append(sub)
            I = cls.truth[i] - cls.truth[j] + cls.datum[p] + rng.normal(0, 0.25, (NY, NXB))
            if p == SLIP_PAIR:
                I[SLIP] += 2 * np.pi * r / (r + 1)
            for dd, sign in ((i, 1.0), (j, -1.0)):
                if dd == OUT_DATE:
                    I[OUT_PIX] += sign * 60.0
                if dd == HOLE_DATE:
                    I[HOLE] = np.nan
            sg = np.full((NY, NXB), 0.25)
            _write(f"iono_est/{sub}/I.grd", I.astype(np.float32), xB, y, node_offset=1)
            _write(f"iono_est/{sub}/sigma.grd", sg.astype(np.float32), xB, y, node_offset=1)
            unw = (np.add.outer(np.zeros(NY), 0.05 * xA) + 20.0 * (p + 1)).astype(np.float32)
            wph = ie.wrap(unw + rng.normal(0, 0.3, unw.shape)).astype(np.float32)
            for nm, arr in (("unwrap", unw), ("phasefilt", wph), ("phase", wph),
                            ("corr", np.full((NY, NXA), 0.5, np.float32))):
                _write(f"intf/{sub}/{nm}.grd", arr, xA, y, node_offset=1)
            Path(f"intf/{sub}/{DATES[i]}.PRM").write_text("x")
            cls.raw[sub] = (unw.copy(), wph.copy())
        cls.run_driver(ARGS + ["--workers", "2", "--chunk-rows", "10"])

    @staticmethod
    def run_driver(extra):
        argv = sys.argv
        sys.argv = ["iono_correct_stack", "intf.in", "config.txt"] + extra
        try:
            ics.main()
        finally:
            sys.argv = argv

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.old)
        shutil.rmtree(cls.root, ignore_errors=True)

    def screen(self, d, kind="screen"):
        return _read(f"iono_screens_sbas/{DATES[d]}_{kind}.grd")[0]

    def test_outputs_in_separate_dirs(self):
        self.assertTrue(os.path.isdir("iono_screens_sbas") and os.path.isdir("intf_iono_sbas"))
        self.assertFalse(os.path.exists("iono_screens") or os.path.exists("intf_iono"))

    def test_full_resolution_grid(self):
        S, x, y, info = _read(f"iono_screens_sbas/{DATES[1]}_screen.grd")
        self.assertEqual(S.shape, (NY, NXB))                       # no block averaging
        np.testing.assert_allclose(x, self.xB * RATIO)
        np.testing.assert_allclose(y, self.y)
        self.assertTrue(np.all(self.screen(MASTER) == 0))

    def test_screens_accurate_and_complete(self):
        for d in range(1, len(DATES)):
            S = self.screen(d)
            self.assertTrue(np.isfinite(S).all())                   # gaps filled before inversion
            e = S - self.truth[d]
            e = e - np.median(e)
            self.assertLess(np.sqrt(np.mean(e ** 2)), 0.2, DATES[d])

    def test_smooth_because_pairs_filtered(self):
        # each pair's raw estimate has 0.25 rad white noise; after per-pair
        # filtering and inversion the screens are far smoother than that
        for d in range(1, len(DATES)):
            self.assertLess(np.std(np.diff(self.screen(d), axis=1)), 0.05)

    def test_no_raw_grid_written(self):
        self.assertFalse(os.path.exists(f"iono_screens_sbas/{DATES[1]}_raw.grd"))
        self.assertTrue(os.path.exists(f"iono_screens_sbas/{DATES[1]}_sigma.grd"))

    def test_slip_and_outlier_suppressed(self):
        for d, region in ((1, SLIP), (2, SLIP), (OUT_DATE, OUT_PIX)):
            S, T = self.screen(d), self.truth[d]
            off = np.median(S - T)
            self.assertLess(np.max(np.abs((S - T - off)[region])), 0.6, (d, region))

    def test_hole_filled(self):
        S, T = self.screen(HOLE_DATE), self.truth[HOLE_DATE]
        off = np.median(S - T)
        self.assertLess(np.abs(S[HOLE] - T[HOLE] - off).max(), 0.6)

    def test_matches_direct_inversion_of_filtered_pairs(self):
        # the screens are exactly the inversion of the stacked filtered estimates
        import iono_net_lib as nl
        I = np.load("iono_screens_sbas/I.npy")
        sg = np.load("iono_screens_sbas/sig.npy")
        self.assertTrue(np.isfinite(I).all() and (sg > 0).all())
        datum = [float(l.split()[5]) for l in
                 Path("iono_screens_sbas/pair_qc.txt").read_text().splitlines()[1:]]
        r = nl.solve_network(PAIRS, I, sg, len(DATES), MASTER, datum=np.array(datum),
                             require_connected=True)
        for d in range(len(DATES)):
            np.testing.assert_allclose(self.screen(d), r["S"][d], atol=2e-3)

    def test_step5_unchanged_and_closure(self):
        for p, (i, j) in enumerate(PAIRS):
            sub = self.sub[p]
            corr = _read(f"intf_iono_sbas/{sub}/ph_iono.grd")[0]
            unw_c = _read(f"intf_iono_sbas/{sub}/unwrap.grd")[0]
            unw0, wph0 = self.raw[sub]
            self.assertTrue(np.isfinite(corr).all())
            self.assertLess(np.abs(unw_c - (unw0 - corr)).max(), 1e-4)
            np.testing.assert_array_equal(_read(f"intf/{sub}/unwrap.grd")[0], unw0)
        for a, b, c in ((0, 1, 4), (1, 2, 5), (2, 3, 6)):
            cu = [_read(f"intf_iono_sbas/{self.sub[k]}/unwrap.grd")[0] for k in (a, b, c)]
            ru = [self.raw[self.sub[k]][0] for k in (a, b, c)]
            d = (cu[0] + cu[1] - cu[2]) - (ru[0] + ru[1] - ru[2])
            self.assertLess(np.abs(d).max(), 2e-4)

    def test_geocode_and_qc(self):
        self.assertGreaterEqual(len([c for c in CMDS if c.startswith("geocode")]), len(PAIRS))
        lines = Path("iono_screens_sbas/pair_qc.txt").read_text().strip().splitlines()
        self.assertEqual(len(lines), len(PAIRS) + 1)

    def test_z_cache_and_workers(self):
        import io
        import contextlib
        before = self.screen(2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.run_driver(ARGS)
        self.assertIn("using cached screens", buf.getvalue())
        np.testing.assert_array_equal(self.screen(2), before)
        # a different filter length redoes the pair filter and the inversion
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.run_driver(["--datum-stride", "2", "--sigma-km", "0.5", "--force"])
        self.assertIn("per-pair filter: sigma 0.5 km", buf.getvalue())
        self.assertGreater(np.abs(self.screen(2) - before).max(), 1e-3)
        # one worker gives the same result as two
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_driver(ARGS + ["--recompute", "--force", "--workers", "1", "--chunk-rows", "37"])
        np.testing.assert_allclose(self.screen(2), before, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
