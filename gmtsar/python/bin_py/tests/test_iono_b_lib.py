#!/usr/bin/env python3
"""test_iono_b_lib -- tests for utils/iono_b_lib.py (A/B parameter table and
derivation of Frequency-B offset grids from A's grids)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

import iono_b_lib as ib  # noqa: E402

C = ib.C


def write_prm(path, near_range, fs, prf=1650.0, clock=2026260.5, nx=4000, ny=800,
              wl=0.2420):
    with open(path, "w") as f:
        f.write(f"near_range = {near_range}\nrng_samp_rate = {fs}\nPRF = {prf}\n"
                f"SC_clock_start = {clock}\nradar_wavelength = {wl}\n"
                f"num_rng_bins = {nx}\nnum_patches = 1\nnum_valid_az = {ny}\n")


FS_A, FS_B = 48.0e6, 6.0e6
DA, DB = C / (2 * FS_A), C / (2 * FS_B)
NA_M, NA_R = 900000.0, 900010.7
NB_M, NB_R = 900001.2, 900011.4   # B near range differs slightly from A's


def delta_r_true(R):
    return 2.0 + 1.0e-3 * (R - 905000.0)    # smooth slant-range change (m)


def make_A_grid(incx, nxn, nyn):
    """Truth-based A offset grid (A pixels) on nodes ix*incx."""
    xm = np.arange(nxn) * incx
    R = NA_M + xm * DA
    xr = (R + delta_r_true(R) - NA_R) / DA
    r = np.tile(xr - xm, (nyn, 1))
    return r


class TestBRangeGrid(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        j = lambda n: os.path.join(self.d, n)
        for name, nr, fs in (("mA", NA_M, FS_A), ("aA", NA_R, FS_A),
                             ("mB", NB_M, FS_B), ("aB", NB_R, FS_B)):
            write_prm(j(name + ".PRM"), nr, fs)
        self.p = {k: ib.slc_params(j(k + ".PRM")) for k in ("mA", "aA", "mB", "aB")}

    def test_slc_params(self):
        p = self.p["mB"]
        self.assertAlmostEqual(p["dr"], DB, places=9)
        self.assertEqual((p["nx"], p["ny"]), (4000, 800))
        self.assertAlmostEqual(p["freq"], C / 0.2420, delta=1.0)

    def test_matches_truth(self):
        incx, nxn, nyn = 64.0, 60, 5
        rA = make_A_grid(incx, nxn, nyn)
        rB = ib.b_range_grid(rA, incx, self.p["mA"], self.p["aA"],
                             self.p["mB"], self.p["aB"])
        # B nodes: same slant range as A node ix
        incB = incx * DA / DB
        xB = np.arange(nxn) * incB
        R = NB_M + xB * DB
        truth = (R + delta_r_true(R) - NB_R) / DB - xB
        # skip last columns where extrapolation matters
        self.assertLess(np.abs(rB[:, :-2] - truth[:-2]).max(), 1e-3)

    def test_zero_geometry_gives_pure_near_range_terms(self):
        # identical everything, A offset = near-range difference only
        incx = 64.0
        nxn = 20
        rA = np.full((3, nxn), (NA_R - NA_M) / DA * 0.0)
        # no change in slant range and no near-range diff encoded -> dR_A =
        # (NA_R-NA_M); pass r=0 -> dR = NA_R-NA_M
        rB = ib.b_range_grid(rA, incx, self.p["mA"], self.p["aA"],
                             self.p["mB"], self.p["aB"])
        expect = ((NA_R - NA_M) - (NB_R - NB_M)) / DB
        self.assertTrue(np.allclose(rB, expect))

    def test_timing_check(self):
        j = lambda n: os.path.join(self.d, n)
        write_prm(j("bad.PRM"), NB_M, FS_B, prf=1700.0)
        with self.assertRaises(ValueError):
            ib.check_ab_timing(self.p["mA"], ib.slc_params(j("bad.PRM")))
        write_prm(j("late.PRM"), NB_M, FS_B, clock=2026260.5 + 5e-9)  # ~0.43 ms
        with self.assertRaises(ValueError):
            ib.check_ab_timing(self.p["mA"], ib.slc_params(j("late.PRM")))
        ib.check_ab_timing(self.p["mA"], self.p["mB"])  # ok


class TestBuildGrids(unittest.TestCase):
    def test_roundtrip(self):
        try:
            import netCDF4  # noqa: F401
            from gmt_grd_io import write_gmt_grd, read_gmt_grd
        except Exception:
            self.skipTest("netCDF4 not available")
        d = tempfile.mkdtemp()
        j = lambda n: os.path.join(d, n)
        for name, nr, fs in (("mA", NA_M, FS_A), ("aA", NA_R, FS_A),
                             ("mB", NB_M, FS_B), ("aB", NB_R, FS_B)):
            write_prm(j(name + ".PRM"), nr, fs)
        incx, incy, nxn, nyn = 64.0, 64.0, 40, 6
        rA = make_A_grid(incx, nxn, nyn)
        aA = np.full((nyn, nxn), 1.25)
        x = np.arange(nxn) * incx
        y = np.arange(nyn) * incy
        write_gmt_grd(j("r.grd"), rA.astype(np.float32), x, y)
        write_gmt_grd(j("a.grd"), aA.astype(np.float32), x, y)
        diag = ib.build_b_grids(j("mA.PRM"), j("aA.PRM"), j("mB.PRM"), j("aB.PRM"),
                                j("r.grd"), j("a.grd"), j("rB.grd"), j("aB.grd"))
        rB, xB, yB, info = read_gmt_grd(j("rB.grd"))
        aB, _, _, _ = read_gmt_grd(j("aB.grd"))
        self.assertEqual(rB.shape, rA.shape)
        self.assertAlmostEqual(info["x_inc"], incx * DA / DB, places=6)
        self.assertAlmostEqual(info["y_inc"], incy, places=6)
        self.assertTrue(np.allclose(aB, 1.25))          # azimuth unchanged
        self.assertAlmostEqual(diag["scale"], DA / DB, places=9)


class TestTable(unittest.TestCase):
    def test_table_and_missing(self):
        d = tempfile.mkdtemp()
        raw = os.path.join(d, "raw")
        os.makedirs(raw)
        write_prm(os.path.join(raw, "NSR_20260101A.PRM"), NA_M, FS_A)
        write_prm(os.path.join(raw, "NSR_20260101B.PRM"), NB_M, FS_B, wl=0.2317)
        write_prm(os.path.join(raw, "NSR_20260113A.PRM"), NA_M, FS_A)
        rows = ib.ab_table(["NSR_20260101A", "NSR_20260113A"], raw)
        self.assertEqual(rows[0]["timing"], "ok")
        self.assertEqual(rows[1]["timing"], "missing")
        txt = ib.format_table(rows)
        self.assertIn("NSR_20260101A", txt)
        self.assertIn("missing", txt)
        with self.assertRaises(ValueError):
            ib.ab_table(["NSR_20260101B"], raw)


if __name__ == "__main__":
    unittest.main()
