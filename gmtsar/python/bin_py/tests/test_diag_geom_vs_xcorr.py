#!/usr/bin/env python3
"""test_diag_geom_vs_xcorr -- logic tests for utils/diag_geom_vs_xcorr.

No GMT / SAT_llt2rat_py / xcorr_py / real data in the sandbox that wrote
this, so these test the numerical logic with synthetic inputs of known
answer (known offsets in -> known residuals out), plus the orchestration
in main() with the external tools and the DEM reader mocked.
"""
from __future__ import annotations

import importlib.machinery as _ilm
import importlib.util as _ilu
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_UTILS_DIR = _HERE.parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

_spec = _ilu.spec_from_loader(
    "diag_mod",
    _ilm.SourceFileLoader("diag_mod", str(_UTILS_DIR / "diag_geom_vs_xcorr")),
)
diag = _ilu.module_from_spec(_spec)
sys.modules["diag_mod"] = diag
_spec.loader.exec_module(diag)

DIMS = (53316, 54720)


def true_dr(r, a):
    return 2.0 + 1.0e-4 * r - 5.0e-5 * a + 2.0e-9 * r * a


def true_da(r, a):
    return -1.5 + 3.0e-5 * r + 8.0e-5 * a - 1.0e-9 * a * a / 10.0


class TestPolynomials(unittest.TestCase):
    def test_exponent_count(self):
        self.assertEqual(len(diag.poly_exponents(3)), 10)
        self.assertEqual(len(diag.poly_exponents(1)), 3)

    def test_fit_recovers_cubic_exactly(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, DIMS[0], 500)
        y = rng.uniform(0, DIMS[1], 500)
        z = true_dr(x, y)
        coef = diag.fit_poly(x, y, z, DIMS, 3)
        pred = diag.eval_poly(coef, x, y, DIMS, 3)
        self.assertLess(np.max(np.abs(pred - z)), 1e-6)

    def test_robust_fit_ignores_outliers(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, DIMS[0], 400)
        y = rng.uniform(0, DIMS[1], 400)
        z = true_dr(x, y) + rng.normal(0, 0.01, 400)
        z[:20] += 25.0  # gross outliers
        coef, keep = diag.fit_poly_robust(x, y, z, DIMS, 3)
        pred = diag.eval_poly(coef, x, y, DIMS, 3)
        self.assertLess(np.median(np.abs(pred[20:] - true_dr(x, y)[20:])), 0.02)
        self.assertFalse(keep[:20].any())


class TestGeometricOffsets(unittest.TestCase):
    def test_sign_and_bounds(self):
        # master pixels, repeat = master + (dr, da); two points out of bounds
        m = np.array([[100.0, 200.0], [5000.0, 6000.0], [-5.0, 100.0], [100.0, 99999.0]])
        a = m + np.array([[1.5, -2.0]] * 4)
        r_m, a_m, dr, da = diag.geometric_offsets(m, a, DIMS, DIMS)
        self.assertEqual(r_m.size, 2)
        np.testing.assert_allclose(dr, 1.5)
        np.testing.assert_allclose(da, -2.0)

    def test_drops_nonfinite(self):
        m = np.array([[100.0, 200.0], [np.nan, 5.0]])
        a = np.array([[101.0, 201.0], [3.0, 4.0]])
        r_m, *_ = diag.geometric_offsets(m, a, DIMS, DIMS)
        self.assertEqual(r_m.size, 1)


class TestDecimate(unittest.TestCase):
    def test_stride_and_nan_drop(self):
        data = np.ones((100, 200), np.float32)
        data[0, 0] = np.nan
        x = np.linspace(-120, -119, 200)
        y = np.linspace(35, 36, 100)
        pts, stride = diag.decimate_dem(data, x, y, 5000)
        self.assertGreaterEqual(stride, 2)
        self.assertEqual(pts.shape[1], 3)
        self.assertTrue(np.isfinite(pts).all())
        self.assertLessEqual(len(pts), 20000 / 4 + 100)


class TestAnalyze(unittest.TestCase):
    def _geo(self, n=3000, seed=2):
        rng = np.random.default_rng(seed)
        r = rng.uniform(100, DIMS[0] - 100, n)
        a = rng.uniform(100, DIMS[1] - 100, n)
        return r, a, true_dr(r, a), true_da(r, a)

    def _xcorr(self, const_r=0.3, const_a=-0.05, noise=0.02, n=40, seed=3):
        rng = np.random.default_rng(seed)
        xs = np.linspace(2000, DIMS[0] - 2000, n)
        ys = np.linspace(2000, DIMS[1] - 2000, n)
        X, Y = np.meshgrid(xs, ys)
        x, y = X.ravel(), Y.ravel()
        dr = true_dr(x, y) + const_r + rng.normal(0, noise, x.size)
        da = true_da(x, y) + const_a + rng.normal(0, noise, x.size)
        snr = rng.uniform(10, 90, x.size)
        return np.column_stack([x, dr, y, da, snr])

    def test_constant_bias_recovered(self):
        res = diag.analyze(self._xcorr(), self._geo(), DIMS, 20.0, 3)
        self.assertAlmostEqual(res["stats_r"]["median"], 0.3, delta=0.01)
        self.assertAlmostEqual(res["stats_a"]["median"], -0.05, delta=0.01)
        self.assertLess(res["geo_fit_rms"][0], 1e-6)
        self.assertAlmostEqual(res["plane_r"]["const"], 0.3, delta=0.01)
        self.assertLess(abs(res["plane_r"]["across_range"]), 0.01)
        # noise floor should be near the injected 0.02
        self.assertLess(res["floor_r"], 0.04)

    def test_snr_cutoff_applied(self):
        xc = self._xcorr()
        res = diag.analyze(xc, self._geo(), DIMS, 50.0, 3)
        self.assertEqual(res["n_kept"], int((xc[:, 4] > 50.0).sum()))
        self.assertTrue((res["snr"] > 50.0).all())

    def test_gradient_detected(self):
        xc = self._xcorr(const_r=0.0, const_a=0.0, noise=0.005)
        # add a range-residual ramp of +0.5 px across the full range width
        xc[:, 1] += 0.5 * (xc[:, 0] / DIMS[0] - 0.5)
        res = diag.analyze(xc, self._geo(), DIMS, 20.0, 3)
        self.assertAlmostEqual(res["plane_r"]["across_range"], 0.5, delta=0.03)

    def test_blocks_shape(self):
        res = diag.analyze(self._xcorr(), self._geo(), DIMS, 20.0, 3)
        self.assertEqual(res["blocks_r"].shape, (4, 4))
        self.assertTrue(np.isfinite(res["blocks_r"]).all())

    def test_no_points_above_cutoff(self):
        res = diag.analyze(self._xcorr(), self._geo(), DIMS, 1000.0, 3)
        self.assertEqual(res["n_kept"], 0)
        diag.print_report(res, None, 1000.0, 3)  # must not raise


class TestPrmAndOutliers(unittest.TestCase):
    def test_read_prm_value_last_occurrence_wins(self):
        """First real run (2026-10-05): SAT_baseline_py appends rshift/
        ashift BELOW the raw PRM's own `rshift = 0`; the report showed
        seed (0, 0) while xcorr_py was really seeded with (-18, -1740)."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.PRM")
            with open(p, "w") as f:
                f.write("rshift = 0\nashift = 0\nPRF = 1520\n"
                        "rshift = -18\nashift = -1740\n")
            self.assertEqual(diag.read_prm_value(p, "rshift"), "-18")
            self.assertEqual(diag.read_prm_value(p, "ashift"), "-1740")
            self.assertEqual(diag.read_prm_value(p, "PRF"), "1520")
            self.assertIsNone(diag.read_prm_value(p, "nope"))

    def test_outliers_counted_and_excluded_from_clean_stats(self):
        rng = np.random.default_rng(7)
        n = 40
        xs = np.linspace(2000, DIMS[0] - 2000, n)
        ys = np.linspace(2000, DIMS[1] - 2000, n)
        X, Y = np.meshgrid(xs, ys)
        x, y = X.ravel(), Y.ravel()
        dr = true_dr(x, y) + rng.normal(0, 0.02, x.size)
        da = true_da(x, y) + rng.normal(0, 0.02, x.size)
        dr[:10] += 60.0          # 10 gross range outliers
        da[10:15] -= 75.0        # 5 gross azimuth outliers
        xc = np.column_stack([x, dr, y, da, np.full(x.size, 60.0)])
        r = rng.uniform(100, DIMS[0] - 100, 3000)
        a = rng.uniform(100, DIMS[1] - 100, 3000)
        res = diag.analyze(xc, (r, a, true_dr(r, a), true_da(r, a)), DIMS, 20.0, 3)
        self.assertEqual(res["n_outliers"], 15)
        self.assertLess(res["stats_r_clean"]["std"], 0.05)
        self.assertGreater(res["stats_r"]["std"], 1.0)


class TestXcorrParse(unittest.TestCase):
    def test_load_xcorr_format(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "freq_xcorr.dat")
            with open(p, "w") as f:
                f.write(" 3000 1.250 4000 -0.500  55.20 \n 3100 1.300 4000 -0.480  12.00 \n")
            xc = diag.load_xcorr(p)
        self.assertEqual(xc.shape, (2, 5))
        self.assertAlmostEqual(xc[0, 1], 1.25)
        self.assertAlmostEqual(xc[0, 3], -0.5)
        self.assertAlmostEqual(xc[1, 4], 12.0)


class TestMainOrchestration(unittest.TestCase):
    """main() with SAT_llt2rat_py and the DEM reader mocked: the 'tool'
    maps (lon, lat) -> pixel with a known affine transform per PRM, the
    repeat adding a known (dr, da) -- so the report must find those."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cwd0 = os.getcwd()
        os.chdir(self.tmp)
        os.makedirs("raw")
        os.makedirs("topo")
        for stem in ("NSR_M", "NSR_R"):
            with open(f"raw/{stem}.PRM", "w") as f:
                f.write("num_rng_bins = 53316\nnum_patches = 1\nnum_valid_az = 54720\n"
                        "rshift = 0\nashift = 0\n")
            open(f"raw/{stem}.SLC", "w").close()
            open(f"raw/{stem}.LED", "w").close()
        open("topo/dem.grd", "w").close()
        np.savetxt("freq.dat", self._xcorr(), fmt=" %.0f %.3f %.0f %.3f %.2f")
        self._orig = (diag.read_dem, diag.run_llt2rat, diag._need)

    def tearDown(self):
        diag.read_dem, diag.run_llt2rat, diag._need = self._orig
        os.chdir(self.cwd0)
        shutil.rmtree(self.tmp)

    def _xcorr(self):
        xs = np.linspace(3000, 50000, 12)
        ys = np.linspace(3000, 50000, 12)
        X, Y = np.meshgrid(xs, ys)
        x, y = X.ravel(), Y.ravel()
        # measured = true geometry + a constant 0.25 px range bias
        return np.column_stack([x, true_dr(x, y) + 0.25, y, true_da(x, y), np.full(x.size, 60.0)])

    def test_report_finds_injected_bias(self):
        rng = np.random.default_rng(5)
        lon = rng.uniform(0, 1, 6000)
        lat = rng.uniform(0, 1, 6000)
        h = rng.uniform(0, 500, 6000)

        data = np.zeros((60, 100), np.float32)
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 60)
        diag.read_dem = lambda path: (data, x, y)
        diag._need = lambda tool: None

        def fake_llt2rat(prm, pts, precise):
            r = 1000.0 + pts[:, 0] * 50000.0
            a = 1000.0 + pts[:, 1] * 50000.0
            if prm.endswith("NSR_R.raw.PRM"):
                return np.column_stack([r + true_dr(r, a), a + true_da(r, a),
                                        pts[:, 2], pts[:, 0], pts[:, 1]])
            return np.column_stack([r, a, pts[:, 2], pts[:, 0], pts[:, 1]])

        diag.run_llt2rat = fake_llt2rat
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = diag.main(["NSR_M", "NSR_R", "--xcorr-dat", "freq.dat",
                            "--dem-points", "3000"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("RESIDUAL = measured", out)
        self.assertTrue(os.path.isfile("diag_NSR_M_NSR_R/diag_residuals.txt"))
        tab = np.loadtxt("diag_NSR_M_NSR_R/diag_residuals.txt")
        # column 5 = dr_res: injected constant bias of 0.25 px
        self.assertAlmostEqual(float(np.median(tab[:, 5])), 0.25, delta=0.02)
        self.assertAlmostEqual(float(np.median(tab[:, 8])), 0.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
