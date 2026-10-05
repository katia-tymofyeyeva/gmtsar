#!/usr/bin/env python3
"""test_geom_align_lib -- logic tests for utils/geom_align_lib.py.

No GMT / SAT_llt2rat_py / data in the sandbox: geometry is injected via a
mocked geometric_model, external commands via a recording run_cmd.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

import geom_align_lib as gal  # noqa: E402

DIMS = (53316, 54720)


def geo_dr(r, a):
    return 2.0 + 1.0e-4 * r - 5.0e-5 * a


def geo_da(r, a):
    return -1.5 + 3.0e-5 * r + 8.0e-5 * a


def make_geo():
    pts = np.array([[0.0, 0.0], [DIMS[0], DIMS[1]]])
    return {
        "coef_r": gal.fit_poly(*_grid(), geo_dr(*_grid()), DIMS, 3),
        "coef_a": gal.fit_poly(*_grid(), geo_da(*_grid()), DIMS, 3),
        "dims": DIMS, "order": 3, "n_points": 1000, "stride": 4,
        "fit_rms": (0.0, 0.0), "span": (0, 1, 0, 1), "empty_blocks": 0, "pts": pts,
    }


def _grid():
    rng = np.random.default_rng(1)
    return rng.uniform(0, DIMS[0], 400), rng.uniform(0, DIMS[1], 400)


def make_xcorr(shift_r=0.0, shift_a=0.0, n_bad=0, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, DIMS[0], 300)
    y = rng.uniform(0, DIMS[1], 300)
    dr = geo_dr(x, y) + shift_r + rng.normal(0, 0.02, x.size)
    da = geo_da(x, y) + shift_a + rng.normal(0, 0.02, x.size)
    dr[:n_bad] += 20.0
    snr = np.full(x.size, 60.0)
    return np.column_stack([x, dr, y, da, snr])


class TestMethodSelect(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop(gal.ENV_METHOD, None)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        self._env.stop()

    def _cfg(self, text):
        p = os.path.join(self.tmp, "config.txt")
        with open(p, "w") as f:
            f.write(text)
        return p

    def test_default_is_geo(self):
        self.assertEqual(gal.align_method(None), "geo")
        self.assertEqual(gal.align_method(self._cfg("threshold_snaphu = 0\n")), "geo")

    def test_config_xcorr(self):
        self.assertEqual(gal.align_method(self._cfg("align_method = xcorr\n")), "xcorr")

    def test_env_overrides_config(self):
        os.environ[gal.ENV_METHOD] = "geo"
        self.assertEqual(gal.align_method(self._cfg("align_method = xcorr\n")), "geo")

    def test_invalid_exits(self):
        with self.assertRaises(SystemExit):
            gal.align_method(self._cfg("align_method = bogus\n"))


class TestPrm(unittest.TestCase):
    def test_last_value_wins(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "a.PRM")
        with open(p, "w") as f:
            f.write("rshift = 0\nnum_rng_bins = 10\nnum_patches = 2\nnum_valid_az = 5\nrshift = 3.5\n")
        self.assertEqual(gal.read_prm_value(p, "rshift"), "3.5")
        self.assertEqual(gal.prm_dims(p), (10, 10))


class TestCorrection(unittest.TestCase):
    def test_recovers_constant_plus_plane(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0, DIMS[0], 300)
        y = rng.uniform(0, DIMS[1], 300)
        u, v = 2 * x / DIMS[0] - 1, 2 * y / DIMS[1] - 1
        res_r = 0.30 + 0.05 * u - 0.02 * v
        res_a = -0.10 + 0.01 * u + 0.04 * v
        c = gal.fit_correction(x, y, res_r, res_a, DIMS, order=1)
        pr = gal.eval_poly(c["coef_r"], x, y, DIMS, 1)
        self.assertLess(np.abs(pr - res_r).max(), 1e-8)
        self.assertEqual(c["n_outliers"], 0)

    def test_outliers_excluded(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(0, DIMS[0], 300)
        y = rng.uniform(0, DIMS[1], 300)
        res = np.full(300, 0.2)
        res_bad = res.copy()
        res_bad[:20] = 5.0
        c = gal.fit_correction(x, y, res_bad, res, DIMS, order=1)
        self.assertEqual(c["n_outliers"], 20)
        self.assertAlmostEqual(float(gal.eval_poly(c["coef_r"], x[:1], y[:1], DIMS, 1)[0]), 0.2, places=6)

    def test_too_many_outliers_exits(self):
        x = np.linspace(0, DIMS[0], 100)
        y = np.linspace(0, DIMS[1], 100)
        res = np.full(100, 3.0)
        with self.assertRaises(SystemExit):
            gal.fit_correction(x, y, res, res, DIMS)

    def test_large_correction_exits(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(0, DIMS[0], 200)
        y = rng.uniform(0, DIMS[1], 200)
        u = 2 * x / DIMS[0] - 1
        res = 0.45 * u
        with self.assertRaises(SystemExit):
            gal.fit_correction(x, y, res, res, DIMS, max_correction_px=0.1)


class TestLattice(unittest.TestCase):
    def test_lattice_equals_geometry_plus_correction(self):
        geo = make_geo()
        corr = {"coef_r": np.array([0.3, 0.0, 0.0]), "coef_a": np.array([-0.1, 0.0, 0.0]), "order": 1}
        xyz_r, xyz_a = gal.lattice_model(geo, corr, (0, DIMS[0], 0, DIMS[1]), spacing=4096)
        x, y = xyz_r[:, 0], xyz_r[:, 1]
        self.assertLess(np.abs(xyz_r[:, 2] - (geo_dr(x, y) + 0.3)).max(), 1e-6)
        self.assertLess(np.abs(xyz_a[:, 2] - (geo_da(x, y) - 0.1)).max(), 1e-6)
        self.assertAlmostEqual(x.min(), 0.0)
        self.assertAlmostEqual(x.max(), DIMS[0])

    def test_parse_region(self):
        self.assertEqual(gal.parse_region("-R0/53316/0/54720\n"), (0.0, 53316.0, 0.0, 54720.0))
        with self.assertRaises(ValueError):
            gal.parse_region("nothing")


class TestGeoFitoffset(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.d)
        # netCDF4 absent in some sandboxes: exercise the `gmt grdmath` branch.
        self._env = mock.patch.dict(os.environ, {"GMTSAR_GRDMATH_PY": "0"})
        self._env.start()

    def tearDown(self):
        os.chdir(self.cwd)
        self._env.stop()

    def _run(self, xc, **kw):
        np.savetxt("freq_xcorr.dat", xc, fmt="%f")
        cmds = []
        with mock.patch.object(gal, "geometric_model", return_value=make_geo()):
            out = gal.geo_fitoffset("m.PRM", "a.PRM0", "dem.grd", "freq_xcorr.dat",
                                    run_cmd=cmds.append, region=(0, DIMS[0], 0, DIMS[1]), **kw)
        return out, cmds

    def test_constant_shift_found_in_model_and_commands_issued(self):
        # Monkeypatch savetxt capture: model lattice is written before surface runs,
        # and removed by the final rm command -> capture via np.savetxt spy.
        saved = {}
        real = np.savetxt

        def spy(fname, arr, *a, **k):
            saved[fname] = np.array(arr)
            return real(fname, arr, *a, **k)

        with mock.patch.object(np, "savetxt", spy):
            (geo, corr), cmds = self._run(make_xcorr(shift_r=0.4, shift_a=-0.2))
        x, y, z = saved["r_model.xyz"].T
        self.assertLess(np.abs(z - (geo_dr(x, y) + 0.4)).max(), 0.02)
        x, y, z = saved["a_model.xyz"].T
        self.assertLess(np.abs(z - (geo_da(x, y) - 0.2)).max(), 0.02)
        joined = "\n".join(cmds)
        self.assertIn("gmt surface r_model.xyz", joined)
        self.assertIn("-rp -I64/64 -T.3 -Gr_tmp.grd", joined)
        self.assertIn("a_tmp.grd", joined)
        self.assertIn("FLIPUD", joined)

    def test_outlier_points_rejected_not_fatal(self):
        (geo, corr), _ = self._run(make_xcorr(n_bad=15))
        self.assertEqual(corr["n_outliers"], 15)

    def test_too_few_points_exits(self):
        xc = make_xcorr()
        xc[:, 4] = 5.0
        with self.assertRaises(SystemExit):
            self._run(xc)

    def test_wrong_geometry_exits(self):
        xc = make_xcorr(shift_r=3.0)  # all points 3 px from geometry
        with self.assertRaises(SystemExit):
            self._run(xc)


class TestFitAlignmentGrids(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.d)
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop(gal.ENV_METHOD, None)

    def tearDown(self):
        os.chdir(self.cwd)
        self._env.stop()

    def test_missing_dem_is_fatal_with_clear_message(self):
        with self.assertRaises(SystemExit) as cm:
            gal.fit_alignment_grids("m", "a", dem_path="../topo/dem.grd", run_cmd=lambda c: None)
        self.assertIn("dem.grd", str(cm.exception))
        self.assertIn("align_method = xcorr", str(cm.exception))

    def test_xcorr_method_runs_legacy_fitoffset_without_dem(self):
        os.environ[gal.ENV_METHOD] = "xcorr"
        cmds = []
        m = gal.fit_alignment_grids("m", "a", dem_path="nope.grd", run_cmd=cmds.append)
        self.assertEqual(m, "xcorr")
        self.assertEqual(cmds, ["fitoffset_ra 10 10 freq_xcorr.dat 20"])

    def test_geo_method_calls_geo_fitoffset_with_expected_files(self):
        open("dem.grd", "w").close()
        with mock.patch.object(gal, "geo_fitoffset") as g:
            m = gal.fit_alignment_grids("M", "A", dem_path="dem.grd", run_cmd=lambda c: None)
        self.assertEqual(m, "geo")
        args = g.call_args[0]
        self.assertEqual(args[:4], ("M.PRM", "A.PRM0", "dem.grd", "freq_xcorr.dat"))


if __name__ == "__main__":
    unittest.main()
