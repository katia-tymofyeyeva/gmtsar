#!/usr/bin/env python3
"""test_iono_correct_stack -- integration test of utils/iono_correct_stack on a
synthetic stack, with an in-memory stand-in for gmt_grd_io (no netCDF4 / GMT in
the sandbox) and a recording stand-in for gmtsar_lib.run.

Truth: known per-date screens S_d(x, y); per-pair raw estimates I_p = S_ref -
S_rep + datum_p + noise on a B-like grid; A-like grids (7.998x finer in range,
different coordinates) for unwrap/phasefilt/phase/corr.  Checks: screens and the
per-pair correction recover the truth, corrected unwrap = unwrap - correction,
phasefilt/phase wrapped, ph_iono.grd written, geocode invoked in intf_iono, the
originals in intf/ untouched, and closure preserved through the real outputs.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

_UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

STORE = {}


def _write(path, data, x, y, node_offset=0, **kw):
    STORE[os.path.abspath(path)] = (np.array(data), np.array(x, dtype=float), np.array(y, dtype=float),
                                    {"node_offset": node_offset, "x_inc": float(x[1] - x[0]),
                                     "y_inc": float(y[1] - y[0])})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"x")


def _read(path):
    d, x, y, info = STORE[os.path.abspath(path)]
    return d.copy(), x.copy(), y.copy(), dict(info)


stub = types.ModuleType("gmt_grd_io")
stub.write_gmt_grd = _write
stub.read_gmt_grd = _read
sys.modules["gmt_grd_io"] = stub

import iono_est_lib as ie  # noqa: E402

CMDS = []
_loader = importlib.machinery.SourceFileLoader("iono_correct_stack", str(_UTILS_DIR / "iono_correct_stack"))
_spec = importlib.util.spec_from_loader("iono_correct_stack", _loader)
ics = importlib.util.module_from_spec(_spec)
_loader.exec_module(ics)
ics.run = lambda cmd, *a, **k: CMDS.append(cmd)

FS_A, FS_B = 47.988e6, 6.0e6
C = 299792458.0
DA, DB = C / (2 * FS_A), C / (2 * FS_B)
RATIO = DB / DA
DATES = ["NSR_20260101A", "NSR_20260113A", "NSR_20260125A", "NSR_20260206A", "NSR_20260218A"]
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4), (0, 3)]
MASTER = 0
NXB, NY = 160, 96
NXA = int(NXB * RATIO / 4) + 2          # A cells are 4 A px wide


def write_prm(path, fs, clock, nx):
    Path(path).write_text(
        f"near_range = 880000.0\nrng_samp_rate = {fs}\nPRF = 1520.0\nSC_clock_start = {clock}\n"
        f"radar_wavelength = 0.24\nnum_rng_bins = {nx}\nnum_patches = 1\nnum_valid_az = {NY * 8}\n"
        f"SC_vel = 7125.0\nSC_height = 755000.0\nearth_radius = 6370000.0\n")


def screen_fn(xA, yA, d):
    """Truth screen of date d (rad) as a function of A-pixel x and line y; master = 0."""
    ph = 0.7 * d
    f = (1.5 * np.sin(2 * np.pi * xA / 1800.0 + ph) * np.cos(2 * np.pi * yA / 1500.0 - ph)
         + 0.002 * (d + 1) * xA / 100 - 0.001 * d * yA / 100)
    f0 = (1.5 * np.sin(2 * np.pi * xA / 1800.0) * np.cos(2 * np.pi * yA / 1500.0))
    return f - (f0 if d == MASTER else 0)


class TestIonoCorrectStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        STORE.clear()
        CMDS.clear()
        cls.root = tempfile.mkdtemp()
        cls.old = os.getcwd()
        os.chdir(cls.root)
        rng = np.random.default_rng(0)
        os.makedirs("SLC")
        os.makedirs("SLC_B")
        os.makedirs("topo")
        Path("topo/trans.dat").write_bytes(b"x")
        clocks = [2026001.5, 2026013.5, 2026025.5, 2026037.5, 2026049.5]
        for d, stem in enumerate(DATES):
            write_prm(f"SLC/{stem}.PRM", FS_A, clocks[d], NXA * 4)
        write_prm(f"SLC_B/{DATES[MASTER][:-1]}B.PRM", FS_B, clocks[0], NXB)
        Path("config.txt").write_text(f"master_image = {DATES[MASTER]}\nthreshold_geocode = 0.1\n"
                                      "filter_wavelength = 200\n")
        Path("intf.in").write_text("\n".join(f"{DATES[i]}:{DATES[j]}" for i, j in PAIRS) + "\n")
        xB = (np.arange(NXB) + 0.5)                    # B pixel-centre coords (B px)
        y = 4.0 + 8.0 * np.arange(NY)
        xAB = xB * RATIO                               # A px
        cls.truth = np.stack([screen_fn(xAB[None, :], y[:, None], d) for d in range(len(DATES))])
        xA = 2.0 + 4.0 * np.arange(NXA)
        cls.xA, cls.y = xA, y
        cls.datum = rng.uniform(-3, 3, len(PAIRS))
        cls.sub = []
        cls.raw = {}
        for p, (i, j) in enumerate(PAIRS):
            sub = f"{int(clocks[i])}_{int(clocks[j])}"
            cls.sub.append(sub)
            I = cls.truth[i] - cls.truth[j] + cls.datum[p] + rng.normal(0, 0.25, (NY, NXB))
            sg = np.full((NY, NXB), 0.25)
            _write(f"iono_est/{sub}/I.grd", I.astype(np.float32), xB, y, node_offset=1)
            _write(f"iono_est/{sub}/sigma.grd", sg.astype(np.float32), xB, y, node_offset=1)
            # A-side step-4 products on the A grid
            unw = (np.add.outer(np.zeros(NY), 0.05 * xA) + 3.0 * rng.normal(size=(NY, NXA)) * 0.0
                   + 20.0 * (p + 1)).astype(np.float32)
            wph = ie.wrap(unw + rng.normal(0, 0.3, unw.shape)).astype(np.float32)
            _write(f"intf/{sub}/unwrap.grd", unw, xA, y, node_offset=1)
            _write(f"intf/{sub}/phasefilt.grd", wph, xA, y, node_offset=1)
            _write(f"intf/{sub}/phase.grd", wph, xA, y, node_offset=1)
            _write(f"intf/{sub}/corr.grd", np.full((NY, NXA), 0.5, np.float32), xA, y, node_offset=1)
            Path(f"intf/{sub}/{DATES[i]}.PRM").write_text("x")
            Path(f"intf/{sub}/unwrap.cpt").write_text("x")
            cls.raw[sub] = (unw.copy(), wph.copy())
        argv = sys.argv
        sys.argv = ["iono_correct_stack", "intf.in", "config.txt", "--block", "4", "--sigma-km", "0.3"]
        try:
            ics.main()
        finally:
            sys.argv = argv

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.old)
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_screens_recover_truth(self):
        xc = _read(f"iono_screens/{DATES[1]}_screen.grd")
        S1, x, y, info = xc
        self.assertEqual(info["node_offset"], 1)
        # coarse-grid centres are in A-pixel units: first centre = mean of first block * ratio
        self.assertAlmostEqual(x[0], np.mean(np.arange(4) + 0.5) * RATIO, places=6)
        errs = []
        for d in range(1, len(DATES)):
            S = _read(f"iono_screens/{DATES[d]}_screen.grd")[0]
            xB = (np.arange(S.shape[1] * 4) + 0.5)
            T = self.truth[d].reshape(NY // 4, 4, -1, 4).mean(axis=(1, 3))[:S.shape[0], :S.shape[1]]
            e = S - T
            e = e - e.mean()                    # datum is unobservable
            errs.append(np.sqrt(np.mean(e ** 2)))
        self.assertLess(max(errs), 0.15)

    def test_master_screen_zero_and_sigma(self):
        self.assertTrue(np.all(_read(f"iono_screens/{DATES[MASTER]}_screen.grd")[0] == 0))
        sg = _read(f"iono_screens/{DATES[2]}_sigma.grd")[0]
        self.assertTrue(np.isfinite(sg).all() and sg.max() < 1.0)

    def test_corrected_products(self):
        for p, (i, j) in enumerate(PAIRS):
            sub = self.sub[p]
            corr, x, y, _ = _read(f"intf_iono/{sub}/ph_iono.grd")
            unw_c = _read(f"intf_iono/{sub}/unwrap.grd")[0]
            unw0, wph0 = self.raw[sub]
            self.assertLess(np.abs(unw_c - (unw0 - corr)).max(), 1e-4)
            pf = _read(f"intf_iono/{sub}/phasefilt.grd")[0]
            self.assertLess(np.abs(ie.wrap(pf - ie.wrap(wph0 - corr))).max(), 1e-4)
            self.assertTrue(np.abs(pf).max() <= np.pi + 1e-6)
            # correction ~ truth difference up to the (unobservable) constant
            xb = (x / RATIO)                                    # B px
            T = screen_fn(x[None, :], y[:, None], i) - screen_fn(x[None, :], y[:, None], j)
            e = corr - T
            e = e - e.mean()
            self.assertLess(np.sqrt(np.mean(e ** 2)), 0.25, sub)
            # originals untouched
            np.testing.assert_array_equal(_read(f"intf/{sub}/unwrap.grd")[0], unw0)

    def test_closure_preserved_in_outputs(self):
        def corrected(p):
            return _read(f"intf_iono/{self.sub[p]}/unwrap.grd")[0]

        def raw(p):
            return self.raw[self.sub[p]][0]
        # triplets (0,1),(1,2),(0,2) and (1,2),(2,3),(1,3): indices into PAIRS
        for a, b, c in ((0, 1, 4), (1, 2, 5), (2, 3, 6)):
            d = (corrected(a) + corrected(b) - corrected(c)) - (raw(a) + raw(b) - raw(c))
            self.assertLess(np.abs(d).max(), 2e-4)

    def test_geocode_invoked(self):
        geo = [c for c in CMDS if c.startswith("geocode")]
        self.assertEqual(len(geo), len(PAIRS))
        self.assertTrue(all(c == "geocode 0.1" for c in geo))

    def test_cache_and_qc_written(self):
        self.assertTrue(os.path.isfile("iono_screens/coarse_pairs.npz"))
        lines = Path("iono_screens/pair_qc.txt").read_text().strip().splitlines()
        self.assertEqual(len(lines), len(PAIRS) + 1)


if __name__ == "__main__":
    unittest.main()
