#!/usr/bin/env python3
"""test_intf_batch_iono — verification for intf_batch's correct_iono support.

Added 2026-09-21 at the user's explicit request: correct_iono didn't exist
anywhere in intf_batch before this (the batch/stack driver behind
batch_processing step 4 never even read the key from config). See
docs/PATHWAY_FORWARD.md's "Built 2026-09-21" entry for the full design
rationale, including a real, separate, pre-existing latent bug found (but
NOT fixed here, out of scope) in p2p_stages.py's
_iono_LH_fitoffset_and_resamp along the way.

No real GMT/GMTSAR/data available in the environment that wrote this test
(same limitation as the rest of this fork's iono-adjacent code — the
regression sweep runs with correct_iono=0 in every case config) — this
mocks gmtsar_lib.run/file_shuttle/grep_value/check_file_report/
replace_strings and verifies the exact COMMAND SEQUENCE intf_batch issues,
not real numerical output. The most important guarantee this test enforces
is #1 below: correct_iono unset must add literally zero new behavior,
since the user was explicit that future changes must not alter how
existing interferograms are produced.
"""
from __future__ import annotations

import importlib.machinery as _ilm
import importlib.util as _ilu
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_UTILS_DIR = _HERE.parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

import gmtsar_lib  # noqa: E402

_spec = _ilu.spec_from_loader(
    "intf_batch_mod",
    _ilm.SourceFileLoader("intf_batch_mod", str(_UTILS_DIR / "intf_batch")),
)
intf_batch = _ilu.module_from_spec(_spec)
sys.modules["intf_batch_mod"] = intf_batch
_spec.loader.exec_module(intf_batch)


class _FakeCompletedProcess:
    stdout = b"-100/100/-50/50"


class TestIntfBatchIono(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_run(cmd):
            self.calls.append(("run", cmd))
            class R:
                returncode = 0
            return R()

        def fake_file_shuttle(fn0, fn1, opt):
            self.calls.append(("file_shuttle", fn0, fn1, opt))

        def fake_grep_value(fn, s, i):
            self.calls.append(("grep_value", fn, s, i))
            if "SC_clock_start" in s:
                return 1000 + len(self.calls)
            if s == "high_wavelength":
                return 0.25
            if s == "low_wavelength":
                return 0.23
            return 0

        def fake_check_file_report(fn):
            self.calls.append(("check_file_report", fn))
            return os.path.isfile(fn)

        def fake_replace_strings(fn, s0, s1):
            self.calls.append(("replace_strings", fn, s0, s1))

        for mod in (gmtsar_lib, intf_batch):
            mod.run = fake_run
            mod.file_shuttle = fake_file_shuttle
            mod.grep_value = fake_grep_value
            mod.check_file_report = fake_check_file_report
            mod.replace_strings = fake_replace_strings

        self._orig_subprocess_run = subprocess.run

        def fake_subprocess_run(*a, **kw):
            self.calls.append(("subprocess.run", a, kw))
            return _FakeCompletedProcess()

        subprocess.run = fake_subprocess_run

    def tearDown(self):
        subprocess.run = self._orig_subprocess_run

    def _build_case(self, tmpdir, with_split_outputs=False):
        for d in ("raw", "SLC", "topo", "SLC_H", "SLC_L"):
            os.makedirs(os.path.join(tmpdir, d), exist_ok=True)
        for stem in ("MASTER", "REP1", "REP2"):
            open(os.path.join(tmpdir, "SLC", f"{stem}.PRM"), "w").close()
            open(os.path.join(tmpdir, "raw", f"{stem}.LED"), "w").close()
        open(os.path.join(tmpdir, "topo", "dem.grd"), "w").close()
        if with_split_outputs:
            open(os.path.join(tmpdir, "SLC_H", "MASTER.SLC"), "w").close()
            open(os.path.join(tmpdir, "SLC_L", "MASTER.SLC"), "w").close()
        intf_in = os.path.join(tmpdir, "intf.in")
        with open(intf_in, "w") as f:
            f.write("MASTER:REP1\nMASTER:REP2\n")
        return intf_in

    def _write_config(self, tmpdir, extra=""):
        path = os.path.join(tmpdir, "config.txt")
        with open(path, "w") as f:
            f.write(f"""\
proc_stage = 1
master_image = MASTER
filter_wavelength = 200
dec_factor = 2
topo_phase = 1
shift_topo = 0
threshold_snaphu = 0
threshold_geocode = 0
region_cut =
switch_land = 0
defomax = 0
near_interp = 0
mask_water = 0
{extra}
""")
        return path

    def _run_case(self, correct_iono_lines, with_split_outputs=False):
        tmpdir = tempfile.mkdtemp()
        try:
            intf_in = self._build_case(tmpdir, with_split_outputs)
            config = self._write_config(tmpdir, extra=correct_iono_lines)
            cwd0 = os.getcwd()
            os.chdir(tmpdir)
            try:
                sys.argv = ["intf_batch", "NSR_A", intf_in, config]
                self.calls.clear()
                intf_batch.intf_batch()
            finally:
                os.chdir(cwd0)
            return list(self.calls)
        finally:
            shutil.rmtree(tmpdir)

    def test_correct_iono_unset_adds_zero_new_calls(self):
        """The regression-safety guarantee: correct_iono defaulting to 0/
        unset must be byte-for-byte the pre-existing command sequence."""
        calls = self._run_case("")
        iono_related = [
            c for c in calls
            if any("iono" in str(x).lower() or "split_spectrum" in str(x).lower()
                   for x in c)
        ]
        self.assertEqual(iono_related, [], f"got unexpected iono calls: {iono_related}")
        run_cmds = [c[1] for c in calls if c[0] == "run"]
        self.assertTrue(any("intf MASTER.PRM REP1.PRM" in c for c in run_cmds))
        self.assertTrue(any(c.startswith("geocode") for c in run_cmds))

    def test_correct_iono_1_default_skip_est_splits_once_per_scene(self):
        """3 unique scenes across 2 pairs -> exactly 3 split_spectrum calls
        (not 4), 6 iono filter calls (2 pairs x 3 sides), no estimate call
        (iono_skip_est defaults to 1, matching pop_config's template)."""
        calls = self._run_case("correct_iono = 1\nrange_dec = 8\nazimuth_dec = 8\n")
        run_cmds = [c[1] for c in calls if c[0] == "run"]
        split_calls = [c for c in run_cmds if c.startswith("split_spectrum")]
        self.assertEqual(len(split_calls), 3, split_calls)
        estimate_calls = [c for c in run_cmds if c.startswith("estimate_ionospheric_phase")]
        self.assertEqual(estimate_calls, [])
        filter_500 = [c for c in run_cmds if c.startswith("filter") and " 500 " in c]
        self.assertEqual(len(filter_500), 6, filter_500)
        for c in filter_500:
            self.assertIn(" 500 2 8 8", c)

    def test_correct_iono_1_skip_est_0_runs_full_correction(self):
        calls = self._run_case(
            "correct_iono = 1\nrange_dec = 8\nazimuth_dec = 8\niono_skip_est = 0\n"
        )
        run_cmds = [c[1] for c in calls if c[0] == "run"]
        estimate_calls = [c for c in run_cmds if c.startswith("estimate_ionospheric_phase")]
        self.assertEqual(len(estimate_calls), 2, estimate_calls)
        snaphu_interp_calls = [c for c in run_cmds if c.startswith("snaphu_interp")]
        self.assertEqual(len(snaphu_interp_calls), 6)
        grdmath_calls = [c for c in run_cmds if "grdmath" in c and "phasefilt_non_corrected" in c]
        self.assertEqual(len(grdmath_calls), 2)

    def test_correct_iono_1_without_range_dec_fails_loud(self):
        tmpdir = tempfile.mkdtemp()
        try:
            intf_in = self._build_case(tmpdir)
            config = self._write_config(tmpdir, extra="correct_iono = 1\n")
            cwd0 = os.getcwd()
            os.chdir(tmpdir)
            try:
                sys.argv = ["intf_batch", "NSR_A", intf_in, config]
                with self.assertRaises(SystemExit) as ctx:
                    intf_batch.intf_batch()
                self.assertIn("range_dec and azimuth_dec", str(ctx.exception))
            finally:
                os.chdir(cwd0)
        finally:
            shutil.rmtree(tmpdir)

    def test_already_split_scene_is_skipped(self):
        calls = self._run_case(
            "correct_iono = 1\nrange_dec = 8\nazimuth_dec = 8\n",
            with_split_outputs=True,  # MASTER already split
        )
        run_cmds = [c[1] for c in calls if c[0] == "run"]
        split_calls = [c for c in run_cmds if c.startswith("split_spectrum")]
        self.assertEqual(len(split_calls), 2, split_calls)
        self.assertFalse(any("MASTER.PRM" in c for c in split_calls), split_calls)


if __name__ == "__main__":
    unittest.main()
