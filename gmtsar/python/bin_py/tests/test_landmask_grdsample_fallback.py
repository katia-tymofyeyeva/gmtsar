#!/usr/bin/env python3
"""test_landmask_grdsample_fallback -- verifies utils/landmask falls back to
the real `gmt grdsample` subprocess when the in-process grdsample port
can't handle a landmask_ra.grd that covers less than region_cut.

Real bug found 2026-09-29 on real NISAR_CSAF data: proj_ll2ra.csh's
radar-coordinate landmask_ra.grd can legitimately cover LESS than the
full nominal region_cut (edge/corner radar pixels that don't geocode
within the DEM's lon/lat extent) -- expected, and already handled a few
lines below in landmask ("if the landmask region is smaller than the
region_cut pad with NaN", verbatim from the original csh). The real gmt
grdsample binary tolerates resampling to a -R that overshoots the
input's actual coverage; the in-process port (gmt_grdsample_py) has a
hard PAD=2-cell limit and raised a bare RuntimeError ("BCR query index
left the padded grid") the moment a real scene's shortfall exceeded
that. Fixed by catching that RuntimeError in utils/landmask and falling
back to the real `gmt grdsample` subprocess -- the same code path
GMTSAR_GRDSAMPLE_PY=0 already exercises.

No real GMT available in this environment -- mocks _grdsample_inproc,
run, file_shuttle, check_file_report and verifies the exact fallback
behavior (which function gets called), not real numerical output.
"""
from __future__ import annotations

import importlib.machinery as _ilm
import importlib.util as _ilu
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_UTILS_DIR = _HERE.parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

_spec = _ilu.spec_from_loader(
    "landmask_mod",
    _ilm.SourceFileLoader("landmask_mod", str(_UTILS_DIR / "landmask")),
)
landmask_mod = _ilu.module_from_spec(_spec)
sys.modules["landmask_mod"] = landmask_mod
_spec.loader.exec_module(landmask_mod)


class TestLandmaskGrdsampleFallback(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_run(cmd):
            self.calls.append(("run", cmd))
            class R:
                returncode = 0
            return R()

        def fake_file_shuttle(fn0, fn1, opt):
            self.calls.append(("file_shuttle", fn0, fn1, opt))

        def fake_check_file_report(fn):
            self.calls.append(("check_file_report", fn))
            return False  # ~/.quiet doesn't exist -> V = '-V'

        def fake_delete(fn):
            self.calls.append(("delete", fn))

        landmask_mod.run = fake_run
        landmask_mod.file_shuttle = fake_file_shuttle
        landmask_mod.check_file_report = fake_check_file_report
        landmask_mod.delete = fake_delete
        landmask_mod._HAVE_GRDSAMPLE_PY_LM = True

    def _run_landmask(self, region_cut="0/53316/0/54720"):
        sys.argv = ["landmask", region_cut]
        landmask_mod.landmask()
        return list(self.calls)

    def test_inproc_success_skips_subprocess_fallback(self):
        """Happy path: in-process resample succeeds -> no gmt grdsample
        subprocess call for the resample step."""
        landmask_mod._grdsample_inproc = lambda *a, **kw: None
        calls = self._run_landmask()
        run_cmds = [c[1] for c in calls if c[0] == "run"]
        resample_subprocess_calls = [c for c in run_cmds if c.startswith("gmt grdsample landmask_ra.grd -Gtmp.grd -R")]
        self.assertEqual(resample_subprocess_calls, [], resample_subprocess_calls)

    def test_inproc_pad_overshoot_falls_back_to_subprocess(self):
        """The actual bug: in-process port raises RuntimeError (landmask_ra
        .grd covers less than region_cut by more than PAD=2 cells) ->
        landmask must fall back to the real `gmt grdsample` subprocess
        instead of propagating the exception and crashing the whole
        pipeline."""
        def _raise(*a, **kw):
            raise RuntimeError(
                "internal: BCR query index left the padded grid -- "
                "output region outside input by more than the PAD of 2 cells."
            )
        landmask_mod._grdsample_inproc = _raise
        calls = self._run_landmask(region_cut="0/53316/0/54720")
        run_cmds = [c[1] for c in calls if c[0] == "run"]
        resample_subprocess_calls = [
            c for c in run_cmds
            if c.startswith("gmt grdsample landmask_ra.grd -Gtmp.grd -R0/53316/0/54720")
        ]
        self.assertEqual(len(resample_subprocess_calls), 1, run_cmds)

    def test_grdsample_py_disabled_uses_subprocess_directly(self):
        """Regression guard: GMTSAR_GRDSAMPLE_PY=0 (or import failure) must
        still go straight to the subprocess, unchanged from before this
        fix."""
        landmask_mod._HAVE_GRDSAMPLE_PY_LM = False
        landmask_mod._grdsample_inproc = None
        calls = self._run_landmask(region_cut="0/100/0/200")
        run_cmds = [c[1] for c in calls if c[0] == "run"]
        resample_subprocess_calls = [
            c for c in run_cmds
            if c.startswith("gmt grdsample landmask_ra.grd -Gtmp.grd -R0/100/0/200")
        ]
        self.assertEqual(len(resample_subprocess_calls), 1, run_cmds)


if __name__ == "__main__":
    unittest.main()
