#!/usr/bin/env python3
"""test_dem2topo_ra_trans_dat_cache — verifies the trans.dat reuse guard
added to utils/dem2topo_ra on 2026-09-23 at the user's request (the
`gmt grd2xyz ... | SAT_llt2rat_py ... > trans.dat` pipe took 15+ minutes
on real NISAR_CSAF data, and dem2topo_ra runs it unconditionally on every
intf_batch invocation -- even a re-run that only adds correct_iono=1 on
top of an already-completed step 4).

dem2topo_ra() is one large monolithic function (same situation as
utils/geocode -- see PATHWAY_FORWARD.md's 2026-09-23 ph_iono_ll entry):
calling it end-to-end needs real PRM fields, a real dem.grd, and further
downstream gmt calls (gmtconvert/blockmedian/surface/triangulate) this
sandbox can't run. So rather than claim integration coverage that doesn't
exist, this test isolates the exact reuse-decision expression that was
added (trans.dat exists AND is newer than both master.PRM and dem.grd) and
exercises it directly against real files with controlled mtimes -- the
same boolean logic dem2topo_ra now runs before deciding whether to
re-run the expensive pipe.

Run:
    python3 -m unittest test_dem2topo_ra_trans_dat_cache -v
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_UTILS_DIR = _HERE.parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from gmtsar_lib import check_file_report  # noqa: E402


def _reuse_trans_dat(prm_path: str, dem_path: str) -> bool:
    """The exact expression added to utils/dem2topo_ra, line ~for-line."""
    return (
        check_file_report("trans.dat")
        and os.path.getmtime("trans.dat") >= os.path.getmtime(prm_path)
        and os.path.getmtime("trans.dat") >= os.path.getmtime(dem_path)
    )


class TestTransDatReuseGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_dem2topo_ra_")
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _touch(self, name, mtime=None):
        open(name, "w").close()
        if mtime is not None:
            os.utime(name, (mtime, mtime))
        return name

    def test_no_trans_dat_recomputes(self):
        self._touch("master.PRM")
        self._touch("dem.grd")
        self.assertFalse(_reuse_trans_dat("master.PRM", "dem.grd"))

    def test_fresh_trans_dat_is_reused(self):
        self._touch("master.PRM", mtime=100)
        self._touch("dem.grd", mtime=100)
        self._touch("trans.dat", mtime=200)
        self.assertTrue(_reuse_trans_dat("master.PRM", "dem.grd"))

    def test_trans_dat_older_than_dem_recomputes(self):
        """DEM re-downloaded/changed after trans.dat was written -- must
        not silently reuse a trans.dat that no longer matches the DEM."""
        self._touch("master.PRM", mtime=100)
        self._touch("trans.dat", mtime=150)
        self._touch("dem.grd", mtime=200)  # dem.grd touched AFTER trans.dat
        self.assertFalse(_reuse_trans_dat("master.PRM", "dem.grd"))

    def test_trans_dat_older_than_prm_recomputes(self):
        """master.PRM re-aligned/updated after trans.dat was written --
        must not silently reuse a stale geometry LUT."""
        self._touch("dem.grd", mtime=100)
        self._touch("trans.dat", mtime=150)
        self._touch("master.PRM", mtime=200)  # PRM touched AFTER trans.dat
        self.assertFalse(_reuse_trans_dat("master.PRM", "dem.grd"))

    def test_exact_same_mtime_counts_as_fresh(self):
        """>= (not >) -- a trans.dat written in the same build step as the
        DEM/PRM (identical mtimes) should still be reused."""
        t = time.time()
        self._touch("master.PRM", mtime=t)
        self._touch("dem.grd", mtime=t)
        self._touch("trans.dat", mtime=t)
        self.assertTrue(_reuse_trans_dat("master.PRM", "dem.grd"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
