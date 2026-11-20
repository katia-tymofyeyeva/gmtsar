#!/usr/bin/env python3
"""nisar_stack_auto: end-to-end orchestration with stub commands on PATH (no GMT/GMTSAR/data).

The stubs create exactly the files each real step leaves behind, and record every call.
Checks: stage order and arguments, master_image from stem_data.in, inclusive thresholds
passed to select_pairs, config defaults, resume (a rerun runs nothing that is complete),
failure when a step leaves no output, and the disconnected-network stop."""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

UTILS = Path(__file__).resolve().parent.parent.parent / "utils"
SCRIPT = UTILS / "nisar_stack_auto"
DATES = ["20260101", "20260113", "20260125", "20260206"]
GRAN = [f"NISAR_L1_PR_RSLC_001_002_A_003_4005_DHDH_A_{d}T000000_{d}T000010_X05010_N_F_J_001" for d in DATES]

STUB = r'''#!/usr/bin/env python3
import os, sys
log = os.environ["STUB_LOG"]
open(log, "a").write(" ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]) + "\n")
name = os.path.basename(sys.argv[0])
fail = os.environ.get("STUB_FAIL", "")
def touch(p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True); open(p, "a").close()
if name == "pop_config":
    print("threshold_snaphu = 0\nthreshold_geocode = .10\nintf_skip_existing = 0\n# reuse comment\n")
elif name == "make_dem":
    touch("dem.grd")
elif name == "select_pairs":
    dt, db = float(sys.argv[2]), float(sys.argv[3])
    rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
    sel = [(a[0], b[0]) for a in rows for b in rows
           if float(a[2]) < float(b[2]) and float(b[2]) - float(a[2]) < dt and abs(float(a[4]) - float(b[4])) < db]
    open("intf.in", "w").write("".join(f"{x}:{y}\n" for x, y in sel))
elif name == "batch_processing":
    step = sys.argv[4]
    if step == fail:
        sys.exit(0)                      # like the real driver: inner failure only warns
    dates = os.environ["STUB_DATES"].split()
    stems = [f"NSR_{d}A" for d in dates]
    if step == "1":
        open("raw/stem_data.in", "w").write("\n".join(stems) + "\n")
        bperp = [0, 40, 400, 420]
        open("raw/baseline_table.dat", "w").write("".join(
            f"{s} 0 {12 * k} 0 {bperp[k]} 0 0\n" for k, s in enumerate(stems)))
        for s in stems:
            touch(f"raw/{s}.SLC"); touch(f"raw/{s[:-1]}B.SLC")
    elif step == "2":
        for k, s in enumerate(stems):
            touch(f"SLC/{s}.SLC")
            open(f"SLC/{s}.PRM", "w").write(f"SC_clock_start = {2026001 + 12 * k}.25\n")
            if k:
                touch(f"SLC/offsets/{s}_r.grd")
    elif step in ("4", "7"):
        cfg = sys.argv[5]
        method = [l.split()[2] for l in open(cfg) if l.startswith("iono_method")] or ["network"]
        d = "intf" if step == "4" else ("intf_iono_sbas" if method[0] == "sbas" else "intf_iono")
        f = "unwrap_ll.grd" if step == "4" else "unwrap.grd"
        print(f">>> 2026-11-20 12:00:00 | step {step} stub | [1/1] pair X: working", flush=True)
        for line in open(sys.argv[3]):
            r, p = line.strip().split(":")
            cid = lambda s: int(float(open(f"SLC/{s}.PRM").read().split()[2]))
            touch(f"{d}/{cid(r)}_{cid(p)}/{f}")
    elif step == "8":
        os.makedirs("closure_unwrap.grd", exist_ok=True)
'''


class T(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.bin = self.d / "bin"
        self.bin.mkdir()
        for n in ("pop_config", "batch_processing", "select_pairs", "make_dem", "gmt"):
            (self.bin / n).write_text(STUB)
            (self.bin / n).chmod(0o755)
        self.case = self.d / "case"
        (self.case / "raw").mkdir(parents=True)
        for g in GRAN:
            (self.case / "raw" / f"{g}.h5").write_text("x")
        self.log = self.d / "calls.txt"

    def run_it(self, *extra, fail="", code=0):
        env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", STUB_LOG=str(self.log),
                   STUB_DATES=" ".join(DATES), STUB_FAIL=fail)
        r = subprocess.run([sys.executable, str(SCRIPT), str(self.case / "raw"), GRAN[1] + ".h5",
                            "--max-days", "24", "--max-bperp", "100", "--region", "-118/-117/34/35", *extra],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(r.returncode, code, r.stdout[-3000:])
        return r.stdout

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_unused_dates_warn(self):
        out = self.run_it("--max-days", "12", "--max-bperp", "30")
        self.assertIn("2 date(s) in no pair", out)
        self.assertIn("the master NSR_20260101A is in no pair", out)

    def test_connected_run(self):
        # raise the baseline limit so every date joins one network
        env_extra = ["--set", "iono_method=sbas"]
        out = self.run_it(*env_extra, "--max-bperp", "500")
        c = self.calls()
        steps = [l.split()[4] for l in c if l.startswith("batch_processing")]
        self.assertEqual(steps, ["1", "2", "4", "7", "8"])
        self.assertTrue(all(l.split()[2] == GRAN[1] for l in c if l.startswith("batch_processing")))
        self.assertIn("make_dem -118 -117 34 35 1", c)
        sp = [l for l in c if l.startswith("select_pairs")][0].split()
        self.assertEqual((float(sp[2]), float(sp[3])), (24.001, 500.001))
        cfg = (self.case / "config.NSR_A.txt").read_text()
        self.assertIn("master_image = NSR_20260101A", cfg)          # first line of stem_data.in
        self.assertIn("threshold_snaphu = 0.1", cfg)
        self.assertIn("near_interp = 1", cfg)
        self.assertEqual(cfg.count("threshold_snaphu"), 1)              # replaced, not appended
        self.assertIn("intf_skip_existing = 1", cfg)
        self.assertIn("iono_method = sbas", cfg)
        self.assertEqual((self.case / "data.in").read_text().splitlines()[0], GRAN[1])
        self.assertTrue((self.case / "intf_iono_sbas").is_dir())
        self.assertIn("FINISHED", out)
        self.assertTrue((self.case / "nisar_stack_auto.status").read_text().startswith("stage 8/8 closure"))
        self.assertIn("FINISHED", (self.case / "nisar_stack_auto.status").read_text())
        logtxt = (self.case / "nisar_stack_auto.log").read_text()
        self.assertIn(">>> 2026-11-20 12:00:00 | step 4 stub | [1/1] pair X: working", logtxt)
        self.assertEqual(len([l for l in logtxt.splitlines() if "| nisar_stack_auto | stage" in l]), 8)
        # rerun: preproc/align/intf/dem skipped; only pairs, 7 and 8 run again
        self.log.unlink()
        self.run_it(*env_extra, "--max-bperp", "500")
        steps = [l.split()[4] for l in self.calls() if l.startswith("batch_processing")]
        self.assertEqual(steps, ["7", "8"])
        self.assertFalse(any(l.startswith(("make_dem", "pop_config")) for l in self.calls()))

    def test_disconnected_stops_before_intf(self):
        out = self.run_it("--max-days", "12", "--max-bperp", "50", code=1)
        self.assertIn("2 disconnected parts", out)
        self.assertNotIn("4", [l.split()[4] for l in self.calls() if l.startswith("batch_processing")])

    def test_step_without_output_stops(self):
        out = self.run_it("--max-bperp", "500", fail="2", code=1)
        self.assertIn("step 2 did not produce", out)
        st = (self.case / "nisar_stack_auto.status").read_text()
        self.assertTrue(st.startswith("stage 4/8 align") and "STOPPED" in st, st)

    def test_disconnected_allowed_without_iono(self):
        out = self.run_it("--max-days", "12", "--max-bperp", "50", "--no-iono")
        steps = [l.split()[4] for l in self.calls() if l.startswith("batch_processing")]
        self.assertEqual(steps, ["1", "2", "4", "8"])
        self.assertIn("WARNING: the network falls into 2", out)

    def test_region_with_negative_longitude_and_bad_region(self):
        self.run_it("--max-bperp", "500", "--stop-after", "dem")
        self.assertIn("make_dem -118 -117 34 35 1", self.calls())

    def test_stop_after(self):
        self.run_it("--max-bperp", "500", "--stop-after", "pairs")
        steps = [l.split()[4] for l in self.calls() if l.startswith("batch_processing")]
        self.assertEqual(steps, ["1", "2"])
        self.assertTrue((self.case / "raw" / "intf.in").is_file())
        self.assertIn("stopped after stage pairs", (self.case / "nisar_stack_auto.status").read_text())

    def test_set_overrides_auto_defaults(self):
        self.run_it("--max-bperp", "500", "--stop-after", "setup", "--set", "threshold_snaphu=0.3",
                    "--set", "near_interp=0")
        cfg = (self.case / "config.NSR_A.txt").read_text()
        self.assertIn("threshold_snaphu = 0.3", cfg)
        self.assertIn("near_interp = 0", cfg)


if __name__ == "__main__":
    unittest.main()
