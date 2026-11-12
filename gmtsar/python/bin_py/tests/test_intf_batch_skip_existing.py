"""intf_batch: config intf_skip_existing=1 skips finished pairs; default rebuilds all."""
import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest

UTILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
sys.path.insert(0, UTILS)


def load():
    loader = importlib.machinery.SourceFileLoader("intf_batch_mod", os.path.join(UTILS, "intf_batch"))
    spec = importlib.util.spec_from_loader("intf_batch_mod", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


class T(unittest.TestCase):
    def setUp(self):
        self.old = os.getcwd()
        self.d = tempfile.mkdtemp()
        os.chdir(self.d)
        os.makedirs("intf/1_2")
        os.makedirs("SLC")
        open("intf.in", "w").write("a:b\nb:c\n")
        ids = {"a": 1, "b": 2, "c": 3}
        self.ids = ids

    def tearDown(self):
        os.chdir(self.old)

    def run_it(self, extra, final_name="unwrap_ll.grd", snaphu="0.1"):
        m = load()
        open("cfg", "w").write(f"master_image = a\nthreshold_snaphu = {snaphu}\n{extra}")
        if final_name:
            open(f"intf/1_2/{final_name}", "w").write("x")
        built, topo = [], []
        m.grep_value = lambda f, k, i: str(self.ids[os.path.basename(f)[0]])
        m._intf_one_pair = lambda ref, rep, *a, **k: built.append((ref, rep))
        m._topo_stage = lambda *a: topo.append(a)
        sys.argv = ["intf_batch", "NSR_A", "intf.in", "cfg"]
        m.intf_batch()
        return built, topo

    def test_default_rebuilds_all(self):
        built, topo = self.run_it("")
        self.assertEqual(built, [("a", "b"), ("b", "c")])
        self.assertEqual(topo[0][-1], 0)           # reuse_topo stays 0

    def test_skip_finished(self):
        built, topo = self.run_it("intf_skip_existing = 1\n")
        self.assertEqual(built, [("b", "c")])
        self.assertEqual(topo[0][-1], 1)           # implies reuse_topo

    def test_unfinished_not_skipped(self):
        built, _ = self.run_it("intf_skip_existing = 1\n", final_name="phasefilt.grd")
        self.assertEqual(built, [("a", "b"), ("b", "c")])

    def test_no_unwrap_uses_phasefilt_ll(self):
        built, _ = self.run_it("intf_skip_existing = 1\n", final_name="phasefilt_ll.grd", snaphu="0")
        self.assertEqual(built, [("b", "c")])


if __name__ == "__main__":
    unittest.main()
