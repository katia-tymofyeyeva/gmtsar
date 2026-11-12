"""batch_processing steps 7 (iono_workflow) and 8 (check_closure): command construction."""
import importlib.machinery
import importlib.util
import os
import sys

import pytest

UTILS = os.path.join(os.path.dirname(__file__), "..", "..", "utils")
sys.path.insert(0, UTILS)


def load():
    path = os.path.join(UTILS, "batch_processing")
    loader = importlib.machinery.SourceFileLoader("batch_processing_mod", path)
    spec = importlib.util.spec_from_loader("batch_processing_mod", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


CFG = """master_image = NSR_2026257A
closure_products  = unwrap.grd,los_ll.grd
closure_which     = iono
closure_max_loops = 7
closure_coh_min   = 0.3
closure_stride    = 3
"""


def run_step(tmp_path, monkeypatch, step, sat="NSR_A", cfg=CFG, dirs=("intf", "intf_iono")):
    mod = load()
    calls = []
    monkeypatch.setattr(mod, "run", lambda c, *a, **k: calls.append(c))
    monkeypatch.chdir(tmp_path)
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    (tmp_path / "intf.in").write_text("a:b\n")
    (tmp_path / "cfg.txt").write_text(cfg)
    monkeypatch.setattr(sys, "argv", ["batch_processing", sat, "NSR_2026257A", "intf.in", str(step), "cfg.txt"])
    mod.batch_processing()
    return calls


def test_step8_commands(tmp_path, monkeypatch):
    calls = run_step(tmp_path, monkeypatch, 8)
    assert calls == [
        "check_closure unwrap.grd --which iono --max-loops 7 --coh-min 0.3 --stride 3",
        "check_closure los_ll.grd --which iono --max-loops 7 --coh-min 0.3 --stride 3",
    ]


def test_step8_defaults(tmp_path, monkeypatch):
    calls = run_step(tmp_path, monkeypatch, 8, cfg="master_image = x\n")
    assert calls == ["check_closure unwrap.grd --which both --max-loops 20 --coh-min 0 --stride 2"]


def test_step8_bad_which(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_step(tmp_path, monkeypatch, 8, cfg="closure_which = bogus\n")


def test_step8_no_products_dirs(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_step(tmp_path, monkeypatch, 8, dirs=())


def test_step7_calls_workflow(tmp_path, monkeypatch):
    calls = run_step(tmp_path, monkeypatch, 7, dirs=("intf",))
    assert len(calls) == 1 and calls[0].startswith("iono_workflow ") and calls[0].endswith("cfg.txt")


def test_step7_requires_nisar_a(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_step(tmp_path, monkeypatch, 7, sat="S1_TOPS", dirs=("intf",))
