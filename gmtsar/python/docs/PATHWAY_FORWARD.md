# Pathway forward — what's ported, what's not, and why

## Fixed 2026-09-24 (4): real heap-buffer-underflow in `split_spectrum.c`'s `cos_window()` -- a genuine, pre-existing C bug, not a Python port issue

After the earlier fixes, `split_spectrum` itself crashed on real
NISAR_CSAF data (`NSR_20260331A`) with `double free or corruption (out)`
(exit 134/SIGABRT) partway through writing lines. Unlike every other bug
found this week, this is in the **C binary itself**
(`gmtsar/split_spectrum.c`), not the Python framework.

Root cause: `cos_window()` builds a bandpass window into `filter[]`, a
heap array of exactly `N` (`nffti`) doubles, using three loops whose
start/end indices are derived from `nc`, `flat_nb`, `cos_nb` (all
functions of the sub-band center frequency and bandwidth relative to the
sampling rate). For some real parameter combinations -- confirmed on this
exact scene -- `nc - flat_nb - cos_nb - 1` comes out negative, so the
second loop starts writing at a negative index: `filter[-1]`,
`filter[-2]`, ... -- a **heap-buffer-underflow**, writing a few doubles
*before* the start of the `filterh`/`filterl` allocations. This doesn't
crash immediately; it corrupts the allocator's bookkeeping for whatever
chunk sits just before `filter[]` in memory, which only gets detected
later when *that* chunk is `free()`'d -- explaining why the crash
happened at the very end of the write loop, seemingly unrelated to
`cos_window()`, which ran first and returned normally.

Confirmed with a standalone AddressSanitizer harness (not the full GMTSAR
build, which needs `gmt.h`/`tiffio.h` unavailable in this sandbox) built
from `cos_window()`'s code verbatim: reproduced a real
`heap-buffer-overflow ... 8 bytes to the left of` the `filter[]`
allocation with `bc=50` (a tiny bandwidth relative to `fs`/`N`, which
drives `nc` down near 0). Confirmed the reproduction is genuine (not a
harness artifact) before writing the fix.

Fixed by clamping each loop's start index to `>= 0` and adding an `i < N`
guard to each loop's continuation condition -- eliminates the
out-of-bounds write for any input, while leaving every write for
already-in-bounds indices completely unchanged (verified: a realistic
in-bounds case still produces `filter[nc] == 1.0` exactly as before).

**Verification**: confirmed the original (unfixed) `cos_window()` code,
extracted verbatim into a scratch harness, reproduces a real
`heap-buffer-overflow ... 8 bytes to the left of` the `filter[]`
allocation under ASan with `bc=50` -- before writing the fix. Added
`gmtsar/tests/test_split_spectrum_cos_window.c`, a standalone (no
`gmt.h`/`tiffio.h`/GMTSAR-build dependency) ASan harness with the FIXED
`cos_window()` copied verbatim, covering the exact crash reproduction
(`bc=50`), `nc==0` exactly, a realistic in-bounds NISAR-like bandwidth
(regression check: `filter[nc] == 1.0` unchanged), and the `fs/3`
boundary case. Build & run: `gcc -fsanitize=address -g -O0
test_split_spectrum_cos_window.c -o /tmp/tcw -lm && /tmp/tcw` -- prints
`ALL CASES PASSED` and exits 0. Confirmed passing.

**This requires a rebuild, not just `git pull`** -- `split_spectrum` is a
compiled C binary. `install.py --rebuild` is sufficient and is the right
tool for this (simpler than a manual `cd gmtsar && make`): `do_build()`
runs `make`/`make install` from the repo root, and GMTSAR's Makefile is
recursive, so that one call recompiles anything under `gmtsar/` whose
source changed -- including `split_spectrum.c` -- via normal Make
dependency tracking, then re-stages the rebuilt binary into `<repo>/bin`.
```
python3 gmtsar/python/install.py --system conda --conda-env <env> --rebuild
```
Confirm the binary's mtime is newer than `split_spectrum.c` before
re-running `correct_iono=1`.

## Fixed 2026-09-24 (3): `cleanup topo` deleted `trans.dat` before the reuse guard ever got a chance to run

The user noticed `dem2topo_ra` still prints `no file trans.dat` on every
single run, even after the earlier-today caching fix. Traced it: `_topo_stage`
(intf_batch's stage-1 helper) unconditionally runs `run("cleanup topo")` as
its very first line, every time -- and `cleanup topo` deletes every file in
`topo/` except `dem.grd`, including `trans.dat` and `topo_ra.grd`. So the
file is already gone by the time `dem2topo_ra`'s reuse check ever looks
for it. This predates every change made today; it's the original design,
sensible for the old one-shot workflow, but it means the trans.dat caching
added earlier today was effectively dead code inside `intf_batch` itself
(it still helps `dem2topo_ra`'s other callers, like `merge_unwrap_geocode_tops`'s
own pattern, which don't clean topo/ first).

Per the user's standing instruction not to change how existing
interferograms are produced without opting in, this is a new config key,
default off: `reuse_topo` (default `0`) preserves the exact prior
behavior -- `cleanup topo` still runs every time. Setting `reuse_topo = 1`
skips that one command, letting `dem2topo_ra`'s own mtime check (still
requiring `trans.dat` to be newer than both `master.PRM` and `dem.grd`)
decide whether to reuse it.

**Verification**: added `test_reuse_topo_unset_still_runs_cleanup_topo`
(regression guard: unset must still run `cleanup topo`, byte-for-byte) and
`test_reuse_topo_1_skips_cleanup_topo` to `bin_py/tests/test_intf_batch_iono.py`.
All 10 tests in the file pass.

## Fixed 2026-09-24 (2): the self-heal check itself could crash with `IndexError` instead of healing

Immediately after the self-heal fix directly below, a fresh single-pair
`batch_processing` run hit a NEW crash in the fix's own validation code:

```
File "intf_batch", line 133, in _prm_has_valid_wavelength
    wl = grep_value(prm_path, "radar_wavelength", 3)
File "gmtsar_lib.py", line 211, in grep_value
    val = line.split()[i-1]
IndexError: list index out of range
```

Root cause: `gmtsar_lib.grep_value`'s `line.split()[i-1]` assumes the
matched line has at least `i` whitespace-separated tokens. A PRM left
behind by the *original*, pre-2026-09-23 version of the `correct_iono`
bug can have a line like `radar_wavelength = ` -- key and `=` present,
but no value token at all (not merely a missing line, which
`grep_value` already handled fine by leaving `val` at `""`). `line.split()`
on that line is `["radar_wavelength", "="]` -- only 2 tokens -- so
`[i-1]` with `i=3` raises `IndexError` instead of returning `""`.
`_prm_has_valid_wavelength` called `grep_value` expecting it to return a
value or `""`, not raise -- so the exception propagated straight out and
killed `intf_batch` entirely, which is the opposite of what a self-heal
check is for: the one file it's specifically meant to be checked *because
it might be broken* is the one that crashed it.

Fixed by wrapping both `grep_value` calls in `_iono_split_scene`
(`_prm_has_valid_wavelength`'s own check, and the `high_wavelength`/
`low_wavelength` read from `split_spectrum`'s output) in
`try/except (IndexError, ValueError, OSError)`, treating any failure to
read a usable value the same as reading an explicitly blank one: "not
valid, re-split" for the cache check, and the existing 2026-09-23
fail-loud `RuntimeError` for the `split_spectrum` output read.

**Not fixed** (deliberately out of scope): `gmtsar_lib.grep_value`'s
`line.split()[i-1]` is a general landmine used throughout this codebase
wherever a PRM/params file might have a key with a missing value --
fixing every call site is a much larger, separate audit. This entry only
hardens the two call sites this feature itself introduced.

**Verification**: added `test_prm_with_blank_wavelength_line_does_not_crash`
to `bin_py/tests/test_intf_batch_iono.py` -- unlike the other tests in
that file, it uses the REAL `gmtsar_lib.grep_value` (captured before
`setUp()` overwrites it with a mock) against a real on-disk PRM
containing `"radar_wavelength = \n"`, to reproduce the exact crash rather
than a stand-in that might not match. Confirmed the crash is real and
reproducible standalone (`grep_value` on that exact line raises
`IndexError`) before writing the fix. All 8 tests in the file pass.

## Fixed 2026-09-24: `_iono_split_scene`'s "already split" cache could never self-heal from the 2026-09-23 bug

The 2026-09-23 fix (below) stopped a failed `split_spectrum` from writing
a blank `radar_wavelength` silently -- but only for *scenes split after
that fix landed*. `_iono_split_scene`'s idempotency check only looked for
`SLC_H/<stem>.SLC` and `SLC_L/<stem>.SLC` -- not whether their PRMs
actually carry a usable `radar_wavelength`. Any scene already split by a
run from *before* the fix has those `.SLC` files sitting on disk next to
a PRM with a blank `radar_wavelength`, so the check trusted it as
"already split" and skipped re-splitting it -- forever. Confirmed on real
data: the exact same `ZeroDivisionError` recurred on 2026-09-24 for
`NSR_20251225A`/`NSR_20260106A`, a *different* scene pair than the
original 2026-09-23 report, well after that fix was committed and pulled.

Fixed by having `_iono_split_scene` also validate both PRMs
(`_prm_has_valid_wavelength`: file exists and `radar_wavelength` parses as
nonzero) before trusting the cache; if either is missing or still blank,
it re-splits and overwrites, self-healing instead of failing the same way
forever.

**If you've already hit this**: any scene that failed with this
`ZeroDivisionError` before pulling this fix has a broken `SLC_H`/`SLC_L`
PRM sitting on disk right now. You do NOT need to manually delete
anything -- the next `intf_batch` run will detect the invalid
`radar_wavelength` and re-split that scene automatically.

**Verification**: added `test_stale_broken_split_scene_is_repaired` to
`bin_py/tests/test_intf_batch_iono.py` (simulates the pre-fix broken
on-disk state -- `.SLC` present, `radar_wavelength` invalid -- and asserts
`_iono_split_scene` re-runs `split_spectrum` instead of skipping). All 7
tests in that file pass.

## Built 2026-09-23: `dem2topo_ra` reuses an existing `trans.dat` instead of always recomputing it

The user flagged that `gmt grd2xyz --FORMAT_FLOAT_OUT=%lf dem.grd -s |
SAT_llt2rat_py master.PRM 0 -bod > trans.dat` took 15+ minutes on real
NISAR_CSAF data, and `intf_batch` calls `dem2topo_ra` unconditionally at
the top of every run (before the pairs loop) -- so simply re-running step
4 to pick up an unrelated change (e.g. adding `correct_iono=1`) redoes
this every time, even though `trans.dat` hasn't actually changed.

There's already precedent for skipping this exact step elsewhere in the
codebase: `merge_unwrap_geocode_tops` does `if not
check_file_report("trans.dat"): ...`. But a bare existence check isn't
enough here -- `trans.dat` is a pure function of `master.PRM` (orbit/
geometry fields) and `dem.grd`; if either changes (a new DEM, a re-aligned
master) after `trans.dat` was written, reusing it silently produces a
wrong `topo_ra.grd` with no error at all -- exactly the class of quiet-
wrong-answer bug this whole project has been chasing all session, just
pointed the other direction (trusting a stale cache instead of missing a
file). So `dem2topo_ra` now also requires `trans.dat`'s mtime to be `>=`
both `master.PRM`'s and `dem.grd`'s before reusing it; otherwise it
recomputes exactly as before. Deleting `topo/trans.dat` by hand still
forces a recompute at any time.

**Verification**: `python3 -m py_compile utils/dem2topo_ra` passes.
`dem2topo_ra()` is one large monolithic function (needs real PRM fields,
a real `dem.grd`, and further real `gmt` calls this sandbox can't run --
same situation as `geocode`'s `ph_iono_ll.grd` change below), so rather
than claim integration coverage that doesn't exist, added
`bin_py/tests/test_dem2topo_ra_trans_dat_cache.py`, which isolates the
exact reuse-decision expression added and exercises it against real files
with controlled mtimes: no `trans.dat` -> recompute, fresh `trans.dat` ->
reused, `trans.dat` older than `dem.grd` -> recompute, `trans.dat` older
than `master.PRM` -> recompute, identical mtimes -> reused. All 5 pass.

## Fixed 2026-09-23: `correct_iono` silently wrote a blank `radar_wavelength`, surfacing as a `ZeroDivisionError` ~15 minutes later inside `filter`

Real failure from the user's first full `correct_iono=1` run on real
NISAR_CSAF data (pair NSR_20260331A/NSR_20260412A): `intf` for the
`intf_h`/`intf_l` sub-bands "succeeded" (rc=0 — the wrapper script always
exits 0 regardless of what happened inside it, same tolerance pattern as
everywhere else in this codebase), but never actually wrote
`real.grd`/`imag.grd`/`phase.grd`. ~15 minutes later, `filter` crashed with
`FileNotFoundError: ... 'realfilt.grd'` — nowhere near the real cause.

Root cause: `_iono_split_scene` (added 2026-09-21, see below) does

```python
wl = grep_value(params_path, key, 3)
...
replace_strings(f"{stem}.PRM", "wavelength", f"radar_wavelength = {wl}")
```

`grep_value` returns `""` (never raises) if `split_spectrum`'s redirected
stdout (`params_{stem}`) doesn't contain a `high_wavelength`/
`low_wavelength` line — which is exactly what happened for this scene
pair. `replace_strings` then wrote `radar_wavelength = ` (empty value) into
the split-band PRM. `phasediff_py`'s `_parse_prm` sees the empty value,
skips the key (`if not val: continue`), and `p["lambda"]` stays at its
`0.0` default — so `cnst = -4.0 * PI / p2["lambda"]` inside `phasediff`
divides by zero, `phasediff_py` exits 1, `run()` swallows it with a WARN,
and `intf`'s wrapper script "finishes" anyway with nothing written. The
actual upstream question — why `split_spectrum` didn't produce a usable
`high_wavelength`/`low_wavelength` line for this scene — is still open;
this fix stops the failure from masquerading as a `filter`/`conv` bug 15
minutes downstream, but doesn't explain `split_spectrum`'s own behavior
here. Next time this fires, the new error message points straight at the
scene and the exact command to re-run manually to see why.

Fixed by validating `wl` in `_iono_split_scene` before writing it into the
PRM: if it isn't a nonzero number, raise immediately with the scene name,
the `params_path`, and the exact `split_spectrum` command to re-run by
hand — instead of writing a broken PRM that fails silently, far away, much
later. Consistent with this project's no-silent-fallback rule.

**Verification**: added `test_split_scene_empty_wavelength_fails_loud` to
`bin_py/tests/test_intf_batch_iono.py` (mocks `grep_value` to return `""`,
asserts `_iono_split_scene` raises `RuntimeError` naming `high_wavelength`
and `split_spectrum`). All 6 tests in that file pass
(`python3 -m unittest test_intf_batch_iono`).

## Built 2026-09-23: `geocode` now projects `ph_iono.grd` to `ph_iono_ll.grd`

Real gap found by the user while inspecting the first correct_iono run
(see the 2026-09-21 entry below): `geocode` projects a fixed list of
radar-geometry grids to lon/lat (`corr.grd`, `phasefilt.grd`, `unwrap.grd`,
...) but never touched `ph_iono.grd` (the ionospheric phase screen
`intf_batch`'s iono block produces) — so the screen only ever existed in
radar coordinates, with no geocoded counterpart to actually look at
geographically. Fixed by adding one more guarded block, following the
exact pattern already used for every other optional grid in this function
(`if check_file_report('X.grd'): _project(...)`) — `ph_iono.grd` only
exists for a pair processed with `correct_iono=1`, so a workflow run
without ionospheric correction hits `check_file_report`'s `False` branch
and this is a no-op, identical to before. Produces `ph_iono_ll.grd`.

**Verification note**: `python3 -m py_compile utils/geocode` passes. Did
NOT add a dedicated mocked test for this one line — `geocode()` is one
large monolithic function that reads many other grids unconditionally
before reaching this point, so exercising it end-to-end would require
mocking most of the function's internals for a change that is a direct,
structural copy of four already-adjacent guarded blocks (`unwrap.grd`,
`unwrap_mask.grd`, `xphase_mask.grd`, `phasefilt_mask.grd`) doing exactly
the same `check_file_report` + `_project` + `grdedit` sequence. Flagging
this honestly rather than claiming coverage that doesn't exist — if this
turns out wrong, the four neighboring blocks are the reference to compare
against.

## Built 2026-09-21: `correct_iono` (split-spectrum ionospheric correction) wired into `intf_batch`

**New feature, at the user's explicit request — NOT a port of existing
tested code, and NOT yet verified against real data.** Before this,
`intf_batch` (the batch/stack driver behind `batch_processing` step 4)
never read `correct_iono` from config at all — setting it had zero effect.
Ionospheric correction only existed in the single-pair `p2p_processing`
path (`p2p_stages.py`'s `P2P4MakeFilterInterferograms`), which the user
isn't using for their NISAR_CSAF stack.

**What it does**: split-spectrum ionospheric phase estimation (Gomba et al.
2016, Fattahi et al. 2017 filtering) — the same algorithm `p2p_stages.py`
already implements, not a different method. `intf_batch` gained two new
functions: `_iono_split_scene` (splits one already-aligned scene's SLC
into high/low sub-bands via `split_spectrum`, once per unique scene
appearing in `intf.in` — idempotent, skips a scene already split) and
`_iono_one_pair` (per pair: intf+filter each of high/low/original,
optionally unwrap and call `estimate_ionospheric_phase` to produce a
corrected `phasefilt.grd`). Gated end-to-end by `correct_iono` (default 0
— every line of this feature is dead code unless a config explicitly sets
`correct_iono = 1`); further gated by `iono_skip_est` (default 1, matching
the existing project-wide default in `pop_config`'s template — i.e. even
with `correct_iono=1`, the unwrap+correction step is skipped by default,
only the raw high/low/orig `phasefilt.grd` screens are produced until the
user also sets `iono_skip_est = 0`). `correct_iono=1` additionally requires
`range_dec`/`azimuth_dec` in config (used to decimate the iono-specific
filter step) — fails loud with a clear message if either is missing,
rather than guessing a default (neither has a universally sensible one).

**A real design simplification, not a straight port**: `p2p_stages.py`'s
iono path runs `split_spectrum` mid-alignment, before the final `resamp_py`
swap — so it has to re-derive a resample step for the high/low sub-bands
afterward, via `_iono_LH_fitoffset_and_resamp`. While investigating that
function to decide whether to reuse it, found a real, separate, pre-existing
latent bug in it (not touched by this change, not exercised by anything):
it symlinks a frequency file under one name (`freq_alos2.dat` for most
SATs, including a nonexistent one for NSR) but then unconditionally reads
a different hardcoded filename (`freq_xcorr.dat`) — broken for any SAT
outside the small `_SAT_RAW_INPUT` group. Rather than depend on (or fix)
that function, `intf_batch`'s version sidesteps the whole problem: because
`intf_batch` only ever runs after `align_batch`/`align_batch_nsr` has
*already* produced a fully-aligned, master-conforming `SLC/<stem>.SLC`,
`split_spectrum`'s high/low outputs inherit that same alignment directly
(splitting the range spectrum doesn't move any pixels) — no resample step
needed at all. This was confirmed as sound reasoning with the user (who
has NISAR ionospheric-correction domain expertise) before implementing,
not assumed unilaterally.

**Explicitly out of scope for this round, flagged by the user**: NISAR
actually acquires two REAL, physically-separate frequency bands (A and
B) — using the real Frequency-B interferogram as the "other band" for
ionospheric estimation, instead of a synthetic `split_spectrum` of one
band, would be a more scientifically appropriate method for NISAR
specifically. Checked the actual upstream reference this fork is ported
from (`gmtsar/csh/p2p_processing_nsr.csh`, itself a very recent,
provisional addition marked "needs to be integrated back into the generic
p2p_processing.csh") — it does NOT implement real dual-frequency
correction either; it uses the same generic synthetic split-spectrum
method for NSR as every other single-band sensor. So there is no existing
implementation anywhere in this codebase (Python or csh) to fall back on
or port. The user is going to look for a GMTSAR variant that does this
properly and follow up — worth revisiting once found, per their explicit
request to keep this in mind as a future option.

**Verified**: `python3 -m py_compile utils/intf_batch` clean. Standalone
mock test (no real GMT/GMTSAR/data available in the environment that made
this change — same limitation as every other fix this session) covering:
(1) `correct_iono` unset produces literally zero new calls versus the
pre-existing behavior — the regression-safety guarantee the user explicitly
asked for when scoping this work; (2) `correct_iono=1` with the default
`iono_skip_est=1` splits each of 3 unique scenes across 2 pairs exactly
once (not once per pair), builds high/low/orig interferograms per pair,
and does not call `estimate_ionospheric_phase`; (3) `correct_iono=1,
iono_skip_est=0` does call `estimate_ionospheric_phase` once per pair and
composes a corrected `phasefilt.grd`; (4) `correct_iono=1` without
`range_dec`/`azimuth_dec` fails loud with a clear message rather than
guessing; (5) a scene already split (an existing `SLC_H`/`SLC_L` `.SLC`)
is correctly skipped on a re-run. **None of this has been run against
real NISAR data or real GMT** — the user is running it against their
actual NISAR_CSAF stack next and will inspect the resulting phase screens
directly; if the reasoning above about not needing a resample step turns
out to be wrong, or the results look off, this is the first place to
revisit.

## Fixed 2026-09-18: `install.py`'s `locate_conda_env` guessed a fixed path instead of asking conda

**Real bug found via an actual remote `install.py --system conda --conda-env
gmtsar --rebuild` re-run** (a JupyterHub-style host, not a fixture): it
failed with `ERROR: conda create exited 0 but /opt/conda/envs/gmtsar still
doesn't exist -- check the conda output above` — even though the preceding
`conda create -n gmtsar -y -c conda-forge gmt=6.4 ...` line logged `done in
66.747s (rc=0)`, i.e. conda really did spend a minute genuinely installing
packages, it just didn't put them where the script assumed.

Root cause: `locate_conda_env()`'s pre-create existence check
(`_find_existing_conda_env`) and its post-create success check both only
ever looked at a fixed `conda_base/envs/<name>` path (`conda_base` resolved
from `$CONDA_EXE`/`which conda`/a short list of common install roots). On
this host, conda's `envs_dirs` config redirects new envs somewhere other
than `conda_base/envs` — a common pattern where the base conda install
(`/opt/conda`) is root-owned/read-only, so per-user envs live under the
user's own home directory instead (this project's own earlier netCDF4 fix,
2026-09-xx, already pointed at exactly such a path,
`/home/jovyan/.local/envs/gmtsar`, for the same reason). `conda create -n
gmtsar` has no `--prefix`/`-r` override here, so it silently honors
`envs_dirs` and creates the env exactly where configured — just not at the
one fixed path the script checked.

This is the identical bug class already found and fixed for the Windows
code path on 2026-07-23 (`_windows_conda_env_paths`, using `conda env list
--json` as the authoritative source instead of directory-guessing) — that
fix was never mirrored onto POSIX, so the same failure mode was still live
here.

**Fixed**: added `_conda_env_list_json()` (POSIX port of
`_windows_conda_env_paths`' approach) and wired it into `locate_conda_env`
as a fallback both before deciding a fresh `create` is needed and after
`create` returns, before declaring failure — so a genuinely-existing env
under a non-default `envs_dirs` location is found either way, and the
error message only fires when the env truly doesn't exist anywhere `conda`
itself reports. **Verified**: a standalone test (fake `conda` executable
whose `env list --json` reports an env living outside `conda_base/envs`,
and whose `create` exits 0 without ever populating `conda_base/envs/<name>`
— reproducing the real host's exact behavior) confirms `locate_conda_env`
now returns the real env path instead of erroring; a second test confirms
a genuinely-nonexistent env (empty `env list --json`, still-missing
directory after create) still fails loudly, with an updated message noting
both checks were tried.

**Severity correction, same day, once re-run against the real remote host:**
this was NOT merely a wrong error message — the failed `conda create -n
gmtsar -y ...` call that produced the original error had already silently
**wiped and recreated** the user's real `gmtsar` env in place, destroying
everything installed into it beyond this fork's bare bootstrap package list
(h5py, numba, and netCDF4, all installed by hand over the course of this
session, were gone; `import h5py` started failing in scripts that had
worked minutes earlier). Root mechanism, confirmed against conda's own
documented behavior (see
https://github.com/conda/conda/issues/10432): `conda create -n <name>`,
when `<name>` already exists ANYWHERE in conda's own environment registry
(not just under the fixed path this script was checking), prints `WARNING:
A conda environment already exists at '<path>' / Remove existing
environment (y/[n])?` — and `-y` (which `install.py`'s create call always
passes) auto-confirms that prompt too, not just package-plan confirmations.
So the pre-fix code's blind-guess existence check didn't just fail to
*find* the real env — by failing to find it, it walked straight into
conda's own destroy-and-recreate path for a same-named env it didn't know
it was colliding with. The fix above (checking `conda env list --json`
*before* ever calling `create`) closes this for good going forward — the
real env is now found and returned before the destructive `create -y` call
is ever reached — but it cannot undo damage from a run that predates the
fix. **Recovery for anyone hitting this**: re-run `install.py --system
conda --conda-env <env>` *without* `--rebuild` (so `do_python_deps()` runs
again) to reinstall `requirements.txt` into the recreated env.
## Built 2026-08-09: NISAR batch/stack processing — `pre_proc_batch_nsr`, `align_batch_nsr`

**Prototype, NOT yet validated against a real 3+ scene stack** (per Rule 13,
recorded here as its own state rather than folded into "done"). Before this,
NISAR only had the single-pair path: `p2p_processing NSR_A`/`NSR_B` +
`pre_proc_nsr`, proven end-to-end on the real 2-scene `NISAR_Ethiopia` case
(see the "Attempted 2026-07-12: SAR sensor preprocessors" entry below).
`pre_proc_batch`/`intf_batch`'s own `_SUPPORTED_SATS` explicitly excluded
NSR — there was no way to align/interferogram more than one pair.

Two new scripts (`gmtsar/python/utils/pre_proc_batch_nsr`,
`.../align_batch_nsr`) generalize the proven single-pair recipe to a stack,
verbatim — they do NOT reuse generic `align_batch`'s algorithm (DEM-based
`SAT_llt2rat` ray-tracing + 3rd-order fit), because that's not what NISAR's
validated path actually does. Instead they replay
`p2p_stages.py`'s exact NSR branch (`P2P2FocusAlign`, `_xcorr_and_fitoffset`,
`_resamp_and_swap`: amp-based `xcorr_py` 40×40 grid + `fitoffset_ra`
10,10-order polynomial → `r.grd`/`a.grd` → `resamp_py` factor 5) in a loop
against one master. `baseline_table` and `select_pairs` needed no changes —
both were already NISAR-aware (baseline_table's SC_identity==5/year>=2013
epoch branch predates this work). `intf_batch` gained an NSR branch in its
OFFSET_TOPO step (dynamic decimation via `topo_ra.grd`'s x_inc, mirroring
`p2p_stages._offset_topo_shift`, since NISAR has no fixed decimation
constant the way ALOS/TSX do) plus a fail-loud `else` for any future SAT
that hits that branch with no rule — a latent Rule-1 gap in the pre-existing
code, fixed as a side effect. `batch_processing`'s dispatcher now routes
step 1/2 to the NSR scripts for `SAT in (NSR_A, NSR_B)`.

**What's real vs. unverified:**
- Every individual command in the alignment loop (`SAT_baseline_py`,
  `slc2amp`, `xcorr_py`, `fitoffset_ra`, `resamp_py`) is the exact command
  the proven single-pair path already runs — not reimplemented, only
  looped. `pre_proc_nsr` itself is unchanged (already wired ON,
  byte-identical to C, see below).
- **Not yet run end-to-end against real data.** All Python files
  `py_compile` clean; no case in `tests/cases.py` currently exercises a
  3+ scene NISAR stack (`NISAR_Ethiopia` is 2 scenes = 1 pair, which
  doesn't exercise the *batch* loop logic — a master + exactly one repeat
  is a degenerate stack). The user has additional NISAR granules on
  another machine; a real run there is the next step, not yet done here.
- One unvalidated-but-flagged micro-optimization opportunity: `align_batch_nsr`
  recomputes `amp-<master>.grd` on every loop iteration (verbatim with the
  single-pair recipe, which has no batch context to reuse it across calls).
  Safe-looking, not hoisted out of the loop until a real stack run confirms
  parity — see the comment in `_align_one`.

**TODO before this can be called "wired ON" per Rule 13:** a real
multi-scene NISAR run (ideally added as a new `tests/cases.py` fixture
distinct from `NISAR_Ethiopia`, since that tarball only has 2 scenes),
a parity/regression test per Rule 8, and a `master_image` auto-population
step in `pop_config`/`pre_proc_batch_nsr` so the manual "set master_image
to the derived stem before step 4" requirement isn't a footgun.

**Real bug found 2026-09-01, via an actual remote run against real
NISAR_CSAF data (not synthetic):** `pre_proc_nsr`'s `_make_slc_nsr()`
passed `region_cut` through to `make_slc_nsr_py`/the C binary
unconditionally, including the literal string `"-999"` — pop_config's
own "not set" sentinel (same convention as `earth_radius`/`near_range`/
`fd1`, see pop_config's template). Both `make_slc_nsr_py.make_slc_nsr()`
(called with `region_cut=None`) and the C binary (called with only 4
positional args) already correctly expand to the full scene internally
when no crop is given — the bug was only in never routing the sentinel
to that already-working path, so `get_range("-999", ...)` got called and
raised (can't split "-999" into 4 slash-separated ints). **Never caught
by the NISAR_Ethiopia parity test**, because that test's bundled config
always sets a real `region_cut` — this is a real, exactly-Rule-9 case of
"passing on synthetic/curated inputs isn't the same as passing on a real
unedited default." Only surfaced because `pre_proc_batch_nsr` got run
against pop_config's unedited output. Fixed in `pre_proc_nsr` (both
dispatch branches now treat `region_cut in (None, "", "-999")` as "full
scene"). **No regression test added yet** — this fork's `h5py`/real
NISAR fixture isn't available in the environment that made this fix;
flagging per Rule 8 rather than silently calling it done.

**Second real bug found 2026-09-01, same remote NISAR_CSAF run, immediately
after the fix above:** `pop_config` wrote its generated config directly to
a hardcoded `./config.py` via `open('config.py', 'w')`, ignoring stdout
entirely — not NSR-specific, affects every SAT. Every existing caller
happened to redirect stdout to a file literally named `config.py` (this
script's own usage docs say `pop_config SAT > config.py`;
`tests/case_runner.py`'s `_stage_config` does the same), so it "worked"
only by coincidence: the real content came from the internal write, and
the shell's `>` redirect silently captured nothing. `batch_processing`'s
own auto-generate-config step — used whenever the caller omits the
trailing `[config]` argument, e.g. `batch_processing NSR_A <master>
data.in 1` with no 5th arg — redirects to `config.<SAT>.txt` instead, a
different filename, which exposed the bug: `config.<SAT>.txt` came out
empty, and `_get_config()` (in `pre_proc_nsr`) read `""` for
`SLC_factor`, crashing on `float('')`. Root-caused by tracing exactly
where `batch_processing`'s auto-generated config filename diverges from
`pop_config`'s hardcoded write target, then confirmed no other caller
relied on the internal-write side effect. Fixed by having `pop_config`
actually write to stdout (`sys.stdout.write(render(sat))`) instead of
opening `config.py` itself — verified manually: `pop_config NSR_A >
config.NSR_A.txt` now contains the real `SLC_factor`/`region_cut` values
and no stray `config.py` is created. **No regression test added yet** —
same environment gap as above (this fork's test harness needs a
subprocess/stdout-capture check added to `tests/`, not yet written).

**Memory blowup found 2026-09-02, same remote NISAR_CSAF run, once past the
two bugs above:** `make_slc_nsr_py.py`'s `write_slc_hdf5()` was extremely
memory-hungry writing out a full, uncropped scene (the normal case now that
`region_cut = -999` is correctly honored as "full scene" rather than
crashing). Root cause: it materialized the *entire* cropped region as a
stack of full-size temporaries at once — complex64 region (8 B/px), float64
real/imag copies (16 B/px), float32 real/imag (8 B/px),
`f32_to_i16_batch`'s own internal finite/hi/lo/trunc/out temporaries
(several more full-size float32/bool arrays), int16 outputs (4 B/px), and a
final interleaved int16 array (4 B/px) duplicating the whole output right
before the single `fp.write()` — many times the 4 B/px final output size,
all held in RAM simultaneously, for a real NISAR scene's full extent.
Fixed by chunking along rows: read/convert/quantize/write one horizontal
stripe of the HDF5 dataset at a time (ordinary h5py slicing, which already
only pulls the requested sub-region off disk) instead of the whole region
in one shot. Column cropping (the multiple-of-4 width adjustment) is
computed once up front exactly as before and applies unchanged to every
stripe; only the row range is chunked. Diagnostic counters
(sat_hi/sat_lo/zero_conv, sigma) are now accumulated across chunks as
exact int64 sums instead of computed once over the full array — integer
summation is associative, so this doesn't change the result. Default chunk
size is 4096 rows (`_DEFAULT_ROWS_PER_CHUNK`), overridable via a new
`rows_per_chunk` kwarg on `write_slc_hdf5()`. **Verified, not just
reasoned through:** no h5py in the environment that made this fix, so
tested with a standalone numpy-array stand-in for the h5py Dataset (slicing
behavior is identical for this purpose) — chunked vs. single-shot produced
byte-identical `.SLC` output and identical diagnostic counters across 7
chunk sizes (1, 3, 4, 7, 16, 64, 1000 rows) against a synthetic scene sized
to NOT be a multiple of any of them, including NaN/±Inf/saturating edge-case
pixels to exercise every branch of `f32_to_i16_batch`. **Still flagged per
Rule 8**: this is not the same as a real-HDF5-file regression test against
actual NISAR data, which still needs h5py + a real fixture to add properly.

**Three bugs found 2026-09-02, in `baseline_table`, while investigating a
user report of "only 4 columns" in `raw/baseline_table.dat`** (expected 7:
`ORB ST0 YDAY Bpl Bperp xshift yshift`):

1. **ORB always empty for NISAR.** `_orb_id()`'s default branch does
   `input_file.split('.')[0]` on the aligned PRM's `input_file` field —
   but `make_slc_nsr_py.py`'s `pop_prm_hdf5()`/`write_prm()` never sets or
   writes an `input_file` field into NISAR PRMs at all (every other
   sensor's preprocessor does). `SC_identity` 14
   (`make_slc_nsr_py.SC_IDENTITY_NSR`) isn't one of `_orb_id`'s
   special-cased values, so it always hit the default branch on `""`.
   Fixed with a dedicated `ssc == 14` branch that derives a stable ID from
   the aligned PRM's own filename stem (e.g. `NSR_20260331A`) instead —
   already the canonical per-scene name used everywhere else in this
   pipeline, no new PRM field needed.
2. **xshift/yshift always empty, for every sensor, not just NISAR.**
   `baseline_table` greps `SAT_baseline_py`'s stdout for literal keys
   `"xshift"`/`"yshift"` — but `SAT_baseline_py`'s `write_prm_baseline()`
   actually emits fields named `"rshift"`/`"ashift"` (range shift /
   azimuth shift). Those two lookups have always returned `""`. Fixed by
   matching the real key names.
3. **Wrong epoch for NISAR's YDAY column (off by ~6 years' worth of
   days).** The module docstring already documented an intended
   `SC_identity 5, YR>=2013 (NISAR re-tagged) -> 2014-01-01` epoch — but
   that predates `make_slc_nsr_py.py` actually setting `SC_identity=14`
   (not 5) in NISAR PRMs, so NSR rows always fell into the generic `else`
   branch (2020-01-01 epoch) instead. Didn't break `select_pairs`
   relative-day-separation math (every NSR row shares the same wrong
   epoch, so relative differences were still self-consistent), but did
   make `pre_proc_batch_nsr`'s `stacktable_all.pdf` calendar-year x-axis
   wrong, since that plot hardcodes a 2014 offset. Fixed with an explicit
   `ssc == 14` branch using the 2014 epoch.

All three root-caused by reading `SAT_baseline_py`'s actual output format
and `make_slc_nsr_py.py`'s actual PRM field set side-by-side with
`baseline_table`'s parsing/branching logic — not guessed. **Verified by
reproduction**: a synthetic `SAT_baseline_py` stand-in + a NISAR-shaped PRM
(no `input_file`, `SC_identity=14`) reproduced the exact reported symptom
against the pre-fix code (`" 2026104.5432100000 2296 12.345... -45.678...  "`
— leading space from the empty ORB, trailing double space from empty
rshift/ashift, whitespace-splitting into exactly 4 tokens), and the fixed
code produces all 7 fields correctly against the same synthetic input. Not
a real-HDF5/real-SAT_baseline-binary test (flagged per Rule 8), but a
stronger check than reasoning alone.

**Structural gap found 2026-09-03, via an actual remote NISAR_CSAF run
reaching interferogram formation:** `phasediff_py` failed with `RuntimeError:
The dimensions of range do not match` comparing the master's and a repeat
scene's aligned PRMs from `SLC/` — many steps downstream of where the real
problem occurred. Root cause: `align_batch_nsr`'s `_align_one()` chains
`SAT_baseline_py` → `slc2amp` → `xcorr_py` → `fitoffset_ra` → `resamp_py` →
`mv`/`cp` entirely through `gmtsar_lib.run()`, which by design never raises
on a nonzero exit (see its docstring — matches legacy csh's tolerance of
"benign" warnings). If any command in that chain silently failed for one
scene, the trailing `mv {aligned}.SLCresamp {aligned}.SLC` / `cp
{aligned}.PRMresamp {aligned}.PRM` would just no-op, leaving `SLC/<aligned>`
as whatever `_stage_slc_inputs` originally staged there — a copy of the
*pre-alignment* PRM, carrying that scene's own native dimensions (each
NISAR granule's raw crop is computed independently in `write_slc_hdf5`, so
scene-to-scene native widths can differ by a few pixels even at the same
`region_cut`). `align_batch_nsr` reported success for every scene while one
was silently left un-aligned; the mismatch only surfaced later, far from
the actual point of failure, with the real upstream error already scrolled
off the log. Fixed by adding `_verify_aligned()`, called right after each
scene's `_align_one()`: checks `.PRM`/`.SLC`/`.LED` all exist, then checks
the aligned PRM's `num_rng_bins`/`(num_patches*num_valid_az)` actually match
the master's (which `resamp_py` is supposed to always overwrite from the
master PRM) — failing loudly with the exact scene name and both dimension
sets instead of letting a silent mismatch propagate. **Verified**: a
synthetic test with matching-dims/mismatched-dims/missing-file PRM pairs
confirmed the check passes silently when dimensions match and exits with a
clear diagnostic in both failure cases. Doesn't fix whatever originally
made the upstream command fail for that scene (unknown — the user's log
didn't capture it) — that still needs investigating on a re-run, but now
the failure will point at the right scene immediately instead of surfacing
as a confusing dimension mismatch deep inside `phasediff_py`.

**Root cause + a real code bug found 2026-09-04, from a full `intf.txt` log
of one interferogram's formation on the remote NISAR_CSAF machine:**
`topo_ra.grd` was never created, and the failure cascaded silently through
`grd2cpt`/`grdimage`/`psconvert` (all "Cannot find file topo_ra.grd"),
`intf_batch` symlinking a dangling `topo_ra.grd`, and `filter`/`geocode`
both crashing with `RuntimeError: read_gmt_grd requires netCDF4.` — while
`dem2topo_ra` itself still exited 0 and `intf_batch` reported the whole
run as `rc=0`.

Root cause: **netCDF4 is not installed** in that remote conda environment.
It's a declared dependency (`gmtsar/python/requirements.txt:21`,
`netCDF4>=1.6`) used pervasively by this fork's in-process GMT
replacements (`gmt_grd_io.read_gmt_grd`/`write_gmt_grd` — `dem2topo_ra`'s
FLIPUD, `filter`'s amplitude/HYPOT step, `geocode`'s masking step). Install
it (`pip install netCDF4` or `conda install -c conda-forge netcdf4` in the
`gmtsar` env) and re-run `install.py --rebuild`.

That alone doesn't explain why the failure was so hard to trace, though —
a real bug in `dem2topo_ra`'s `_grdmath_flipud()` made it worse: when the
in-process FLIPUD write raised (here, the missing-netCDF4 `RuntimeError`),
its `except` handler was supposed to "flush the stashed in-memory grid to
disk so the `gmt grdmath` subprocess fallback can still succeed" — but
that flush re-popped `_PENDING_INMEM_GRID`, and the entry had already been
popped earlier in the SAME `try` block, before the failing write. The
second pop always returned `None`, so the entire recovery branch was
unreachable dead code. The fallback subprocess then ran `gmt grdmath
pixel.grd FLIPUD = topo_ra.grd` against a `pixel.grd` that was never
written (the whole point of the in-memory chain is to skip writing it),
failing with `grdmath [ERROR]: pixel.grd is not a number, operator or file
name` — and since `run()` never raises on a nonzero exit, and
`dem2topo_ra()` had no check that `topo_ra.grd` actually got produced,
none of this stopped the pipeline from reporting success.

Fixed two ways: (1) `_grdmath_flipud` now pops the stash exactly once, up
front, into a variable reused by both the primary attempt and the
recovery path, and the recovery's own failure is now fatal (`sys.exit`)
instead of silently falling through to a doomed subprocess call; (2)
`dem2topo_ra()` now checks `topo_ra.grd` actually exists right after the
FLIPUD step and exits loudly if not, instead of printing "END" and
returning normally. **Verified**: a standalone test against the real
`_grdmath_flipud` function (mocking `_write_gmt_grd` to fail like the
missing-netCDF4 case) confirmed the stash is now correctly available to
the recovery path (previously always `None`) and that a recovery failure
now exits before ever reaching the subprocess call; a separate success-path
test confirmed the normal (dependency-present) case still writes directly
with no subprocess and no behavior change.

Separately, the same log showed `conv` failing near-instantly with `Can't
open input data NSR_20260530A.SLC` for one repeat scene — likely the same
class of silent alignment failure the `align_batch_nsr` fix above now
catches, not something new; re-running step 2 after pulling that fix
should confirm whether this scene aligned correctly.

**Wrong default config value found 2026-09-16, via an actual remote
NISAR_CSAF run reaching `geocode`:** `corr.grd` came out entirely `0.0`
(not NaN) for every pixel, and `m2s_py` then raised `ValueError: 'llp' is
empty or not a multiple of 3 float32s (got 0 values)` trying to build the
lon/lat/data triplet file from it. Traced through `filter`'s correlation
formula (`tmp.grd = amp1.grd*amp2.grd`; pixels with `tmp.grd < 5.e-21` are
masked to NaN before the `conv` smoothing step that produces `corr.grd`):
`amp1.grd`/`amp2.grd` (each SLC's own amplitude image) measured ~1e-11 in
magnitude on this real data, so their product (~1e-22) fell below that
hardcoded threshold almost everywhere — GMTSAR's `conv` turns a
near-total NaN mask into an all-zero output rather than propagating NaN,
which is what actually surfaced downstream as "0.0 everywhere" instead of
"NaN everywhere."

Root-caused one level further back via `pre_proc_nsr`'s own diagnostic
prints on both the master and repeat scene, across both frequency bands:
"sigma of integers (2048 < sig < 8192)" came out as `0`, and "fraction
set to 0 after cast" came out at `1.93` out of a max of `2.0` (that
counter sums real+imag channels) — i.e. ~96% of every complex sample was
quantizing to exactly `0` during `make_slc_nsr_py.write_slc_hdf5()`'s
float32→int16 cast. That cast is scaled by `SLC_factor` (a `pop_config`
`SAT_OVERRIDES` field), which was `2.0` for both `NSR_A` and `NSR_B` — a
value carried over from `ALOS2`'s convention, never calibrated against
real NISAR data. NISAR's L1 RSLC product is radiometrically calibrated to
real physical units, a completely different (much smaller) numeric scale
than ALOS-2's raw digital counts, so `2.0` was never in the right
ballpark. Measured directly from a real NISAR_CSAF granule (a 2000×2000
block of `swaths/frequencyA/HH`, sampled by the user via a direct h5py
one-liner): raw complex std ≈0.325 for both real and imag. GMTSAR's own
quantization target is a post-scaling sigma of 2048-8192 (per
`pre_proc_nsr`'s own diagnostic print); solving for the factor that lands
at sigma 4096 gives ≈12,612. **Fixed**: `pop_config`'s `SAT_OVERRIDES`
for `NSR_A`/`NSR_B` changed from `{'SLC_factor': 2.0}` to `{'SLC_factor':
12000.0}` (rounded down from 12,612 for a bit of clipping margin on
brighter pixels), with the full derivation recorded in a comment at the
call site. **Verified**: `python3 -m py_compile utils/pop_config` clean,
and `pop_config NSR_A` now emits `SLC_factor = 12000.0` in its generated
config. **Not verified across multiple granules/incidence angles** — this
is one real granule's measurement; if a different NISAR_CSAF stack still
shows sigma near 0 (or heavy clipping) after this change, this default
needs recalibrating against that data too, not assumed universal. **Two
real-world consequences of this fix that are NOT automatic:** (1) any
`config.NSR_A.txt`/`config.NSR_B.txt` already generated before this fix
still has the old `SLC_factor = 2.0` baked in on disk — pulling this fix
alone does not retroactively edit an existing file; it must be
regenerated or hand-edited. (2) every `.SLC` file already produced by
step 1 under the old factor is ~96% zeroed out (near-total data loss,
not merely misaligned) — unlike the earlier alignment-only issue above,
this is a defect in the data itself, so step 1 genuinely needs to be
re-run for the whole stack, not just step 2/4.

## Landed 2026-07-23 (v2.10.x): native Windows — `install.py --system conda-windows-full`

GMTSAR now builds and runs natively on Windows — no WSL, no
MSYS2/Cygwin toolchain, no admin. CMake/Ninja build against a
conda-provided MinGW-w64 toolchain (`m2w64-toolchain`); the Python
framework and `.csh` scripts stage into `bin/` as flat copies (no
reliable unprivileged symlink on Windows — so `--rebuild` is required
to pick up source edits, unlike POSIX's live symlinks). Git for
Windows is the one external prerequisite (`gmtsar_lib.py` routes all
POSIX-syntax shell-outs through Git Bash; `GMTSAR_WIN_BASH` overrides
the location).

**Verified state (Rule 13 vocabulary): wired ON and clean-room
proven for RS2_SLC_Hawaii only.** Two full clean-room runs 2026-07-23
(fresh clone + fresh conda env `gmtsar_cr`, tarball cache reused per
Rule 14): install rc=0 end-to-end including from-scratch
`conda create` and `pip -r requirements.txt`, then `sweep.py --fast
--cases RS2_SLC_Hawaii --topo-mode-ab` → 6/6 comparisons SUCCESS
(SSIM ≥0.999, grd RMS ≤7e-4), matching a same-commit Linux
py-vs-csh run's numbers. Regression guards:
`bin_py/tests/test_windows_port.py` (one test per real bug found in
the bring-up — cmd metacharacter parsing, missing pip, `_win_bash`
race, WSL-stub bash, os.sep tree-name collapse, conv.c text-mode
fopen).

**Real upstream C bug found by this port**: `gmtsar/conv.c` opened
binary SLC/`.grd=bf` files with `fopen(..., "r")` — text mode, a
silent-corruption no-op on POSIX but catastrophic on Windows
(correlation collapsed to ~0 over 85% of a real RS2 swath). Fix
staged at `c_fixes/conv.c` per the v2.9.0 convention, applied at
build time on every platform by `_apply_c_fixes()` (now called from
`do_windows_build()` too).

**Not done / honest gaps** (per Rule 13, "verified for one case" is
not "verified"):
- No py-vs-csh comparison possible on Windows at all — no `csh`/`tcsh`
  interpreter exists for native Windows; `--topo-mode-ab` (py mode0 vs
  py mode1) is the documented substitute. The C-oracle ground truth
  can only ever come from a POSIX run.
- Only RS2_SLC_Hawaii exercised; the other 20 cases, `bin_py/tests/`
  as a suite on Windows, and `tests/test_install.py` coverage for
  `conda-windows-full` (that script is POSIX-only throughout) are all
  open.
- `distribute_gmtsar_windows.py` (self-contained user bundle): PROVEN
  2026-07-23 (v2.10.3). Root cause of the earlier 27/38 verify failure
  was export FORWARDERS — conda-forge's win-64 libblas/liblapack shims
  forward to mkl_rt.N.dll, invisible to any import-table walk; fixed by
  walking forwarders (`_pe_forwarder_targets`, stdlib PE parse), pinning
  the openblas BLAS variant (`WINDOWS_CONDA_BOOTSTRAP_PACKAGES`), and a
  fail-loud MKL guard. Four more bundle-smoke-found fixes: bundle the
  REAL `usr\bin\bash.exe` (Git's `bin\bash.exe` is a launcher stub),
  launcher PATH gains `pyenv\Library\bin` (the `gmt.exe` CLI lives
  there), `GMT_SHAREDIR` pinned (the gmt.dll copy has the build env's
  share path baked in), and the `gmtsar/python/{utils,bin_py}` tree
  bundled (the `$GMTSAR` import fallback needs it). Evidence: 3-layer
  isolated-PATH verify passes (38/38 exes, bundled python imports,
  bundled bash), AND a full RS2 p2p run from the bundle alone —
  no conda, no Git for Windows in the environment — completed with
  `phasefilt.grd` **bit-identical** (complex-rms 0.0) to the clean-room
  sweep's mode0 reference. Remaining caveat: all runs were on the dev
  host; a physically different bare machine hasn't executed the bundle
  yet. v2.11.0 adds the license-collation step (`do_write_licenses`:
  THIRD_PARTY_NOTICES.md from the env's conda-meta, GMTSAR GPL-3 text,
  Git-for-Windows license, AGPL/ghostscript callout) and publishes the
  zip as a GitHub release asset.

## Explored 2026-07-23: full conda toolchain isolation for `install.py --system conda`

Today's `--system conda` deliberately keeps the SYSTEM's own compiler
chain (gfortran, g++, make, autoconf, csh, ghostscript) rather than
provisioning it via conda — see `do_conda_setup`'s docstring in
`install.py`. User asked whether full isolation (conda providing the
compiler too, nothing system-level required) is actually possible.

**Real answer: yes, confirmed by an actual clean-room build.** Fresh
`git clone`, a genuinely new conda env built with `micromamba` (classic
`conda`'s solver hung 28+ min unsolved on the same package set — a real,
separate finding: this host's conda is old, 4.14.0, pre-libmamba-solver;
switch to `micromamba`/`conda-libmamba-solver` for any future from-conda
bootstrap) from `gfortran_linux-64`, `gxx_linux-64`, `make`, `autoconf`,
`ghostscript`, `tcsh`, plus the existing `CONDA_FORGE_BOOTSTRAP_PACKAGES`
(`gmt=6.4`, `gshhg-gmt`, `dcw-gmt`, `flex`, `hdf5=1.12.*`, `libtiff`,
`liblapack`). Real binaries (`esarp`, `xcorr`, `phasefilt`, ...) built,
linked, and ran correctly — `ldd` showed zero missing shared libraries.

Two real things needed beyond just installing packages:

1. **conda-forge ships no `csh` package**, only `tcsh` — Debian/Ubuntu's
   `csh` is itself just a `tcsh` wrapper. A `csh -> tcsh` symlink inside
   the env's `bin/` is the fix; confirmed sufficient for what GMTSAR's
   Python framework actually needs (per project direction, a real `csh`
   package isn't required, only that csh invocations work).
2. **A genuine GMTSAR source bug, found here for the first time**:
   `gmtsar/fitoffset.c` calls `strlcpy()` with no include/declaration
   anywhere. Implicit-declaration is only a *warning* on GCC < 14 (the
   system's Ubuntu GCC 11.4.0, which today's `--system conda` uses) but
   a **hard error** on GCC 14+ (conda-forge's `gxx_linux-64` is 15.2.0 —
   GCC 14 promoted implicit function declarations to an error by default
   as part of C23 alignment). This is NOT conda-specific — it will also
   break `--system ubuntu` on any host with GCC 14+ (Ubuntu 24.10+,
   Fedora 40+, Arch already ship it). Fix (`strlcpy` -> `snprintf`,
   identical behavior for the fixed short literals involved) is staged
   at `gmtsar/python/c_fixes/fitoffset.c` — NOT applied to the real
   `gmtsar/gmtsar/fitoffset.c` yet (outside `gmtsar/python/`, this
   repo's "everything else stays untouched for clean upstream merges"
   rule) — apply manually or via a proper upstream PR when ready.

`install.py`'s existing `patch_config_mk()` (TIFF/HDF5/GMT path fixup +
`-Wl,-z,muldefs` for GCC 10+'s `-fno-common` default) already handles
the rest correctly once the env is genuinely activated — no new code
needed there, it was just easy to miss applying by hand during manual
clean-room testing (cost real debugging time here).

**Since wired into `install.py` as a real mode**: `--system
conda-linux-full` (v2.9.0), using real env activation and the
target-triplet compiler names (`x86_64-conda-linux-gnu-{cc,c++,gfortran}`),
not plain `gcc`/`g++`/`gfortran` — see `do_conda_setup`'s docstring.

Also landed independently of this exploration: `install.py` now fails
fast with a clear message if `gfortran`/`g++`/`make`/`autoconf`/`csh`/
`ghostscript` are missing under today's `--system conda`, instead of
surfacing as a cryptic `autoconf`/`make` error deep in the build
(`_check_system_build_tools()`, commit `8c169eb`).

## Fixed 2026-07-13: `write_gmt_grd` broke GMT's symlink-follow semantics

Found by a real full 21-case regression sweep (`ALOS_haiti` FAIL:
`phasefilt_mask_ll.png` py 2190×2180 vs csh 2090×2150 — a genuine
100+px, data-content divergence, not a synthetic edge case). Root
cause: `gmtsar/csh/snaphu.csh:41,52` aliases `phase_patch.grd ->
phasefilt.grd` via symlink, then writes `gmt grdmath ... = phase_patch.grd`
— GMT's C netCDF writer follows the symlink and mutates the *target*
file in place (verified directly against the real `gmt` binary).
`utils/gmt_grd_io.py`'s `write_gmt_grd` (the writer behind
`GMTSAR_GRDMATH_PY`, default ON since commit `6814636`, 2026-06-18 —
**not** related to any of today's changes) instead did
`os.remove(grd_path)` before writing, which on a symlink deletes only
the symlink, leaving the target's data untouched. `snaphu.py`'s
landmask-masking step silently never reached `phasefilt.grd`, so its
NaN footprint diverged from the C reference, and that divergence
propagated downstream into `proj_ra2ll`'s data-driven `-R` region
computation — a completely different-looking symptom (pixel dimension
mismatch) three stages removed from the actual bug.

**Fixed**: `write_gmt_grd` now resolves symlinks (`os.path.realpath`)
before removing/writing, mirroring GMT's own behavior. Regression test
added: `bin_py/tests/test_gmt_grd_io.py::TestSymlinkAliasedWrite` (3
tests, including a C-parity test against the real `gmt grdmath`
binary — ground truth, not assumption). Validated via real-data replay
of the ALOS_haiti sweep artifacts: fixed NaN count matches the csh
reference to 1 pixel, and the geocoded region boundary
(`x_min`) moved from 286.9875 (buggy) to 287.033333333, an exact match.
Found and fixed via Mira (`mira-volkov`, agent
`a13485cfdb69e1d4c`), isolated worktree, no defaults flipped.
**Not yet re-validated on a fresh full `ALOS_haiti` sweep run** —
that's the next real-world confirmation step.

## Known bug, not yet fixed

- **`proj_ra2ll_fast`'s NaN mask boundary drifts by 1 pixel column vs.
  the `proj_ra2ll` subprocess, on ~8 pixels out of 1.29M (0.0006%),
  found 2026-07-13 by `bin_py/tests/test_proj_ra2ll_fast.py::
  TestProjRa2llFastVsSubprocess::test_all_files_bit_exact` (a full
  `bin_py/tests/` run, not touched by this session's changes — pre-
  existing since at least v2.5.4, `ccb516f`). All 8 pixels sit at the
  edge of the valid-data footprint (e.g. `(985,235)`: ref=0.846,
  fast=NaN; `(985,236)`: ref=NaN, fast=0.846 — a single-column boundary
  shift, not a value error). Same class of issue as the documented
  proj_ra2ll region-rounding divergence (west-edge cell-count mismatch
  on some TOPS scenes). Negligible in magnitude — explains why the real
  21-case sweep (which uses this exact `geocode` -> `proj_ra2ll_fast`
  path with a shared cache, same as the test) still passes SSIM/RMS
  thresholds cleanly; not chased further this session since it's
  unrelated to any of today's changes and doesn't move any real
  pass/fail outcome. Worth a real fix if the boundary starts mattering
  for a coverage-sensitive case.
- **`utils/tkGUI.gmtsar:86,97`**: `self.gmtsarPath` silently defaults to
  the literal string `'python'` when `p2p_processing` isn't already on
  `$PATH` at GUI launch time, and that value gets prepended into
  `os.environ['PATH']` unconditionally — a silent fallback (violates the
  project's no-silent-fallback discipline). If a user launches the GUI
  from a shell without `$GMTSAR/bin` on PATH, every Config/p2p_processing
  button click fails with an unhelpful subprocess error. Found 2026-07-12
  during a live GUI verification pass (not synthetic — reproduced by
  launching without the PATH export). Cheap fix: fail loud (raise/log a
  clear error) instead of silently substituting a garbage path.
- **`install.sh --python`'s `pip install --upgrade -r requirements.txt`
  is unsafe against a live/shared conda env — incident, found AND
  recovered 2026-07-13.** During a from-scratch new-user onboarding
  test, running `install.sh --conda --python --build` against the same
  `gmtsar` conda env that other processes had open (background test
  sweeps using numba-JIT kernels) caused a partial package upgrade —
  `llvmlite` got bumped to an incompatible version while `numba` didn't
  finish reinstalling (NFS "device or resource busy" mid-swap), breaking
  numba JIT compilation for every kernel in the env. This correctly
  crashed one in-flight sweep (`ALOS_haiti`) — a real, honestly-reported
  failure caused by env fallout, not a code bug; it was not chased as
  one. **Recovered** with a targeted, minimal fix once no other process
  had the env open: `pip install 'numba==0.65.1'` (no `--upgrade`, no
  blanket `-r requirements.txt`), which let pip's resolver pull the one
  correct matching `llvmlite` (0.47.0, satisfying numba 0.65.1's
  `<0.48,>=0.47.0dev0` constraint) without touching any other package.
  Verified via `bin_py/tests/test_gmt_blockmean_py.py` +
  `test_vector.py` (40/40 pass) and a fresh `ALOS_haiti` re-run.
  **Root-cause fix still open**: `install.sh --python` itself still uses
  `--upgrade`, which can repeat this exact failure mode against any
  live/shared env. Fix candidates: drop `--upgrade` (only install
  missing packages, don't touch already-satisfied ones), or add an
  explicit warning/check that the target env has no other attached
  processes before installing.

Living roadmap. Read this before re-deriving "what's left to port" from
scratch — this survey (2026-07, HEAD v2.5.6) already did that work and
cross-checked it against the actual code, not just prior release notes
(several of which turned out to disagree with each other and with the
current code — see the surface entry below for the cautionary tale).

**Governing principle**: a module stays C because a Python port is
judged unlikely to beat it (measured, or reasoned from its nature — e.g.
I/O-bound format parsing), not because "it already works so we didn't
try." Every entry below states which is true.

## Wiring-status ledger (project_rules.md Rule 13)

"Ported" and "wired ON by default" are different states. Every module
below is tagged with exactly one:

- **[1-ON]** wired ON by default, both gates passed and reviewed.
- **[2-OFF-pending]** ported, parity proven, gate-2 evidence exists
  (win or tie) — not yet promoted, pending a full-sweep review.
- **[3-OFF-lost]** ported, parity proven, gate-2 **failed** — a real,
  measured loss, correctly left off. Not a gap.
- **[4-partial]** ported on a subset only, no dispatcher/call site wired.
- **[5-none]** never attempted — split into worth-it / not-worth-it below.

| Module | State | Evidence | Dispatcher |
|---|---|---|---|
| `xcorr_py` | [1-ON] | ~30x faster (2026-07-12); **flagged 2026-07-13 for re-check under a quiet system, see Rule 12c and "Tried, kept on C" below** — real pipeline's own C-binary timer shows a much smaller gap on RS2 | `p2p_stages.py`, unconditional |
| `resamp_py` | [1-ON] | ~1.3x faster, byte-identical (re-wired 2026-07-12, was OFF-lost as v2) | `p2p_stages.py`, unconditional |
| `SAT_llt2rat_py_v2` | [1-ON] | +7.6% vs v1, ties C, no NFS instability (verified 2026-07-12) | `install.sh` symlink, unconditional |
| `gmt_surface_py` | [1-ON, correctness only] | C 269s vs py 412s — C faster; wired for bit-parity, not speed | `dem2topo_ra`/`align_tops`/`proj_ll2ra`/`tide_correction` |
| `gmt_grdmath_py` | [1-ON, selective] | 14.2x per-call (isolated MUL sites only) | 16 sites, see release_notes_v2.5.0 |
| `gmt_grdcut_py` | [1-ON] | 1.2-4.2x depending on call site | 19 sites |
| `gmt_grdsample_py` | [1-ON] | parity per-call; 1.7x in warm multi-call reuse only | `grdsample_wrapper.py` |
| `phasediff_py`, `phasefilt_py`, `gmt_grdfill_py`, `align_tops`, `make_los_py` | [1-ON, audit gap] | correctness evidence only, **no isolated gate-2 timing exists** | various, see "Tried, kept on C" below |
| `make_slc_s1a_py` | [1-ON] | +1.4-1.8x, byte-identical; confirmed end-to-end on real S1A_SLC_TOPS_Greece sweep (10/10 SUCCESS, 2026-07-13) | `pre_proc`, `GMTSAR_S1A_PREPROC_PY=1` |
| `make_slc_nsr_py` | [1-ON] | +19x, byte-identical; confirmed end-to-end on real NISAR_Ethiopia sweep (6/6 SUCCESS, csh 431s->py 152s, 2026-07-13) — found+fixed a real `_get_config` quote-stripping bug along the way, see below | `pre_proc_nsr`, `GMTSAR_NSR_PREPROC_PY=1` |
| `gmt_blockmean_py` | [1-ON] | +3.7-19.3x, tolerance-equal (not byte-identical, see below); low-risk flip since only topo_interp_mode=1 (opt-in per-case, not the default) exercises it | `dem2topo_ra`, `GMTSAR_BLOCKMEAN_PY=1` |
| `make_slc_rs2_py` | [1-ON, Rule 13a] | ~1.3x slower individually; wired anyway (deployment simplicity, pre_proc ~7.8% of total time); confirmed end-to-end (RS2_SLC_Hawaii 6/6 SUCCESS, 1.96x case-level, 2026-07-13) | `pre_proc`, `GMTSAR_RS2_PREPROC_PY=1` |
| `make_slc_tsx_py` | [1-ON, Rule 13a] | slower individually (numpy import tax); wired anyway; confirmed end-to-end (TSX_SLC_Hawaii 6/6 SUCCESS, 1.19x case-level, 2026-07-13) | `pre_proc`/`gmtsar_lib.py`, `GMTSAR_TSX_PREPROC_PY=1` |
| `make_slc_csk_py` | [1-ON, Rule 13a] | ~1.3-2x slower individually; wired anyway; confirmed end-to-end (CSK_SLC_Italy 6/6 SUCCESS, 1.04x case-level, 2026-07-13) | `pre_proc`, `GMTSAR_CSK_MAKE_SLC_PY=1` |
| `make_slc_csk2_py` | [3-OFF-lost] | ~4-5x slower, byte-identical | `pre_proc`, `GMTSAR_CSK_PREPROC_PY=0` |
| `gmt_triangulate_py` | [3-OFF-lost] | 1.4-9x slower (Qhull vs GMT's linked Shewchuk Triangle) | `dem2topo_ra`, `GMTSAR_TRIANGULATE_PY=0` |
| `ALOS_pre_process_py` | [4-partial] | IMG-parsing subset byte-identical, +2.1x — LED/orbit/Doppler not ported | none — parity is partial |
| `SAT_llt2rat_py` (v1) | removed (v2.7.1 cleanup) | byte-identical, but v2 wins on speed+stability | `bin_py/SAT_llt2rat_py` (v2, live baseline) |
| `resamp_py_v2` | removed (v2.7.1 cleanup) | unstable NFS/numba cache, only tied with C at best | `bin_py/resamp_py` (v1, live baseline) |
| `SAT_look` | [5-none] | still C, `calc_look_vector` calls it directly | n/a |
| `iono_gauss` | ported, opt-in correctly | fails gate 1 by design (2% divergence allowed) | `GMTSAR_IONO_GAUSS_PY` unset |
| `gmt triangulate`(via triangulate_py, see above), `grdedit`, `trend2d`, `grdlandmask`, ~60 other `gmt` subcommands | [5-none] | see "Never attempted, not worth it" below | n/a |
| SAR preprocessors not yet attempted (`make_slc_gf3/lt1`, `ENVI_preproc`, `ERS_preproc`, `calc_dop_orb`, `extend_orbit`, `update_PRM`) | [5-none] | no cached test fixture, or disproportionate complexity | n/a |
| `esarp.c` (real SAR focusing DSP) | [5-none, highest-leverage if revisited] | never scoped — see "Deferred by design" below | n/a |

## Done

- 1:1 utility coverage of the legacy csh codebase (every csh script has a
  Python counterpart), merged upstream via PR #1114.
- 13 compute-kernel dispatchers wired on by default at their primary call
  sites: cross-correlation (`xcorr_py`), phase difference (`phasediff_py`,
  baseline ≤1000m only — longer baselines fall back to C), resampling
  (`resamp_py`), `SAT_llt2rat` (unconditional, no C fallback), block median
  (`gmt_blockmedian_py`), phase filter (`phasefilt_py`), grdmath
  (`gmt_grdmath_py`, selective — see below), grdsample, grdcut, grdfill,
  align_tops, make_los, iono (the grdmath/grdfilter/surface subprocess
  chain only — see corrections below).
- **Corrections from a 2026-07-12 audit** (do not repeat these errors):
  `SAT_look` is **still C** — `utils/calc_look_vector` calls the C binary
  directly, unconditionally; no `SAT_look_py` exists. `iono_gauss` (the
  scipy.ndimage Gaussian substitution, distinct from the iono grdmath
  chain above) is **opt-in only** (`GMTSAR_IONO_GAUSS_PY` unset by
  default) — correctly so, its own test asserts up to 2% divergence from
  C, so it fails gate 1 (bit-identical) by design and must stay opt-in.
  `merge_tops` does not exist as a file/dispatcher; the real merge step is
  `merge_unwrap_geocode_tops`, whose sub-calls (grdmath/grdsample/grdcut)
  are already covered above.
- **Gate-2 (speed) audit gap**: of the 13 kernels above, only
  `gmt_blockmedian_py` (~10% faster, single-site number), `gmt_grdmath_py`
  (14.2x per-call on isolated MUL sites), `gmt_grdcut_py` (1.2-4.2x
  depending on call site), and `gmt_grdsample_py` (parity per-invocation,
  1.7x in warm multi-call reuse) have **any** isolated timing evidence.
  `phasediff_py`, `phasefilt_py`, `gmt_grdfill_py`, `align_tops`,
  `make_los_py` are wired on by **correctness evidence only** — no
  standalone C-vs-Python timing exists yet. None show evidence of
  *failing* gate 2 (unlike `resamp_py`, below), but "no evidence of
  failing" is not the same as "passes" — this is an honest gap, not a
  verified pass.
- `topo_interp_mode=1` (triangulation fast-path for `dem2topo_ra`):
  16/21 cases PASS, 2.04x aggregate speedup, v2.5.6 fixed a real
  masked-cell/`phasediff`-segfault bug. Opt-in, not default — see the
  talk deck (`slides/20260722_python_framework_gui/`) for the honestly
  reported accuracy tradeoff on the 5 non-passing cases.

## Tried, kept on C — documented, not a gap

- **`resamp_py` / `xcorr_py`, resolved 2026-07-12, `xcorr_py` re-flagged
  2026-07-13** (fresh, isolated, single-core-pinned, real RS2_SLC_Hawaii
  data, parity-checked byte-identical to C for both): `xcorr_py` is
  genuinely faster — C's `xcorr.c` re-builds a GMT FFT plan on every one
  of 1000 calls (91% CPU but 409s of *system* time out of 651s user —
  plan/malloc churn, not compute), a real, reproducible architectural
  difference. **But the exact ~30x figure is not yet trustworthy as a
  clean number**: a 2026-07-13 re-measurement under `load average: 21`
  (contended by two orphaned processes from an earlier, already-
  completed Mira agent that never got cleaned up, plus another user's
  job) gave C `xcorr` 1054s — while the SAME C binary's own internal
  timer, from a real pipeline sweep run minutes earlier on the identical
  case/parameters, printed `elapsed time: 121.5s`. That's a >8x
  discrepancy on the C side alone, meaning both the original 2026-07-12
  measurement and the 2026-07-13 re-check may be contaminated by system
  load, not a clean single-thread number. Per Rule 12c: **a from-scratch
  re-measurement under a quiet system (check `uptime` first) is needed
  before citing a specific multiplier again.** What's solid: the
  architectural reason (FFT plan rebuild per call) and the direction
  (faster) — not yet the exact number.
  This was a split story at first: `bin/resamp_py` had been symlinked to
  a since-archived alternate implementation (the de facto wired default
  at the time, contradicting the old rc2 note which had benchmarked the
  unwired plain `resamp_py`). That alternate's timing was **unstable
  (10-58s)** because its numba on-disk JIT cache defaulted to
  `bin_py/__pycache__`, which lives on NFS — synchronous NFS stat/open
  round-trips during cache validation. Pointing `NUMBA_CACHE_DIR` at
  local disk stabilized it to ~11-12s, roughly **tied** with C
  (10.9-13.3s) — not a clear win. Plain `resamp_py` was the actually-faster
  variant (~1.3x, byte-identical) and wasn't what was deployed.
  **Fixed 2026-07-12**: `install.sh` and the live `bin/resamp_py` symlink
  now point at the single production `resamp_py` (no version suffix —
  production code shouldn't carry one). `resamp_py` can now be cited as
  a consistent ~1.3x speedup, byte-identical to C. The old alternate's
  NFS-numba-cache instability is no longer reachable via the default
  wiring; it lived at `bin_py/archive/resamp_py_v2` for reference until
  the v2.7.1 cleanup removed it (recoverable from git history at commit
  `445623e` or earlier if ever needed — see `docs/release_notes/
  release_notes_v2.7.0.md` and earlier for the full incident writeup).
  `SAT_llt2rat_py` v1 was removed the same way, same release.
- **`gmt surface` (biharmonic spline fit)**: a bit-faithful numba/Cython
  port (`gmt_surface_py`) exists and is wired ON by default at most call
  sites (`dem2topo_ra`, `align_tops`, `proj_ll2ra`, `tide_correction`) —
  **for correctness/bit-parity, not speed**. Measured (v2.5.5): C 269s vs.
  Python 412s single-threaded — C is faster. Two parallelization attempts
  (red-black SOR, domain-decomposition prototypes in `docs/experiments/`)
  both failed: the kernel is memory-bandwidth-bound, not compute-bound, so
  more threads don't help. This is the single clearest "we tried, we
  measured, we couldn't beat it" case in the whole framework.
  - **Caution for future readers**: earlier release notes (v2.4.0) and an
    even-earlier session memory each stated a *different* reason for any
    non-default surface behavior (one claimed near-parity speed, one
    claimed a 0.46m accuracy divergence at CSK scale). Neither matches the
    current code or the later, more careful v2.5.5 measurement. Always
    verify against the current code (`grep GMTSAR_SURFACE_INPROC utils/`)
    and the most recent release note, not the first one you find.
  - **Narrower, separate exception**: `proj_ra2ll`'s raln/ralt construction
    uses C surface by default via its own gate (`GMTSAR_PROJ_SURFACE_PY`,
    default OFF) — a real, different, already-fixed bug (v2.5.4): the
    Python port's ~1e-5° edge roundoff shifted the geocoded region by a
    full coarse-lattice cell on `S1A_SLC_TOPS_Greece`/`TOPS_LA`. Do not
    conflate this with the general surface-speed story above.
- **`snaphu` (phase unwrapping)**: a full numba/Cython solver port exists
  (`bin_py/snaphu_py/`) but is not production-ready — `docs/dev_notes/
  NOTES_SNAPHU_FIX.md` documents specific unresolved bugs (a min-cost-flow
  pred-chain cycle hang on 30x30 synthetic input, an 8x10 numba-only
  infinite hang, a result divergence from the scalar oracle on some 5x7
  seeds). Never reached real data — synthetic tests fail first. The
  default Python path (`utils/snaphu.py`) does I/O/staging only; the
  actual unwrap still calls the C `snaphu` binary. Low priority to unblock
  regardless: snaphu is ~0.3% of pipeline wall-time.

## Attempted 2026-07-12: gmt_triangulate_py — the predicted win that wasn't

Ported (`utils/gmt_triangulate_py.py`, `scipy.spatial.Delaunay`/Qhull +
barycentric interpolation), wired at both `dem2topo_ra` call sites behind
`GMTSAR_TRIANGULATE_PY`, default OFF. This was the one candidate on the
list flagged as "genuine compute, most likely real win" — turned out to
be the opposite.

**Parity**: pass on typical scale (RS2_SLC_Hawaii, 964,812 pts — bit-
identical, 0 mismatches). One documented gap at large scale
(ALOS4_Pinon, 6.16M pts): 10 of 29.3M grid nodes diverge (max diff 12.2),
traced to near-degenerate/near-cocircular point quads where Qhull and
GMT's linked Shewchuk Triangle library pick a different diagonal — a
genuine, rare (~3.4e-7 of nodes) algorithmic tie-break, not roundoff.

**Speed: fails gate 2**, consistently, 1.4-9x **slower**. Root cause:
`scipy.spatial.Delaunay`'s build step alone is ~9.4-9.9s, an opaque Qhull
C call with no numpy vectorization angle. Tried qhull_options tuning (no
improvement) and `matplotlib.tri.Triangulation` (6.9s, still ~3x slower).
GMT's C reference already uses the field-optimal library for this exact
operation — general-purpose Qhull has no answer for it. Wired but kept
OFF; only a direct C-extension binding to Shewchuk's Triangle (or GMT's
own routine) could plausibly close this, and that needs explicit
sign-off before attempting.

**Lesson**: "genuine compute" was the wrong reason to expect a win here —
the C reference wasn't hand-rolled or naive, it was already using the
best-in-class library. The earlier framing conflated "compute-heavy" with
"beatable"; they're not the same thing.

## Attempted 2026-07-12: gmt_blockmean_py — confirmed the predicted cheap win

Ported (`utils/gmt_blockmean_py.py`), wired at both `topo_interp_mode=1`
call sites in `dem2topo_ra`, gated by `GMTSAR_BLOCKMEAN_PY` (default OFF
pending a full sweep). Reused the `gmt_blockmedian_py` bin-partition
scaffold; mean reduction needs no per-bin sort, so it's simpler *and*
faster than blockmedian's own kernel — no Numba kernel needed, pure numpy
`reduceat`.

**Correction to the original prediction**: byte-identity is **not**
achievable (unlike blockmedian) — GMT's internal float64 summation order
isn't guaranteed to match numpy's, confirmed empirically. Uses the
project's documented doubles tolerance instead (`atol=1e-9`,
project_rules.md Rule 7 Phase C): real-data max abs diff 4.5e-12
(RS2_SLC_Hawaii) to 7.3e-12 (ALOS_haiti), and the **downstream
`topo_ra.grd` is byte-identical end-to-end** — the roundoff is fully
absorbed by the subsequent `surface`/`grdfill` fit, so this is a
non-issue in practice.

**Timing**: 19.3x faster at ALOS_haiti scale (906k rows, 2.28s→0.12s),
3.7x faster at RS2_SLC_Hawaii scale (965k rows, 0.29s→0.08s). Both gates
pass. Next step before flipping default ON: a full sweep across the
known-clean mode=1 PASS cases (see `bin_py/tests/test_gmt_blockmean_py.py`
docstring for the list) — 4 mode=1 cases have a pre-existing, unrelated
mode=0-vs-1 divergence not touched by this port.

## Never attempted, not worth it

| Command | Where | Why not |
|---|---|---|
| `gmt grdedit` | `utils/geocode` (10x), `utils/filter` (1x) | Pure header-metadata rewrite, ~ms each — no pixel compute to gain. |
| `gmt trend2d` | `utils/fitoffset.py:105,119-120` | Trivial 2D fit, <=6 coefficients, negligible wall-time. |
| `gmt grdlandmask` | `utils/landmask:58` | One-shot per case; needs the GSHHG coastline database — high effort for one call. |
| `psconvert`, `grdimage`, `makecpt`, `psscale`, `grdgradient`, etc. | figure-rendering call sites across `p2p_stages.py`, `snaphu.py`, `stack`, `grd2kml`, `grd2geotiff` | Final PNG/PDF output only — doesn't gate any numerical product. |
| ~60 other `gmt` subcommands (`grdinfo`, `grd2xyz`, `xyz2grd`, `gmtconvert`, `grdtrack`, `gmtinfo`, `project`, ...) | throughout `utils/` | Grid I/O / projection / metadata — see `docs/release_notes_v2.4.0.md` "Scope & dependencies" for the full accounting. GMT remains a hard runtime+build dependency; this is not a gap to close, it's the architecture. |

## Attempted 2026-07-12: SAR sensor preprocessors, tested empirically

The "deferred by design" judgment below was speculative until 2026-07-12,
when it was actually tested: parallel Mira ports of
`make_slc_s1a/csk/csk2/tsx/rs2/nsr` and `ALOS_pre_process`, each validated
for bit-parity against the real C binary on real cached test data (Rule 7),
each timed honestly. Preprocessing is **~7.8% of total case wall time**
(profiled sum across the 21-case sweep: `dem2topo_ra` 78.8%, `pre_proc`
7.8%, `merge_unwrap_geocode_tops` 5.6%, `geocode` 2.8%, `intf` 1.9%,
`resamp_py` 1.7%, `xcorr_py` 1.0%, `snaphu` 0.3%) — real but not where the
budget is, so a modest per-sensor slowdown is an acceptable tradeoff for
having a tested, bit-faithful Python alternative on record.

Results (parity always checked on real data, not synthetic — see
`bin_py/tests/test_make_slc_*_py.py` for each):

| Sensor | Parity | Speed vs C | Wired default |
|---|---|---|---|
| `make_slc_s1a` | PASS, byte-identical | **1.4-1.8x faster** (memmap bulk read replaces per-scanline TIFF calls) | `GMTSAR_S1A_PREPROC_PY`, OFF |
| `ALOS_pre_process` | PASS on the IMG-parsing subset only — LED/orbit/Doppler NOT ported (real ~2000-line transitive C closure, multi-day scope) | ~2.1x faster on the ported subset | not wired — parity is partial, no dispatcher created |
| `make_slc_rs2` | PASS, byte-identical | ~1.3x **slower** (I/O-bound, numpy overhead) | `GMTSAR_RS2_PREPROC_PY`, OFF |
| `make_slc_tsx` | PASS, byte-identical | ~on par to slower for one-shot calls (numpy import tax ~2.5-3s dominates) | `GMTSAR_TSX_PREPROC_PY`, OFF |
| `make_slc_nsr` | PASS, byte-identical (real 14GB NISAR_Ethiopia fixture, both freq modes) | **~19x faster** — h5py sliced reads (only the needed region) vs C mallocing the whole ~11GB array + per-pixel scalar cast loop | `GMTSAR_NSR_PREPROC_PY`, OFF pending review |
| `make_slc_csk2` | PASS, byte-identical (CSG-format fixture built via HDF5 hard-link onto real CSK_SLC_Italy pixel data — no real CSG product exists in the repo, disclosed in module/test docstrings) | ~4-5x **slower**, even after 2 optimization rounds (54s→21s) | dispatcher ready, no upstream caller wired (no CSG case exists yet) |
| `make_slc_csk` | PASS, byte-identical (3 real CSK_SLC_Italy SCS_B acquisitions incl. a 1.9GB SLC) | ~1.3-2x **slower** — I/O-bound HDF5 chunked reads dominate both sides | `GMTSAR_CSK_MAKE_SLC_PY`, OFF |

**All 7 planned preprocessor ports are now complete** (S1A, ALOS-partial,
RS2, TSX, NSR, CSK2, CSK). Final tally: 2 real wins (S1A 1.4-1.8x, NSR
19x), 1 partial-scope win (ALOS, LED/orbit not ported), 4 honest losses
(RS2, TSX, CSK2, CSK — all I/O-bound, all correctly wired OFF). Confirms
the empirical pattern: mechanical porting is genuinely easy, speed is a
coin flip on I/O-bound work, and every single port caught at least one
non-obvious verbatim-arithmetic or data-format trap that a synthetic-only
test would have missed (see the `str2double`/CSK1-vs-CSG/pop_led_hdf5
notes in each agent's own report, not repeated here).

Common non-obvious trap found in **every** port so far: each sensor's C
code parses XML/ASCII header fields with a hand-rolled digit-by-digit
`str2double` (not `strtod`/`float()`) — using Python's `float()` instead
silently diverges in the last ULP on real values. All ports above reproduce
`str2double`/`cat_nums`/date-parsing verbatim rather than substituting the
"obviously equivalent" library call. This is the real cost behind "easy to
port": the mechanical transcription is a few hours; catching this class of
trap is what actually takes the time.

**Verdict so far**: mixed, exactly as expected for I/O-bound code — S1A
wins, ALOS partially wins, RS2/TSX lose narrowly. None are wired on by
default pending review; all are honestly documented, tested alternatives.
Original speculative framing (below) is superseded by this table for the
4 sensors above — kept for the remaining un-tested ones.

- **SAR sensor preprocessors, remaining untested** (`make_slc_gf3/lt1`,
  `calc_dop_orb`, `extend_orbit`, `update_PRM`) —
  real, confirmed-unported C/Fortran source under `preproc/*/src*/`
  (233-407 lines each). `make_slc_gf3`/`lt1` excluded from the 2026-07-12
  round: no real regression-test tarball cached in `tests/cases.py`, so
  bit-parity validation per Rule 7a isn't possible yet. `ENVI_preproc` and
  `ERS_preproc` also excluded: ENVI vendors an entire third-party
  `epr_api-2.3` library, ERS has multiple format variants — both
  significantly more complex than the tested set, deferred to separate
  future scoping rather than assumed equally "easy."
- **`esarp.c` — scoped 2026-07-13, real range-Doppler SAR focuser, not a
  quick win.** The actual focusing math (range/azimuth compression) lives
  in `gmtsar/gmtsar/esarp.c` plus 10 linked files (~1150 total C lines:
  `rng_ref.c`, `rng_cmp.c`, `trans_col.c`, `rmpatch.c`, `acpatch.c`,
  `aastretch.c`, `shift.c`, `radopp.c`, `fft_bins.c`, `intp_coef.c`,
  `spline.c`), sharing ~30 PRM-derived globals via `soi.h` (no clean
  function signatures). Full findings:
  - **FFT**: GMT's own `GMT_FFT_1D()` API, runtime-dispatched inside
    libgmt — on this machine (`ldd bin/esarp`), that resolves to
    single-precision FFTW3 + threads. The exact backend is an external
    GMT-build detail, not fixed by this repo — any future port's "bit-
    faithful" oracle is only as stable as the linked GMT build; pin and
    document the GMT version before treating output as ground truth.
  - **Structure**: almost entirely per-line/per-column, not batched — one
    1-D FFT per range line (`rng_cmp.c`), one per range-bin column
    (`trans_col.c`), one forward+inverse pair per range bin in azimuth
    compression (`acpatch.c`). This is exactly a textbook batched-FFT
    vectorization case (`scipy.fft.fft(..., axis=..., workers=-1)`) —
    genuinely promising, 50-100x plausible, but batched vs. per-line FFTW3
    calls are different call sequences and parity must be proven
    empirically, not assumed.
  - **Algorithm**: a real range-Doppler focuser with range cell migration
    correction (`rmpatch.c`, 8-point sinc resampling — not
    `scipy.signal.resample`), range-varying azimuth matched filtering with
    a half-spectrum-split phase treatment (`acpatch.c`), and an optional
    azimuth stretch using a custom 1970-Goddard-algorithm cubic spline
    (`spline.c`) — explicitly NOT `scipy.interpolate.CubicSpline`'s
    boundary conditions. `spline.c`'s own header admits an *unexplained*
    Fortran-vs-C divergence at extrapolation boundaries, never resolved —
    a live known-quirk any port must get explicit sign-off on reproducing.
  - **Reference literature**: none in-tree (unlike `gmt_surface_py`, which
    had a citable Smith & Wessel 1990 paper). Sourced "from Howard Zebker
    ... Stanford interferometry package" (`esarp.c:7-8`) with ad hoc
    1996-2011 modifications layered on — correctness must be inferred by
    reading the C line-by-line, high risk of library-name-alike
    substitution (e.g. reaching for `scipy.signal.resample` where the C
    does something bespoke) on the RCMC and azimuth-compression stages.
  - **Test data**: already on disk, no new download needed —
    `work/csh_test/ALOS_Baja_EQ/raw/*.raw` + `.PRM` (esarp's exact input
    pair, 747MB), with a regenerable C oracle (`esarp` binary present,
    single CLI call, well under a minute to rerun fresh — existing `.SLC`
    outputs there are stale, per Rule 9, and must be regenerated, not
    reused as-is).
  - **Effort**: calibrated against today's 7-preprocessor-in-one-session
    baseline, this is NOT a one-session job — realistically **1-2 weeks**
    end-to-end (3-5x a preprocessor's effort), with RCMC and azimuth
    compression individually harder than any of the 7 preprocessors
    combined, and exactly the "vectorize a branch-dependent algorithm"
    trap Rule 7 warns about.
  - **Verdict**: still worth doing eventually — the batched-FFT case is
    real and test-data logistics are solved — but it needs to be staffed
    as a dedicated 1-2 week project with explicit sign-off on the
    `spline.c` extrapolation question and a pinned GMT/FFTW version for
    the oracle, not slotted in as "the next quick preprocessor-style
    port." The old "highest-leverage if focusing speed is ever
    prioritized" framing undersold both the difficulty and the specific
    two-checkpoint risk.

## Open questions carried over from PLAN.md (2026-07-13)

`PLAN.md` (the pre-2026-05-14 roadmap this file supersedes — see below)
had every Phase 1/2/4 utility it planned confirmed already shipped
(`baseline_table`, `make_dem`, `select_pairs`, `pre_proc_batch`,
`align_batch`, `intf_batch`, `batch_processing`, `unwrap_parallel`,
`prep_sbas`, `stack`, `stack_corr`, `stack_coherence_mask`,
`extract_one_time_series` all exist in `utils/`) — but 3 genuinely open
questions from its §8 never got answered and aren't tracked anywhere
else:

- **SBAS test fixture**: no multi-pair time-series tarball is in the
  regression sweep. `prep_sbas`/`stack`/`stack_corr` exist and presumably
  work, but have no case-level parity coverage against csh — unlike
  every single-pair P2P utility. Does `topex.ucsd.edu/gmtsar/tar/` host
  one, or does this need curating?
- **Parallelism budget**: should `*_parallel` utilities (`unwrap_parallel`,
  etc.) share the test sweep's `MAX_PARALLEL` env var, or manage their
  own? Unresolved — check for resource contention before running both
  concurrently on the same host.
- **csh deprecation horizon**: is the long-term goal to remove the
  remaining csh shell-out shims entirely, or keep them as an intentional
  fallback? Affects how aggressively future ports should touch internals
  vs. leave working shell-outs alone.
- **Gate-2 (speed) isolated benchmarks still missing** for `phasediff_py`,
  `phasefilt_py`, `gmt_grdfill_py`, `align_tops`, `make_los_py` (see
  "Gate-2 (speed) audit gap" under Done, above) — all 5 are wired ON by
  correctness evidence only, no standalone C-vs-Python timing exists.
  None show evidence of *failing* gate 2, but that's not the same as a
  verified pass. Run each in isolation on real data, matched hardware,
  quiet system (Rule 12c — the `xcorr_py` multiplier retraction on
  2026-07-13 was exactly this mistake: measuring under load and trusting
  it). `phasediff_py` is the most requested first target (2026-07-14).

Also carried over (originally from `docs/audits/AUDIT_stage_cache_mira57.md`,
removed in the v2.7.1 doc cleanup — recoverable from git history if the
full investigation transcript is ever needed): `tests/stage_cache.py`
was found "architecturally broken," needing a redesign (fingerprint
post-stage outputs instead of raw/mutate-restore) rather than a bugfix
— still `GMTSAR_STAGE_CACHE=0` by default, unclear if the redesign was
ever done. Worth a fresh look before trusting it.

`PLAN.md`'s full mission-log history (every dated status snapshot,
Mira-by-Mira roadmap, and the now-superseded `gmt_surface_py` perf
numbers — a *third*, independently stale figure for a story this file
already had to reconcile from two others) was archived unedited at
`docs/reports/PLAN_archived_2026-05-14_to_2026-06-13.md` and removed in
the v2.7.1 doc cleanup (recoverable from git history). Its
still-relevant technical content (the GMT netCDF attribute spec) was
extracted to `docs/GMT_NETCDF_ATTR_SPEC.md`, which live code
(`utils/gmt_grd_io.py`, `utils_pygmt/gmt_compat.py`, `utils/gmt_inproc.py`)
now points to instead.

## How to keep this coherent

When any of the above changes, update this file **and** grep the talk
deck (`slides/20260722_python_framework_gui/slides.tex`) and any other
release notes that reference the same claim — this file exists because
three prior sources (two release notes, one session memory) disagreed
with each other and with the code on the surface-fitting story. Don't let
a fourth stale copy start the same problem again.
