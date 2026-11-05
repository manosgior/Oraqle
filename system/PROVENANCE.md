# system/ — provenance

Code of §7 (system_bottleneck.tex, "Readout as a System Bottleneck").

## Origin

| Files | Origin | Changes |
|---|---|---|
| `analyze.py analyze_decoders.py crosscheck_stim.py decode_timing.py discriminators.py plot_readout_bottleneck.py soft_readout_cnn.py sweep_decoders.py sweep_large.py sweep_ler.py sweep_soft.py system_cost.py uncertainty.py workloads.py` | `oraqle-ics/sims/*.py` (paper repository, Overleaf git, HEAD **d7cd4e9**, 2026-11-02; working tree clean for `sims/`). This is the canonical copy: steve/graham `~/oraqle-ics-sims/sims` is identical for 13 files, has older `system_cost.py` and `plot_readout_bottleneck.py` (2026-10-26) and lacks `uncertainty.py`. | path plumbing only (below) |
| `helpers/original/{chain_soft.sh, watch_decoders.sh, probe_cost.py, probe_large.py, probe_time.py, test_uf.py}` | steve/graham `~/oraqle-ics-sims/` (no git), verbatim (md5 8a62ba0c, d23a02af, 47030737, 601601a8, d5d5f015, 0e781bef) | none |
| `helpers/*.{sh,py}` | the same six files | path plumbing; `watch_decoders.sh` now ends with `merge_decoders.sh` (3 files) instead of the original 2-file merge (see below) |
| `summarize_soft.py` | **new**: the inline call that wrote `summary_soft.json` (Claude transcript 94019386, line 1417, 2026-10-26T12:36Z), as a script | — |
| `merge_decoders.sh` | **new**: the 3-file `analyze_decoders.py` call that wrote the published `decoders_summary.{txt,json}` (transcript 94019386, line 2151, 2026-10-29T17:59Z) | — |
| `run_paper_sweeps.sh` | **new**: every sweep command of the paper runs (transcript 94019386, lines 463, 490, 562, 791, 935, 974/1387, 1085, 1901, 2058, 2146, 2872) in the new layout | — |
| `derive_uniform_fraction.py` | **new** (results/external sub-task): the inline stim run that produced `UNIFORM_FRACTION` (transcript 94019386, line 1039, 2026-10-26T11:57Z) | — |
| `paper_numbers.py`, `replay_points.py` | **new**: §7 paper-number check and tier-2 exact replay | — |

The CNN code that `soft_readout_cnn.py` needs is no longer duplicated here: `oraqle-ics/sims/qsd/{CNN.py,
_colleague_prep.py}` were byte copies of `discriminators/networks/CNN.py` (md5 8b269681…) and
`discriminators/runners/_colleague_prep.py` (md5 c0e0a6ff…), which `soft_readout_cnn.py` now imports;
`verify/qec_system.py` checks the two md5s.

## Path plumbing (no change to simulation logic, noise models, seeds or presets)

| File | Was | Now |
|---|---|---|
| `sweep_ler.py` (and through it `decode_timing.py`, `crosscheck_stim.py`, `sweep_*`) | `LATSIM = $LATSIM or /work/lattice-sim/sim` | default `<repo>/qec/lattice-sim/sim` |
| `sweep_decoders.py` | `--soft` default `/work/data/soft_params.json` | `$ORAQLE_RESULTS/system/soft_params.json` |
| `soft_readout_cnn.py` | `/data/five_qubit_data/DRaw_C_{Tr_v0-001,Te_v0-002}`; `sys.path += sims/qsd` | `oraqle_env.RAW_TRAIN/RAW_TEST` (`$ORAQLE_DATA`) + `require_dataset()`; imports from `discriminators/` |
| `analyze.py`, `summarize_soft.py`, `system_cost.py`, `uncertainty.py`, `merge_decoders.sh` | read and wrote `<oraqle-ics>/data/sims` | `--data DIR` (default `$ORAQLE_WORK/system`), still read and write DIR |
| `plot_readout_bottleneck.py` | read `<oraqle-ics>/data/sims`, wrote `<oraqle-ics>/plots` at import time | `--data` (default `$ORAQLE_RESULTS/system`), `--out` (default `$ORAQLE_OUT/paper_figures/system`), `--only <figure>`; loading moved into `configure()` |
| `analyze_decoders.py` | `from analyze import …` relied on the cwd | inserts its own directory into `sys.path` |
| `helpers/probe_*.py`, `helpers/test_uf.py` | `/work/lattice-sim/sim`, `/work/sims` | `$LATSIM` / `<repo>/qec/lattice-sim/sim`, `<repo>/system` |
| `helpers/chain_soft.sh`, `helpers/watch_decoders.sh` | `cd ~/oraqle-ics-sims; -v ~/oraqle-ics-sims:/work` | repo-relative, outputs in `$OUT` (default `$ORAQLE_WORK/system`) |
| docstrings | usage lines with `/work/...` | usage lines for the new layout |

## Decoder-summary merge (correction)

`helpers/original/watch_decoders.sh` merged `ler_decoders.jsonl` + `ler_decoders_soft13.jsonl` only. The main decoder
run was then stopped by hand at 126 of 136 jobs (transcript 94019386 line 2146); the two missing d = 11
400 ns / 10 % soft patterns were run separately (`ler_decoders_soft11.jsonl`, union-find off, `--done-from` the main
run) and the published `decoders_summary.{txt,json}` merges all three files in the order main, soft13, soft11
(`merge_decoders.sh`). Re-running reproduces the published text byte for byte and the JSON to the last float digit.
