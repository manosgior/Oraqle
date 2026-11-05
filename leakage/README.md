# Readout-leakage and error-mode analysis

Code behind the failure analysis of the paper: Sec. 5.3 "Where do qubit-state discriminators fail?"
(`subsec:failures`), Table `tab:error_modes`, Fig. `fig:iq_taxonomy`, Takeaway #5 and App. "Error-mode taxonomy and
attribution" (`app:error-modes`), plus the full leakage study summarised in
[docs/LEAKAGE_REPORT.md](docs/LEAKAGE_REPORT.md). Qubit names: paper Q1..Q5 = code q0..q4.

| Where | What |
|---|---|
| `leakage/*.py` | the final analysis chain (103 scripts; 101 originals + `lkpaths.py` + one reconstructed step) |
| `leakage/iterations/` | superseded versions and probes, unchanged (see its README) |
| `leakage/vendor/networks/` | byte copy of `HybridCNN.py`, imported by `em_e2seg_oldmethod.py` |
| `leakage/docker/Dockerfile` | the `oraqle-torch` image every cluster step ran in |
| `leakage/docs/LEAKAGE_REPORT.md` | the leakage report (2026-11-04) |
| `leakage/PROVENANCE.md` | origin and md5 of every file, and every change made |
| `results/leakage/` | all small results (173 files, 37.6 MB) incl. `em_e8fix_final_numbers.json` and `paper_values.json` |
| `paper_figures/leakage/` | regenerates `tab:error_modes` and `fig:iq_taxonomy` |
| `verify/leakage.py` | checks every paper number (114 items) against `results/leakage` |
| `docs/reproduce/leakage.md` | paper item -> command -> output table |

## 1. Quick start (no dataset)

```bash
python verify/leakage.py                      # 114 paper numbers + table + re-run of the assembly chain (~5 s)
python verify/leakage.py --figure             # also re-render fig:iq_taxonomy and compare with the paper PDF
python paper_figures/leakage/make_error_modes_table.py   # -> $ORAQLE_OUT/leakage/tab_error_modes.tex
python paper_figures/leakage/make_iq_taxonomy.py         # -> $ORAQLE_OUT/leakage/iq_error_gallery_q5_v2.pdf
```
Needs numpy (verify), plus matplotlib and seaborn (figure). Tested with
`Discriminators/.venv/bin/python` (numpy 2.4.4, matplotlib 3.10.9, seaborn 0.13.2) and inside `oraqle-torch`.

## 2. Paths

Every location goes through [`lkpaths.py`](lkpaths.py), which builds on `../oraqle_env.py`. The original scripts
used docker bind mounts (`/data`, `/feat`, `/reports`, `/soft`, `/models`, `/work`, `/out`, `/lk`, `/m3`, `/e4`,
...) or absolute laptop paths; these are now:

| name | default | env override | was |
|---|---|---|---|
| `RAW_TRAIN`, `RAW_TEST` | `$ORAQLE_DATA/DRaw_C_{Tr_v0-001,Te_v0-002}` | `ORAQLE_DATA` | `/data/...` |
| `LK` | `$ORAQLE_WORK/leakage` | `ORAQLE_LEAK_WORK` | graham `~/oraqle-ics-sims/leakage` (`/lk`) |
| `FEAT` | `LK/features` | `ORAQLE_LEAK_FEAT` | `/feat`, `/lk/features`, `/out` of `x_*` |
| `stage(s)` | `LK/<s>` (e.g. `em/e4recon`, `m3time`, `f`) | - | `/work` of that stage's container |
| `REPORTS` | `results/discriminators` | `ORAQLE_REPORTS` | `~/oraqle_reports` (`/reports`, `/rep`, `/r`) |
| `MODELS` | `checkpoints` | `ORAQLE_MODELS` | `~/oraqle_models` (`/models`) |
| `SOFT` | first of `results/system/soft`, ... holding `cnn_soft_1000ns.npz` | `ORAQLE_SOFT` | `oraqle-ics/data/sims/soft`, byte copies in `f/in`, `em/e5vrepro/in` (`/soft`, `/work/in`) |
| `RES` | `results/leakage` | `ORAQLE_LEAK_RESULTS` | `oraqle-ics/data/leakage` |
| `DISCRIMINATORS` | `discriminators` | `ORAQLE_DISCRIMINATORS` | `PYTHONPATH=~/Oraqle/Discriminators` (e3herq runs) |
| `FIGS` | `$ORAQLE_OUT/leakage` | `ORAQLE_LEAK_OUT` | `oraqle-ics/plots` |

`python leakage/lkpaths.py` prints the resolved paths. The local assembly scripts read and write `RES` in place
(as they did in `data/leakage`): to re-run them without touching the committed files, point
`ORAQLE_LEAK_RESULTS` at a copy (this is what `verify/leakage.py` does).

## 3. Running with the dataset

Hardware used: graham, AMD EPYC 7713P (64 cores), 991 GB RAM, CPU only; a few steps ran on steve (same NFS
home) in `oraqle-torch-gpu`, which has the same Python stack (CUDA was not used). Peak memory was not recorded:
the `em_e3herq_regen_*` runs hold all 1.6 M traces (12.8 GB as float64 at 1 us) plus demodulated copies,
`em_e5vrepro_analyze.py` loads 6.4 GB of bins, most other steps memory-map the 3.2 GB feature set.
Intermediates: ~21 GB (features 3.2 GB, `em/e5vrepro` 12.5 GB, the rest < 2 GB per stage).

Generic recipe (repository read-only, outputs under `$WORK`):
```bash
docker build -t oraqle-torch leakage/docker
DRUN="docker run --rm --user $(id -u):$(id -g) -v $PWD:/repo:ro -v $ORAQLE_DATA:/data:ro -v $WORK:/work \
      -e ORAQLE_DATA=/data -e ORAQLE_WORK=/work -e ORAQLE_OUT=/work/out -e PYTHONDONTWRITEBYTECODE=1 -w /work"
$DRUN oraqle-torch python /repo/leakage/x_calib.py
$DRUN oraqle-torch python /repo/leakage/x_build_features.py --workers 24
```
Scripts that take an output directory as an argument (`m1tail_v2_*`, `m2mix_*`, `m4conf_*`, `vphysics_*`, ...)
expect it to exist, as the docker bind mounts used to provide it: `mkdir -p` it first. Scripts with built-in
defaults create their stage directory themselves.

Without docker: any Python 3.11 with numpy 2.1, scipy, scikit-learn, h5py, torch 2.5 (CPU) works the same way:
`ORAQLE_DATA=... ORAQLE_WORK=... python leakage/<script>.py`. The `REPORTS`, `MODELS` and `SOFT` inputs are
committed elsewhere in the repository (owned by `discriminators/` and the QEC/system part); override with
`ORAQLE_REPORTS`, `ORAQLE_MODELS`, `ORAQLE_SOFT` if they live somewhere else.

The column "published run" lists the arguments/environment of the run behind the committed results (from the
container commands, 2026-11-04/05 UTC); unlisted arguments are the script defaults.

## 4. Pipeline (DAG)

External inputs: raw traces (`RAW_*`, not distributed); `REPORTS/preds_MCMitCNN_{pb,tb}.npy` (canonical 1 us
MCMit-CNN, hard predictions + true bits, from `discriminators/runners/_xtalk_cnn.py`);
`SOFT/cnn_soft_{1000,600,400}ns.npz` (MCMit-CNN soft outputs, `sims/soft_readout_cnn.py` of the QEC part);
`MODELS/HERQULES_len{200,500}.pth` + `REPORTS/rmf_len{200,500}.pkl` (per-trace HERQULES checkpoint of
`_train_and_analyze_errors.py`, 2026-06-02); `MODELS/HybridCNN_len{200,500}.pth` +
`REPORTS/herqules_error_modes_len*_hybridcnn_*.csv` (only for the replica of the old, superseded procedure);
`DISCRIMINATORS/{trainers,networks,matched_filter.py}` (HERQULES code); `REPORTS/iq_error_gallery_q5.npz`.

Arrows: `a -> b` = b reads a. `[R]` = file committed in `results/leakage`. Runtime = wall time of the published
run (graham/steve, under load) unless marked *smoke* (re-measured 2026-11-05 in the new layout).

### S0 Shared feature set (raw -> per-shot 100 ns bins)
| script | inputs | outputs | runtime | published run |
|---|---|---|---|---|
| `x_calib.py` | RAW_TRAIN | `FEAT/calib.json` [R `x_calib.json`] | 56 s *smoke* | `x_calib.py /out` |
| `x_build_features.py` | RAW_*, calib.json | `FEAT/{feat,featbox,feat1us,y,idx}_{train,test}.npy` (3.2 GB), `build_meta.json` [R `x_build_meta.json`] | 41 s *smoke* (24 workers) | `--workers 24 --chunk 4000` |
| `x_crosscheck.py` (imports `x_colleague_prep_verbatim.py`) | RAW_*, features | `crosscheck.json` [R `x_crosscheck.json`] | < 1 min | `x_crosscheck.py /out` |
| `x_sanity.py`, `x_xtalk_diag.py` | features | `sanity.json`, `x_hist.npz`, `xtalk.json` [R `x_sanity.json`, `x_hist.npz`, `x_xtalk.json`] | < 2 min | `... /out` |
| `x_plot_iq.py` (local) | [R] `x_sanity.json`, `x_hist.npz` | [R] `x_iq_q{0..4}.png` | s | `x_plot_iq.py <RES>` |

Feature layout: `results/leakage/x_features_README.md`.

### S1 Temporal HMM (m3time) -> e1hmm, LEAKAGE_REPORT
| script | inputs | outputs | runtime | published run |
|---|---|---|---|---|
| `m3time_hmm.py` | features | `LK/m3time/out/{feat_xt1,featbox_xt1,feat_xt0}/m3time_q*.json` (+ arrays) | ~30 min per run (1763 s) | `--kind feat --xt 1 --out .../out/feat_xt1 --iters 300 --iters_test 100` (and `featbox 1`, `feat 0`) |
| `m3time_reanalyze.py` (imports `m3time_hmm`) | out/<run> | `out_re/<run>/m3time_re_q*.json` [R `m3time_re_<run>_q*.json`], `..._arrays.npz` | ~70 s per qubit | `--run .../out/<run> --out .../out_re/<run> --nnull 200000` |
| `m3time_joint.py`, `m3time_xchk.py`, `m3time_xchk2.py` | out_re, `FEAT/y_test.npy` | [R] `m3time_joint_*.json`, `m3time_xchk*_feat_xt1.json`, `m3time_re_cooccurrence_*.json` | < 1 min | `--run .../out_re/feat_xt1` |
| `m3time_examples_re.py` | out_re, features, RAW_TEST | [R] `m3time_re_examples_raw.npz` (100 example traces) | < 1 min | `--run .../out_re/feat_xt1` |
| `m3time_summary.py`, `m3time_plot_re.py` (local) | [R] m3time JSON | [R] `m3time_summary.json`, `m3time_*.png` | s | - |

### S2 HMM-based attribution (e1hmm) -> e4recon_assemble
| script | inputs | outputs | runtime | published run |
|---|---|---|---|---|
| `em_e1hmm_noisecheck.py` | features, m3time/out | `LK/em/e1hmm/noisecheck/em_e1hmm_noisecheck.json` | < 1 min | - |
| `em_e1hmm_attrib_v3.py` (also a library for v4/v6) | features, m3time, SOFT, REPORTS, noisecheck | `em/e1hmm/full_v3/` (`shared_inputs.npz`, ...) | 108 s | `--nsim 6000 --ninj 250 --nboot 300 --kinds featbox,feat --qubits 0,1,2,3,4 --procs 10 --out .../full_v3` |
| `em_e1hmm_attrib_v6.py` (imports v3, v4) | full_v3 (`--shared`), noisecheck | `em/e1hmm/full_v6/em_e1hmm_v6_results.json` (109 MB), `..._allshots_*.npz` | 235 s | `--nsim 6000 --ninj 60 --nboot 300 --kinds featbox,feat --procs 10 --out .../full_v6` |
| `em_e1hmm_oldproc.py` | RAW_*, REPORTS (preds, hybridcnn CSVs), MODELS (HybridCNN), SOFT | `em/e1hmm/oldproc/em_e1hmm_oldproc.json` | ~2 min | `--procs 24 --out .../oldproc` |
| `em_e1hmm_summarize.py` (local) | v6 results, oldproc json | [R] `em_e1hmm_results.json`, `em_e1hmm_summary.md` | s | `<v6_results.json> <oldproc.json> <RES>/em_e1hmm_results.json <RES>/em_e1hmm_summary.md` |

### S3 Segment-based attribution (e2seg) and L-flag cross-tab (f)
| script | inputs | outputs | runtime | published run |
|---|---|---|---|---|
| `em_e2seg_attr_v3.py` | features, REPORTS preds, SOFT | `LK/em/e2seg/out/em_e2seg_v3_raw_main.json` [R], `em_e2seg_v3_shots_*.npz` | 240 s; 288 s *smoke* | `E2SEG_TAG=_main E2SEG_NPROC=25` |
| `em_e2seg_attr_v2.py` (v2, bug kept for the v2-vs-v3 diagnostic) | same | `em_e2seg_v2_raw_main.json` [R] | ~5 min | `E2SEG_TAG=_main E2SEG_NB=16384 E2SEG_NCARRIER=2000 E2SEG_NPROC=25` |
| `em_e2seg_oldmethod.py` (+ `vendor/networks/HybridCNN.py`) | RAW_*, REPORTS, SOFT, MODELS | `em_e2seg_oldmethod.json` [R] | 159 s | - |
| `em_e2seg_oldmethod_l200scan.py` | as above | `em_e2seg_oldmethod_l200scan.json` [R] | ~6 min | - |
| `f_cnn_leak_xtab.py` | features, SOFT, REPORTS preds | `LK/f/out/f_cnn_leak_xtab.json` [R] | 23 s *smoke* | `OMP_NUM_THREADS=8` |
| `em_e2seg_assemble.py` (local) | [R] v3/v2 raw, oldmethod, l200scan, f_cnn_leak_xtab | [R] `em_e2seg_results.json` | < 1 s | - |

Output-file names: run e2seg with `-e E2SEG_TAG=_main` to get `..._raw_main.json` (default tag is empty).

### S4 HERQULES per-shot predictions (e3herq)
Each run needs `DISCRIMINATORS` (added to `sys.path`), the raw data, `HERQULES_TRACE_LENGTH` (samples) and a fresh
working directory (HERQULES_original writes `accuracy/ stats/ logs/` into the cwd and `$HOME`).
| script | inputs | outputs | runtime | published run |
|---|---|---|---|---|
| `em_e3herq_regen_demux.py` ("A": replica of `_herqules_length_both.py`) | RAW_*, discriminators | `LK/em/e3herq/out/em_e3herq_A_{1000,400,600}ns.npz` (~82 MB each), `A_summary.json` | ~42 min | cwd/HOME `.../run_A`, `HERQULES_TRACE_LENGTH=500 --lengths 1000,400,600 --threads 8` |
| `em_e3herq_regen_scrambled.py` ("B": demux pipeline of `_train_and_analyze_errors.py`) | RAW_*, discriminators | `em_e3herq_B_{400,600,1000}ns.npz` | 10-20 min each | cwd `.../run_B{200,300,500}`, `HERQULES_TRACE_LENGTH=200/300/500 --threads 8` |
| `em_e3herq_regen_pertrace_ckpt.py` ("C": per-trace checkpoint, inference only) | RAW_*, MODELS `HERQULES_len{200,500}.pth`, REPORTS `rmf_len*.pkl` | `em_e3herq_C_{400,1000}ns.npz` | ~5 min each | cwd `.../run_C{200,500}`, `HERQULES_TRACE_LENGTH=200/500 --threads 8` |
| `em_e3herq_ycheck.py` | RAW_TEST, REPORTS, e3herq/out | stdout (label alignment check) | s | - |
| `em_e3herq_assemble_v3.py` | e3herq/out, REPORTS preds, SOFT | `<out>/em_e3herq_preds_{1us,600ns,400ns}.npz` (107-129 MB, **not committed**), `em_e3herq_summary_v3.json` [R] | 55 s *smoke* | `em_e3herq_assemble_v3.py LK/em/e3herq/out LK/em/e3herq/assembled` |

### S5 Final attribution (e4recon) -> paper
| script | inputs | outputs | runtime | published run |
|---|---|---|---|---|
| `em_e4recon_make_herq_in.py` (**reconstructed**, see docstring) | `em_e3herq_preds_{1us,400ns}.npz` | `LK/em/e4recon/in/em_e4recon_herq_in.npz` | 8 s *smoke* | - |
| `em_e4recon_glrt.py` (imports `em_e2seg_attr_v3`) | features, REPORTS preds, SOFT, herq_in | `em/e4recon/out/em_e4recon_{allshots_*,latecore,z}.npz`, `glrt_raw.json` | 216 s | - |
| `em_e4recon_analyze.py` | glrt outputs, `FEAT/y_test.npy`, REPORTS, SOFT, herq_in, `em/e1hmm/full_v6/*allshots_featbox*` (cross-check) | `em/e4recon/out/em_e4recon_analysis.json` [R] | ~3 min | - |
| `em_e4recon_assemble.py` (local) | [R] analysis, e1hmm_results, e2seg_results, e3herq_summary_v3 | [R] `em_e4recon_final.json` | < 1 s | - |

### S6 Adversarial verifiers of the attribution
| script | inputs | outputs | runtime | published run |
|---|---|---|---|---|
| `em_e5vstat_check.py` | features, REPORTS, SOFT, `em/e4recon/{in,out}` | `em/e5vstat/out/em_e5vstat_check.json` [R], `pershot_q*.npz` | 23 s *smoke* (5 procs) | - |
| `em_e5vstat_lda.py`, `em_e5vstat_q3core.py` | check pershot, e4recon out | [R] `em_e5vstat_lda.json`, `em_e5vstat_q3core.json` | 13 s, 6 s *smoke* | - |
| `em_e5vstat_summarize.py` (local) | [R] check, lda | [R] `em_e5vstat_summary.json` | < 1 s | - |
| `em_e5vrepro_ifscan.py`, `em_e5vrepro_demod.py` | RAW_* | `em/e5vrepro/out/em_e5vrepro_{ifscan,if,if2}.json` [R `if2`], `bins{,2}_{train,test}.npy` (12.5 GB) | ~1 min (16 procs) | `em_e5vrepro_demod.py 16` |
| `em_e5vrepro_analyze.py` | bins2, REPORTS preds, SOFT (`/work/in` copies) | `em_e5vrepro_raw_v2.json` [R], `pershot_v2.npz` | 321 s | `OMP_NUM_THREADS=32` |
| `em_e5vrepro_diag_{partners,q3relax,xtalk}.py` (exec parts of analyze) | bins2, pershot_v2 | [R] `em_e5vrepro_diag_*.json` | 1-2 min each | - |
| `em_e5vrepro_assemble.py` (local) | [R] raw_v2, e4recon_final, diag_partners, diag_xtalk | [R] `em_e5vrepro_results.json` | < 1 s | - |

### S7 Final numbers -> paper text, table, figure
| script | inputs | outputs |
|---|---|---|
| `em_e6edit_finalize.py` (local) | [R] e4recon_final, e5vstat_summary, e5vstat_check, e5vrepro_results, diag_partners | [R] `em_e6edit_final_numbers.json` (`.md` is a hand-written digest) |
| `em_e8fix_finalize.py` (local) | [R] e6edit numbers, e4recon_final, `m1tail_v2_percfg_test.json` | [R] `em_e8fix_final_numbers.json` (with `paper_text_map`) |
| `paper_figures/leakage/make_error_modes_table.py` | [R] e8fix, e4recon_final | `tab:error_modes` LaTeX |
| `em_e6edit_iq_gallery.py` / `paper_figures/leakage/make_iq_taxonomy.py` | REPORTS `iq_error_gallery_q5.npz`, `discriminators/runners/iq_error_gallery.py` | `fig:iq_taxonomy` PDF |

### S8 Leakage study (LEAKAGE_REPORT.md; m1tail is also an input of e8fix)
| family | scripts (published arguments) | committed outputs |
|---|---|---|
| m1tail (one-sided off-axis tail; `m1tail_v2_percfg` 45 s *smoke*, byte-identical) | `m1tail_v2.py --out LK/m1tail/out_v2 --split test --traj --models` (and `--split train --traj`; `--out .../out_v2_sl --split test --sliding`); `m1tail_v2_{percfg,cooc,models,xtab}.py FEAT LK/m1tail/out_v2_<x> test`; local `m1tail_v2_summary.py RES`, `m1tail_v2_plot.py RES` | `m1tail_v2_*_{test,train}.json`, `m1tail_v2_store_test.npz`, `m1tail_summary.json`, `m1tail_*.png` |
| m2mix (2-D mixture fits) | `m2mix_prep.py FEAT LK/m2mix/out`; `m2mix_fit2.py LK/m2mix/out LK/m2mix/fit2 <split:q:win ...>` (imports `m2mix_core2`); `m2mix_boot2.py .../out .../fit2 .../boot2 <tasks>`; `m2mix_coinc2.py`, `m2mix_traj2.py .../out .../fit2 FEAT .../traj2`; local `m2mix_feature2.py`, `m2mix_summarize2.py`, `m2mix_plot2.py` | `m2mix_fit2/`, `m2mix_coinc2.json`, `m2mix_traj2_summary.json`, `m2mix_summary.{json,csv}`, `m2mix_feature_table.*`, `m2mix_q*.png` |
| m4conf (confounder screens) | `m4conf_analysis.py FEAT DATA LK/m4conf/out`, `m4conf_rawstats.py DATA FEAT out`, `m4conf_rawprobe.py DATA FEAT out`, `m4conf_followup.py out`, `m4conf_followup2.py FEAT out`, `m4conf_final.py FEAT out`, `m4conf_dccheck.py FEAT out`, `m4conf_pairflip.py out`; local `m4conf_summary.py RES`, `m4conf_plots.py <src> RES` | `m4conf_*.json`, `m4conf_*.png` |
| s (surveys) | `s1local_survey_numbers.py` (needs `ORAQLE_PRECLASSIFIER_PKL`), `s2cluster_label_stats.py --data DATA --out LK/s2cluster`, `s2cluster_recompute_perp.py`, `s2cluster_xtalk_probe.py`, `s2cluster_cnn_err_vs_label.py` (need `DATA/three_level_labels.hdf5`), `s3lit_*.py` | `s1local_*`, `s2cluster_*.json` |
| r (reconciliation) | `r_spotcheck.py`, `r_spot2.py` (need m3time out_re arrays and `LK/m4conf/out/m4conf_pershot.npz`); local `r_build_summary.py` | `r_spotcheck.json`, `r_spot2.json`, `r_leakage_summary.{json,md}` |
| v (verifiers) | `vstatistics_{main,line}.py`; `vphysics_raw{,2}.py`, local `vphysics_{circle,circle2,ringup,plot,dispersive_locus,implied_level}.py`; `vreproduction_demod.py`, `vreproduction_estimate.py`, `vreproduction_followup{,2,3}.py` | `vstatistics_*.json`, `vphysics_*`, `vreproduction_*.json` |

Local scripts read their inputs from `RES` (or the directory given as argument) and write next to them.

## 5. What is verified

* **Tier 1 (no dataset, `verify/leakage.py`)**: 112 of 114 paper numbers reproduce at the printed precision from
  the upstream result files; 2 are documented rounding discrepancies (Q5 overlap 8.7 vs printed 8.8; App. "1.7%"
  for a 1.68-1.75 bracket). The 10 dataset-free assembly/summary scripts reproduce their committed JSON byte for
  byte; the LaTeX table equals the paper; the figure re-renders pixel-identical to the paper PDF. The five local
  `vphysics_*` scripts also reproduce their JSON byte for byte (checked once, not part of verify: ringup takes
  ~4 min).
* **Tier 2 (graham, new layout, 2026-11-05)**: `x_calib`, `x_build_features`, `f_cnn_leak_xtab`,
  `em_e2seg_attr_v3`, `em_e3herq_assemble_v3`, `em_e4recon_make_herq_in`, `em_e4recon_glrt`,
  `em_e4recon_analyze`, `em_e5vstat_{check,lda,q3core}` and `m1tail_v2_percfg` re-run from this layout reproduce
  the original outputs byte for byte (features, all per-shot npz and every JSON). Together with tier 1 this
  covers raw traces -> features -> attribution -> `em_e4recon_final.json` / `em_e5vstat_summary.json` ->
  `em_e8fix_final_numbers.json` -> paper. Not re-run: the HMM (m3time, e1hmm), the e3herq HERQULES regenerations,
  the e5vrepro reproduction and the m2mix/m4conf/s/r/v families (their committed outputs are consumed). Details
  in `docs/reproduce/leakage.md`.
