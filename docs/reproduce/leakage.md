# Reproducing the error-mode / leakage results

Paper: `oraqle-ics` working tree of 2026-11-05 (HEAD e2ba1ac + the uncommitted `[audit fix]` edits). Chapters:
`ml` = `chapters/ml_discriminators.tex`, `app` = `chapters/appendix_herqules.tex`. Pipeline, paths and
docker recipe: [`leakage/README.md`](../../leakage/README.md). File origins: [`leakage/PROVENANCE.md`](../../leakage/PROVENANCE.md).

Verification tiers: **tier 1** = recomputed/regenerated without the dataset (`verify/leakage.py`);
**tier 2** = re-run from this repository layout on graham with the raw dataset, outputs compared with the
committed results (2026-11-05); **not run** = only the original run exists.

## 1. Paper items

| paper item (label, chapter:line) | command | output | needs dataset? | runtime | verified |
|---|---|---|---|---|---|
| Table `tab:error_modes` (ml:397-418) | `python paper_figures/leakage/make_error_modes_table.py` | `$ORAQLE_OUT/leakage/tab_error_modes.tex` | no | < 1 s | tier 1: identical to the paper block; all 27 cells recomputed from counts (Q5 overlap 8.7495 % -> 8.7, as printed since the fix, §3); `--stored` gives the old 8.8. Upstream numbers tier 2 (§2) |
| Fig. `fig:iq_taxonomy` (ml:342-353), `plots/iq_error_gallery_q5_v2.pdf` | `python paper_figures/leakage/make_iq_taxonomy.py` (plot step) | `$ORAQLE_OUT/leakage/iq_error_gallery_q5_v2.pdf` | no (uses `results/discriminators/iq_error_gallery_q5.npz`) | ~3 s | tier 1: pixel-identical to the paper PDF at 300 dpi (`verify/leakage.py --figure`) |
| ... its extraction step (`iq_error_gallery_q5.npz`) | `IQ_QUBIT=5 IQ_REPORTS=<dir> HERQULES_TRACE_LENGTH=500 python discriminators/runners/iq_error_gallery.py` (extract + plot; owner: discriminators) | `<dir>/iq_error_gallery_q5.npz` | yes | minutes (HERQULES demodulation of 1.6 M shots) | not run (the June-2 npz is the committed input) |
| Sec. 5.3 methodology scope (ml:340) | `python verify/leakage.py` | stdout | no | ~5 s | tier 1 |
| Sec. 5.3 results: state change / overlap, half-plane 91/95 vs 33/15 %, step 2-8 %, 400 ns overlap 40/56 %, leakage 11.5/4.7 %, 97-99 %, 3.5/6.5/11.5 % (ml:372) | `python verify/leakage.py` | stdout | no | ~5 s | tier 1 (18 numbers); attribution inputs tier 2 |
| Sec. 5.3 analysis: 81/17 %, bands 79-81 / 17-20, 1.8 %, 28 %, 1.6x, 4.4->4.5, 37->81 %, 2.5-14x, triples, 68/92 %, 69 %, 23 % (ml:424) | `python verify/leakage.py` | stdout | no | ~5 s | tier 1 (19 numbers) |
| Takeaway #5: ~80 %, <=20 %, 0.4-6 %, <=23 % (ml:430) | `python verify/leakage.py` | stdout | no | ~5 s | tier 1 |
| App. `app:error-modes` scope (app:346-351) | `python verify/leakage.py` | stdout | no | ~5 s | tier 1 |
| App. attribution procedure: 16 configs, z>3, 100 ns, alpha=0.01, 225-775 ns, 0.44, 16-28 %, gate 0.6-1.3 / >=5 %, excluded tests, 1.5 widths, 1.5 pp (app:358-384) | `python verify/leakage.py` | stdout | no | ~5 s | tier 1; parameters read from the code (`ALPHA0`, `ALPHAS[ALPHA_IDX]`, `LATE_R`) and the gate rule re-applied to every test |
| App. validation: <=0.03 pp, 1.4 pp, 0.5 sigma, 79-81 %, 90-95 vs 6-33 %, s=89 % (90 %), 24 vs 54, 4-8 % / 1.7-1.8 %, obs/exp 1.0 / 5.4 (app:387-405) | `python verify/leakage.py` | stdout | no | ~5 s | tier 1 ("<=0.03 pp" is quoted from a robustness note, not recomputable) |
| App. leakage paragraph: 11.5, 0.8, 4.7, 0.8 %, ~90 %, 1.6 %, Q4 n/m, 13 %, 3.1 %, 99.3 % (app:407-412) | `python verify/leakage.py` | stdout | no | ~5 s | tier 1 |
| Final numbers file `em_e8fix_final_numbers.json` (source of all of the above) | `ORAQLE_LEAK_RESULTS=<copy of results/leakage> python leakage/em_e8fix_finalize.py` (after `em_e6edit_finalize.py` etc.) | `<copy>/em_e8fix_final_numbers.json` | no | < 1 s | tier 1: byte-identical (`verify/leakage.py` regen) |
| `discussion.tex` "mostly qubit state changes and SNR overlap" | - | qualitative | - | - | consistent with the above (not owned here) |

`verify/leakage.py` result on 2026-11-05 (paper with the `[audit fix]` edits): 114/114 numbers at the printed
precision, 0 documented discrepancies, table identical, 10/10 regenerated files byte-identical, all 114 line references valid (`--tex`).
Exit status 0 (1 with `--strict`).

## 2. Pipeline steps

All commands as in `leakage/README.md` §3 (`$DRUN oraqle-torch python /repo/leakage/<script>`). Runtimes:
graham, measured in the smoke test unless marked (orig) = original run.

| step | command (new layout) | output | dataset? | runtime | verified |
|---|---|---|---|---|---|
| IF calibration | `x_calib.py` | `$ORAQLE_WORK/leakage/features/calib.json` | yes | 56 s | **tier 2**: byte-identical to the original and to `results/leakage/x_calib.json` |
| feature set (1.6 M shots) | `x_build_features.py --workers 24` | `features/*.npy` (3.2 GB) | yes | 41 s | **tier 2**: all 10 arrays md5-identical to graham `~/oraqle-ics-sims/leakage/features` |
| L-flag cross-tab | `f_cnn_leak_xtab.py` | `f/out/f_cnn_leak_xtab.json` | features | 23 s | **tier 2**: byte-identical |
| segment attribution (e2seg v3) | `-e E2SEG_TAG=_main em_e2seg_attr_v3.py` | `em/e2seg/out/em_e2seg_v3_raw_main.json` + 25 npz | features | 4 min 48 s | **tier 2**: JSON and all 25 npz byte-identical (run with the default empty tag) |
| HERQULES predictions assembly | `em_e3herq_assemble_v3.py <e3herq/out> <e3herq/assembled>` | `em_e3herq_preds_*.npz` (365 MB), `em_e3herq_summary_v3.json` | no (needs the e3herq run outputs) | 55 s | **tier 2** (inputs: original graham e3herq/out): all 4 files byte-identical |
| herq_in (reconstructed step) | `em_e4recon_make_herq_in.py` | `em/e4recon/in/em_e4recon_herq_in.npz` | no | 8 s | **tier 2**: all 10 arrays identical to the original |
| step detector on all shots | `em_e4recon_glrt.py` | `em/e4recon/out/*` | features | 3 min 43 s | **tier 2**: `glrt_raw.json` and 17 npz byte-identical |
| final attribution | `em_e4recon_analyze.py` (needs `em/e1hmm/full_v6` for a cross-check field) | `em/e4recon/out/em_e4recon_analysis.json` | features | 20 s | **tier 2**: byte-identical to `results/leakage` |
| e5vstat verifier | `em_e5vstat_check.py`, `em_e5vstat_lda.py`, `em_e5vstat_q3core.py` | `em/e5vstat/out/em_e5vstat_{check,lda,q3core}.json` | features | 23 s, 13 s, 6 s | **tier 2**: all three byte-identical |
| per-configuration leakage (input of e8fix) | `m1tail_v2_percfg.py $ORAQLE_WORK/leakage/features <outdir> test` (`mkdir -p <outdir>` first) | `m1tail_v2_percfg_test.json` | features | 45 s | **tier 2**: byte-identical |
| assembly / summaries (10 scripts) | `verify/leakage.py` (regen) | `results/leakage/*.json` | no | 5 s | **tier 1**: byte-identical |
| vphysics local scripts (5) | `vphysics_{circle,circle2,ringup,dispersive_locus,implied_level}.py` | JSON | no | ~4 min (ringup) | tier 1 (once): byte-identical |
| HMM (m3time), e1hmm v3/v6, oldproc | see README S1/S2 | `em_e1hmm_results.json` (via summarize) | features | ~30 min + 5 min (orig) | not run (outputs consumed; e4recon cross-check used the original full_v6) |
| e3herq A/B/C regeneration | see README S4 | `em/e3herq/out/*.npz` | yes + discriminators/ + checkpoints | ~1.5 h (orig) | not run (assembly from their outputs: tier 2) |
| e5vrepro (independent reproduction) | see README S6 | `em_e5vrepro_*.json` | yes | ~10 min (orig: demod ~1 min, analyze 321 s, 3 diagnostics) | not run (assembly tier 1) |
| other m1tail steps, m2mix, m4conf, s, r, v families | see README S8 | LEAKAGE_REPORT inputs | yes / features | min-h (orig) | not run (summaries tier 1) |

## 3. Documented discrepancies (paper text vs. results): none open

Both rounding-level discrepancies of the first audit were fixed in the paper on 2026-11-05 (`[audit fix]`):

1. **ml:412, Table `tab:error_modes`, Q5 overlap**: printed 8.8 (double rounding of the stored 8.75), now 8.7
   (2901.5 / 33162 errors = 8.7495 %). The table script now recomputes every share from counts by default.
2. **app:402, Q3 detector share**: printed "1.7 %", now "1.7--1.8 %" (variants 1.752, 1.72, 1.68 %).

(Separately, the table generator recomputes the error-rate column from counts: re-rounding the stored 3-decimal
5.315 % of Q4 would print 5.32, while the exact 5.3146 % gives the paper's correct 5.31.)

## 4. Caveats

* **HERQULES in Sec. 5.3 / App.** is the per-trace checkpoint `HERQULES_len500.pth` + `rmf_len500.pkl`
  (`_train_and_analyze_errors.py`, 2026-06-02), re-inferred by `em_e3herq_regen_pertrace_ckpt.py`. The
  `tab:fidelity` HERQULES row comes from `_herqules_length_both.py`, which retrains and saves no checkpoint.
  Per-qubit fidelity at 1 us differs by at most 0.23 pp from the unrounded CSV (Q4: 0.9414 vs 0.9437; 0.26 pp from
  the printed 0.944), F5Q 0.904 vs 0.9046. Since the 2026-11-05 fix the paper says so (ml:340 "a per-trace HERQULES checkpoint",
  app:350 "a separately trained per-trace HERQULES checkpoint (per-qubit fidelity within 0.26 pp of Table
  tab:fidelity)").
* **MCMit-CNN**: 1 us = `preds_MCMitCNN_pb.npy` (`_xtalk_cnn.py`, a separate training of the `tab:fidelity`
  recipe; per-qubit fidelity within 1e-4 of the table); 400/600 ns = `cnn_soft_{400,600}ns.npz` of
  `sims/soft_readout_cnn.py` (same recipe, owned by the QEC/system part).
* "Q1 peaks at 0.44" (app:372): the maxima are 0.445 and 0.437; 0.445 sits exactly on the rounding boundary.
* Leakage values are k=3 mirror-corrected lower bounds; Q4 leakage and Q2 state change/overlap are n/m by
  construction.
* `em_e4recon_make_herq_in.py` is a reconstruction of an inline step (verified byte-identical output).
* The `.md` digests in `results/leakage` are hand-written; `LEAKAGE_REPORT.md` numbers predate the final
  attribution (the paper uses `em_e8fix_final_numbers.json`).
* `results/leakage/m3time_re_examples_raw.npz` holds 100 demodulated example traces (20 per qubit, 461 kB):
  derived data, not raw ADC records; drop it if even that is not acceptable for redistribution (only
  `m3time_plot_re.py` reads it).

## 5. Smoke-test log (tier 2, 2026-11-05 UTC, graham)

Work dir `~/oraqle-unify-smoke/leakage/` (`repo/` = rsync of `oraqle_env.py`, `leakage/`, `results/leakage/`,
`paper_figures/leakage/`, `verify/leakage.py`; `work/` = outputs; `logs/`). Image `oraqle-torch:latest`
(f40dbb670f94). Recipe (`env.sh`):
```bash
DRUN="docker run --rm --user $(id -u):$(id -g) -v $HOME/oraqle-unify-smoke/leakage/repo:/repo:ro \
 -v $HOME/qubit_readout_klinq/data/five_qubit_data:/data:ro \
 -v $HOME/oraqle_reports:/reports:ro -v $HOME/oraqle-ics-sims/data/soft:/soft:ro \
 -v $HOME/oraqle-unify-smoke/leakage/work:/work \
 -e ORAQLE_DATA=/data -e ORAQLE_WORK=/work -e ORAQLE_OUT=/work/out \
 -e ORAQLE_REPORTS=/reports -e ORAQLE_SOFT=/soft -e PYTHONDONTWRITEBYTECODE=1 -w /work"
$DRUN --name unify-leak-calib    oraqle-torch:latest python /repo/leakage/x_calib.py               # 13:27Z, 56 s
$DRUN --name unify-leak-features oraqle-torch:latest python /repo/leakage/x_build_features.py --workers 24  # 41 s
$DRUN --name unify-leak-fxtab    oraqle-torch:latest python /repo/leakage/f_cnn_leak_xtab.py       # 23 s
$DRUN --name unify-leak-e2seg    oraqle-torch:latest python /repo/leakage/em_e2seg_attr_v3.py      # 4 min 48 s
$DRUN -v $HOME/oraqle-ics-sims/leakage/em/e3herq/out:/e3out:ro --name unify-leak-e3asm oraqle-torch:latest \
      python /repo/leakage/em_e3herq_assemble_v3.py /e3out /work/leakage/em/e3herq/assembled     # 55 s
$DRUN --name unify-leak-herqin   oraqle-torch:latest python /repo/leakage/em_e4recon_make_herq_in.py   # 8 s
$DRUN --name unify-leak-glrt     oraqle-torch:latest python /repo/leakage/em_e4recon_glrt.py       # 3 min 43 s
$DRUN -v $HOME/oraqle-ics-sims/leakage/em/e1hmm/full_v6:/work/leakage/em/e1hmm/full_v6:ro \
      --name unify-leak-e4an oraqle-torch:latest python /repo/leakage/em_e4recon_analyze.py      # 20 s
$DRUN --name unify-leak-e5check  oraqle-torch:latest python /repo/leakage/em_e5vstat_check.py      # 23 s
$DRUN --name unify-leak-e5lda    oraqle-torch:latest python /repo/leakage/em_e5vstat_lda.py        # 13 s
$DRUN --name unify-leak-e5q3     oraqle-torch:latest python /repo/leakage/em_e5vstat_q3core.py     # 6 s
mkdir -p work/leakage/m1tail/out_v2_percfg
$DRUN --name unify-leak-m1percfg oraqle-torch:latest python /repo/leakage/m1tail_v2_percfg.py \
      /work/leakage/features /work/leakage/m1tail/out_v2_percfg test                          # 45 s
```
`REPORTS`/`SOFT` were pointed at the graham originals (byte-identical to `results/discriminators` and the soft
npz; md5 checked) because those directories belong to other parts of the repository. All 102 leakage scripts
compile under the image's Python 3.11.11. Outcomes: see §2. The first `em_e3herq_assemble_v3.py` attempt failed
because the output directory did not exist (the script had always written into an existing directory); it now
creates it. `m1tail_v2_percfg.py` likewise needs its output directory to exist (`mkdir -p`, documented; the
script was not changed). The large smoke outputs (features, per-shot npz, preds) were deleted afterwards; logs and
JSON outputs remain in `~/oraqle-unify-smoke/leakage/`.
