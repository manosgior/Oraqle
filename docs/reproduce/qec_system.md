# Reproducing §6, §7, the discussion and the QEC appendix

Scope: qec_impact.tex (§6), system_bottleneck.tex (§7), discussion.tex, and the QEC parts of appendix_herqules.tex
(App. "ECCentric vs. lattice-sim", "Decoherence-only", "Fixed-fidelity"). Paper source: oraqle-ics working tree of
2026-11-05 (HEAD d7cd4e9 + uncommitted chapter edits); `chapter:line` refers to it. Commands run from the repository
root. Outputs go to `$ORAQLE_OUT` (default `out/`) and `$ORAQLE_WORK` (default `work/`), never into `results/`.
`$COMMON` in docker commands is the mount/env set of `docker/README.md` (repo at /repo read-only, `work/`, `out/`).

## 1. Quick start (no dataset, no docker, ~45 s)

```bash
python verify/qec_system.py --figures      # any Python with numpy + matplotlib; poppler-utils for --figures
make -C paper_figures/system all           # the 5 §7 figures + tab:sys_reaction -> out/paper_figures/system
make -C paper_figures/qec all              # the 7 §6/appendix figures             -> out/paper_figures/qec
```
`verify/qec_system.py` re-runs the §7 analysis chain from the raw sweep files (`results/system/*.jsonl`) into
`out/verify/system`, compares every re-derived summary with the committed one, checks the 165 §7 and 32 §6/appendix
paper values (`results/{system,qec}/paper_values.json`, each with its chapter:line), the qShell transcriptions, the md5
of the CNN code the soft-readout run imports, and (with `--figures`) regenerates all 12 figures and compares them with
the paper PDFs. It exits 1 on any mismatch. Result on 2026-11-05 (laptop, Discriminators/.venv: Python 3.13,
numpy 2.4.4, matplotlib 3.10.9): **19/19 checks passed**; in `oraqle-latsim` on steve with `--stim`: 15/15.
Known content issues of the paper are printed (section 6) but do not fail the run.

With docker (images: `docker/README.md`): `system/replay_points.py` re-runs published sweep jobs and reproduces them
bit for bit; `system/run_paper_sweeps.sh <step>` re-runs the paper's sweeps; `qec/eccentric/fetch_eccentric.sh` +
`docker/eccentric/README.md` re-run the ECCentric simulations.

## 2. Status

* **§7: every figure, the table and every prose number reproduce from `results/system`.** The five figures regenerate
  with identical drawing streams (only the PDF Creator/CreationDate differ). With the laptop environment (numpy
  2.4.4) `summary.json`, `summary_soft.json` and `decoders_summary.txt` are byte-identical and
  `decoders_summary.json`/`system_cost.json` equal to rel. 1e-12; in the oraqle-latsim image (numpy 2.3.1) the fits
  differ in the last float digit (rel. ~1e-16), which `verify` tolerates (rel. 1e-9). All 165 paper values match
  within the paper's rounding.
* **§7 simulations are deterministic.** Every job is seeded from its point; re-running published grid, disc_hi,
  soft, decoder (all 7 decoders incl. union-find) and timing jobs in the original images and in images rebuilt from
  `docker/latsim` gave bit-identical shots and failures (tier 2, steve).
* **Two §7 inputs that had no producer now have one:** `summary_soft.json` (`system/summarize_soft.py`) and
  `UNIFORM_FRACTION` in `system_cost.py` (`system/derive_uniform_fraction.py`, reproduces 0.0166/0.0481/0.0774/
  0.1428/0.2448 exactly with stim 1.16.0). Both were recovered from the transcript of the session that made them.
* **The decoder-summary merge needs three files** (`system/merge_decoders.sh`); the cluster helper merged two.
* **§6 and the QEC appendix: all 7 figures regenerate pixel-identically** from `results/qec`, including the three
  `ler_pair_*.pdf` whose script (`plot_paper_pairs.py`) existed only in a Claude transcript and was recovered;
  32/32 paper values match. ECCentric is pinned to public commits (TUM-DSE/eccentric @da7164c, qiskit-qec @7136727,
  more-bacon-less-threshold @df98313) and its drivers run from the new layout (tier 2 smoke points agree with the
  paper CSVs within sampling error; ECCentric sampling is unseeded).
* **External numbers** (Micro Blossom, Helios, GPU, qShell, CXL-Quantum, Acharya, Litinski/Ross-Selinger, ibm_boston)
  are in `results/external` with source, commit/citation and check status; qShell numbers are re-extracted from the
  verbatim qShell files. Comparison qShell vs §7: `results/external/COMPARISON_qshell_vs_sec7.md` (section 7).
* **Not re-run (cost):** the full §7 sweeps (≈4 h at 180 cores for the decoders, ≈1 h for the rest), the soft-readout
  CNN retraining (5 × ~30 min on an H200, needs the dataset) and the full ECCentric sweeps (6-98 CPU-h each).

## 3. §7 (system_bottleneck.tex): figures, table and numbers

`verify` = `python verify/qec_system.py` (checks every number below against `results/system/paper_values.json`;
`python system/paper_numbers.py --check` checks the committed summaries directly). tier1 = checked locally from the
committed results; tier2 = executed from the new layout on steve (2026-11-05).

| Paper item (label, chapter:line) | Command | Output | Needs dataset? | Image | Runtime | Verified |
|---|---|---|---|---|---|---|
| fig:sys_spacetime, plots/readout_spacetime.pdf (system_bottleneck:27-34) | `make -C paper_figures/system sys_spacetime` | `$ORAQLE_OUT/paper_figures/system/readout_spacetime.pdf` | no | none (numpy+matplotlib) | 2 s | tier1: drawing streams identical to the paper PDF |
| fig:sys_decoder, plots/decoder_load.pdf (:52-59) | `make -C paper_figures/system sys_decoder` | `.../decoder_load.pdf` | no | none | 1 s | tier1, identical |
| fig:sys_cost, plots/system_cost.pdf (:96-103) | `make -C paper_figures/system sys_cost` | `.../system_cost.pdf` | no | none | 1 s | tier1, identical |
| fig:sys_soft, plots/soft_readout.pdf (:141-148) | `make -C paper_figures/system sys_soft` | `.../soft_readout.pdf` | no | none | 1 s | tier1, identical |
| fig:sys_decoders, plots/decoder_choice.pdf (:168-175) | `make -C paper_figures/system sys_decoders` | `.../decoder_choice.pdf` | no | none | 1 s | tier1, identical |
| All five vs the paper PDFs | `make -C paper_figures/system all compare` | stdout, exit 1 on mismatch | no | none | 8 s | tier1 |
| tab:sys_reaction (:61-78), 20 cells | `make -C paper_figures/system sys_reaction` | `.../tab_sys_reaction.tex` | no | none | <1 s | tier1: every cell identical to the paper's LaTeX |
| §7.1 methodology numbers: Λ≈2-2.6, t_r=T+340 ns, 54-75%, d 5..15, ~1000/~200 failures, grid ranges (:9); 772 ns, 308 ns, 2.5×, 2.1 µs, d≤57, d^3.4 from d=9..21 (:13); 30-2,050 logical qubits, 1e-6, 1% (:15) | `verify` | stdout | no | none | <1 s | tier1 (19 values). The decoder latencies are external constants (results/external) |
| §7.2 prose + caption (:31, 36, 38, 40): grid Λ 4.07/2.50/2.49/1.44; CNN Λ 2.41/2.61/2.49/1.49, d 47/43/45/105; 0.62-0.65×; 6.1×; 6.2% vs 4.3%, Λ 1.79, 2.3×, 3.4×; HERQULES 1.0-1.3×; 4-12× | `verify` | stdout | no | none | <1 s | tier1 (27 values); 3 known issues (section 6) |
| §7.3 caption + prose (:56, 80, 82, 84): 15/23/46/250 cores at d=17; 1.1×, 1.7-1.9×, 5-7×; detection 10.2/11.3/15.9/33%; p≈0.7-1.2%; p99 ~1 µs; d 23-107; waiting 2-11%; 1.5-3× | `verify` | stdout | no | none | <1 s | tier1 (17 values) |
| §7.4 caption + prose (:100, 105, 107, 109): 0.68-0.72×, 0.85-1.0×; Ising-100 12.3 → 8.5 s, 0.92×; 1.19-1.25×, 4.8-5.3×; QubiCML 1.4-1.5×, 2.2-2.5×; 3e4-4e7 qubits | `verify` | stdout | no | none | <1 s | tier1 (13 values); 1 known rounding issue |
| §7.5 prose (:121, 123, 125): 1.6-2.2 and 14.9 Gb/s; V80 d≤19 / d≤21; 12-17 and 254 cards; 25% smaller; QFT-30 184-223 vs 3,721; ≤1.3×, 7-9×, 15-21×; QubiCML 2.3×, 4.0×; d=43-133 | `verify` | stdout | no | none | <1 s | tier1 (16 values) |
| §7.6 methodology, caption, prose (:139, 145, 150, 152, 154): 1.12 M shots, f grid, 8 patterns; 4×, 0.46×, 1.5-1.8×; 42%, 7.9→4.9%, →3.7%; Λ 1.50→1.87, 2.52→2.90, 2.48→2.59; ~2.0 at 20%; d 103→65, 6.5×→1.65×, 45→39; 3.2 vs 2.2, 14.3→11.4 Gb/s; 24-42% | `verify` | stdout | no | none | 1 s | tier1 (32 values); 2 known issues |
| §7.7 hypothesis, methodology, caption, prose (:164, 166, 172, 177, 179, 181): 35-65×, 0.12%; 5 variants, ≥300 failures, d 5..13, soft points; Λ 2.46-2.53 → 1.31-1.38, 0.95; d 133-159; 19-27×, 26.8 vs 23.4; q16 within 0.03; q4 0.1-0.5, 1.5-2.2×; correlated +0.16-0.29, 0.47×; flag 1.69/3.00/3.30, 0.29×; weighted Helios 2-15%, 5 rounds; unweighted Helios 2.8-2.9×/8-10×, 2.3-2.5×/12-14×, 1.2×/2.9-3.2×; MB 0.58-0.62×/0.63-0.72× | `verify` | stdout | no | none | <1 s | tier1 (41 values); 3 known issues |
| fig:regime_map (qec_impact:123-144, references §7) | hand-made TikZ in the paper source | n/a | no | n/a | n/a | n/a |

### §7 analysis chain (committed summaries from the raw sweeps)

| Result | Command | Output | Needs dataset? | Image | Runtime | Verified |
|---|---|---|---|---|---|---|
| summary.json | `python system/analyze.py --data DIR` | `DIR/summary.json` | no | none (numpy) | 1 s | tier1: byte-identical (numpy 2.4.4); tier2 (oraqle-latsim, numpy 2.3.1): last-digit float differences only |
| summary_soft.json (had no script) | `python system/summarize_soft.py --data DIR` | `DIR/summary_soft.json` | no | none | <1 s | tier1: byte-identical; tier2: last-digit float differences only |
| decoders_summary.{txt,json} (3-file merge) | `sh system/merge_decoders.sh DIR` | `DIR/decoders_summary.{txt,json}` | no | none | <1 s | tier1 + tier2: txt byte-identical; json equal to rel 1e-12 |
| system_cost.json | `python system/system_cost.py --data DIR` | `DIR/system_cost.json` | no | none | <1 s | tier1 + tier2: equal to rel 1e-9 (1e-12 locally) |
| UNIFORM_FRACTION (system_cost.py constant; had no script) | `docker run $COMMON oraqle-latsim python system/derive_uniform_fraction.py --no-write` | stdout (`results/external/uniform_fraction_d9.json` without `--no-write`) | no | oraqle-latsim (stim 1.16.0) | 3 s | tier1 (MCMit venv, stim 1.16.0) + tier2 (steve): all 5 values and the 5 d=9 detection fractions exact |
| uncertainty.json (not cited) | `python system/uncertainty.py --data DIR` | `DIR/uncertainty.json` | no | none | ~1 min | not re-checked (bootstrap, seeded) |
| qShell transcriptions (QSHELL_P99_NS, LUT_D9, V80 budget/shell) | `python results/external/extract_qshell.py --no-write` | stdout | no | none (stdlib) | <1 s | tier1: exact |

`DIR` must contain the raw sweep files: `cp -r results/system work/system` (or the output of
`run_paper_sweeps.sh`); `verify/qec_system.py` does this in `out/verify/system`.

### §7 simulations (`system/run_paper_sweeps.sh <step>`, commands as run on steve/graham, recovered from transcript 94019386)

| Result (paper item) | Step / command | Output | Needs dataset? | Image | Runtime (paper run) | Verified |
|---|---|---|---|---|---|---|
| ler_grid.jsonl (fig:sys_spacetime a; :9, :36) | `grid`: `sweep_ler.py --set grid --workers 110` | `$OUT/ler_grid.jsonl` | no | oraqle-latsim | 21 min at 110 cores (≈40 core-h) | tier2: exact replay of 2 jobs; quick fresh run |
| ler_disc_hi.jsonl (all of §7.2-7.5) | `disc_hi`: `sweep_ler.py --set disc --target-fails 1000 --cap-scale 5 --workers 90` | `$OUT/ler_disc_hi.jsonl` | no | oraqle-latsim | 15 min at 90 cores | tier2: exact replay of 2 jobs; `--quick` disc run from the new layout (d=5, 30 points, 2 s) |
| ler_disc.jsonl (earlier ~200-failure set, superseded) | `disc` | `$OUT/ler_disc.jsonl` | no | oraqle-latsim | 5 min at 90 cores | covered by the same code path |
| decode_timing_d{5..17}.jsonl (fig:sys_decoder; :13, :56, :80) | `timing` (one pinned core per d) | `$OUT/decode_timing_d*.jsonl` | no | oraqle-latsim (graham) | 13 min (d=17) | tier2: detection events per round of d=5 at 400/1000 ns exact (times are machine-dependent) |
| decode_large/d{21..57}_T*.jsonl (CPU fits, "up to d=57") | `timing_large` | `$OUT/decode_large/` | no | oraqle-latsim (graham) | ≤25 min per job | not re-run (same script) |
| soft/cnn_soft_*ns.npz (§7.6 inputs) | `soft_cnn`: `soft_readout_cnn.py` per length, `CNN_LEN` 100..500 | `$OUT/soft/` | **yes** | oraqle-torch-gpu | ~30 min per length on an H200 | tier2 plumbing only: imports from `discriminators/`, dataset found via `ORAQLE_DATA=/data`, first step (`calibrate`) runs on the real dataset (0.7 s), CNN has 3,061 parameters; retraining not re-run |
| soft_params.json (flag error rates; :150, :152) | part of `soft_qec` (`sweep_soft.flag_params`) | `$OUT/soft_params.json` | no (uses the npz) | oraqle-latsim | seconds | tier2: re-computed from the 5 committed npz, identical to the committed file |
| ler_soft.jsonl (fig:sys_soft) | `soft_qec`: `sweep_soft.py --workers 150` | `$OUT/ler_soft.jsonl` | no | oraqle-latsim | ≈7 min at 150 cores | tier2: exact replay of 2 jobs |
| ler_decoders.jsonl (fig:sys_decoders, §7.7) | `decoders`: `sweep_decoders.py --workers 180` (paper run stopped by hand at 126/136 jobs) | `$OUT/ler_decoders.jsonl` | no | oraqle-latsim:dec | 3.6 h at 180 cores | tier2: exact replay of 2 jobs (all 7 decoders); `--quick` fresh run (32 jobs, 28 s) |
| ler_decoders_soft13.jsonl / _soft11.jsonl | `dec13`, `dec11` (union-find off) | `$OUT/ler_decoders_soft1{3,1}.jsonl` | no | oraqle-latsim:dec | 22 / 2.5 min | covered by the decoder replay (same function, seeds per point) |
| ler_crosscheck.jsonl (not cited) | `crosscheck` | `$OUT/ler_crosscheck.jsonl` | no | oraqle-latsim:dec | minutes at 24 cores | not re-run |
| helper scripts (probes, union-find test) | `docker run $COMMON oraqle-latsim:dec python system/helpers/test_uf.py` | stdout | no | oraqle-latsim:dec | 5 s | tier2: ran (union-find vs MWPM on Stim memory circuits) |

## 4. §6 (qec_impact.tex) and the QEC appendix

These rows come from the ECCentric part (`qec/eccentric/README.md`). "eccentric" = the image built by
`qec/eccentric/fetch_eccentric.sh && docker build -f work/eccentric/Dockerfile.oraqle -t eccentric work/eccentric`.
Simulation runs mount `-v $PWD/work/ecc_out:/qec/experiment_results` (and `-v $PWD/results:/results:ro -e
ORAQLE_RESULTS=/results` for the discriminator driver, which reads `results/discriminators/master_fidelity.csv`).
Simulation CPU-hours are the sum of the per-configuration `seconds` column of the paper's CSV. tier2 = executed from
the new layout on graham. No row needs the raw dataset.

### §6 / appendix: figures, tables and prose numbers

| Paper item (label, chapter:line) | Command | Output | Needs dataset? | Image | Runtime | Verified |
|---|---|---|---|---|---|---|
| fig:ler_discriminators, plots/ler_pair_discriminators.pdf (qec_impact:23-30) | `make -C paper_figures/qec ler_discriminators` | `$ORAQLE_OUT/paper_figures/qec/ler_pair_discriminators.pdf` | no | none (numpy+matplotlib) or eccentric | 2 s | tier1 + tier2: 0 differing pixels at 300 dpi, drawing streams identical (script recovered from transcript eb84b470) |
| fig:six_noise, plots/ler_pair_noise.pdf (qec_impact:97-104) | `make -C paper_figures/qec six_noise` | `.../ler_pair_noise.pdf` | no | same | 2 s | tier1 + tier2, identical as above |
| fig:six_future, plots/ler_pair_future.pdf (qec_impact:111-118) | `make -C paper_figures/qec six_future` | `.../ler_pair_future.pdf` | no | same | 2 s | tier1 + tier2, identical as above |
| fig:ler_current, plots/readout_ler_herqules_current.pdf (appendix_herqules:413-418) | `make -C paper_figures/qec ler_current` | `.../readout_ler_herqules_current.pdf` | no | same | 1 s | tier1 + tier2, identical (plot script @7858c2c); content issue: old HERQULES readout, see below |
| fig:ler_futuristic, plots/readout_ler_herqules_futuristic.pdf (appendix_herqules:422-427) | `make -C paper_figures/qec ler_futuristic` | `.../readout_ler_herqules_futuristic.pdf` | no | same | 1 s | tier1 + tier2, identical; same content issue |
| fig:six_decoh, plots/noise_models_ler_decoherence_only.pdf (appendix_herqules:435-440) | `make -C paper_figures/qec six_decoh` | `.../noise_models_ler_decoherence_only.pdf` | no | same | 1 s | tier1 + tier2, identical (cluster-only script plot_noise_models_paper.py) |
| fig:six_fut_fixed, plots/noise_models_ler_futuristic_fixedreadout.pdf (appendix_herqules:458-463) | `make -C paper_figures/qec six_fut_fixed` | `.../noise_models_ler_futuristic_fixedreadout.pdf` | no | same | 1 s | tier1 + tier2, identical (cluster-only plot_single_model_paper.py; git's plot_single_model.py gives the same PDF) |
| All seven figures vs the paper PDFs | `make -C paper_figures/qec all compare` | report on stdout, exit 1 on mismatch | no | same | 10 s | tier1 + tier2 |
| §6.1 prose: ~590% (surface 0.419→0.061), ~415%, ~10%/~39% spread, 69% (QubiCML 0.103 vs CNN 0.061), ~26%, Bacon ~0.5 (qec_impact:27, 32, 38) | `python qec/eccentric/paper_numbers.py --check` (or `make -C paper_figures/qec numbers`) | stdout; exit 1 on mismatch | no | none (stdlib+numpy) | 1 s | tier1: all within rounding (590.3, 0.419/0.0607, 416.0, 10.0/39.3, 69.2, 0.1027, 26.1) |
| §6.1 d=12: gap ≤6% (surface 1, hh 2, Bacon 6, Gross 0), spread ~2.4% (top 7.5%), saturation 0.5 / 1.0 (qec_impact:27, 35, 38) | same | same | no | same | 1 s | tier1 (1.4, 1.9, 6.3, 0.1; 2.36; 7.48; 0.479-0.516; ≥0.998) |
| Footnote noise constants CZ 1.191e-3, SX 1.637e-4, T1/T2 284.95/322.68 µs (qec_impact:20) | same (`ECCENTRIC_DIR=work/eccentric` adds run_ler_noise_models.py and compare_readout_eccentric.py) | same | no | same | 1 s | tier1 (all three drivers match) |
| §6.2 "up to ~27%" low coherence + backlog (qec_impact:101, 106, 109) | same | same | no | same | 1 s | tier1 (26.7%, surface d=5) |
| App. sim comparison prose: lattice-sim U-shaped at ~600 ns, orders of magnitude lower (appendix_herqules:409, 416, 425) | same | same | no | same | 1 s | tier1 (argmin 600 ns at every d; ECCentric/lattice-sim ≥ 66× at 600 ns) |
| App. decoherence-only prose: Bacon ≈0, Gross fails (appendix_herqules:433, 438) | same | same | no | same | 1 s | tier1 (Bacon ≤ 0.0006, Gross ≥ 0.999) |
| fig:regime_map (qec_impact:123-144) | hand-made TikZ in the paper source | n/a | no | n/a | n/a | n/a (conceptual drawing) |
| Extra (not in the paper): CNN-readout simulator comparison with both panels | `make -C paper_figures/qec extras` | `$ORAQLE_OUT/extras/qec/readout_ler_cnn_{current,futuristic}.pdf` | no | same | 2 s | tier1 + tier2 (ran) |

### §6 / appendix: simulations behind the CSVs (results/qec/experiment_results)

| Paper item | Command | Output | Needs dataset? | Image | Runtime | Verified |
|---|---|---|---|---|---|---|
| Six-code sweeps, current noise (fig:six_noise left, fig:six_decoh, CNN line of fig:ler_discriminators) | `docker run --rm -v $PWD/work/ecc_out:/qec/experiment_results -e LER_MODE=current eccentric python -u run_ler_noise_models.py` | `work/ecc_out/noise_models/ler_noise_models.csv` | no | eccentric | 98 CPU-h; longest config 3.0 h | tier2 smoke: surface d=5 heron_cnn 600 ns, 20k shots: LER 0.07275 vs paper 0.0703 (z = 0.96), 4 s |
| Low coherence + backlog (fig:six_noise right, ~27%) | same with `-e LER_MODE=boston_lowt_backlog` | `.../ler_noise_models_boston_lowt_backlog.csv` | no | eccentric | 12.5 CPU-h; longest 58 min | tier2 smoke: surface d=5 200 ns, 50k shots: 0.02242 vs 0.0222 (z = 0.24) |
| High-fidelity / futuristic (fig:six_future) | `-e LER_MODE=highfid`, `-e LER_MODE=futuristic` | `..._highfid.csv`, `..._futuristic.csv` | no | eccentric | 15 / 17 CPU-h; longest 1.5 / 2.1 h | not run (same driver as the smoke rows) |
| Fixed-fidelity futuristic (fig:six_fut_fixed) | `-e LER_MODE=futuristic_fixedreadout` | `..._futuristic_fixedreadout.csv` | no | eccentric | 16 CPU-h; longest 2.2 h | not run |
| Discriminator sweep d=5 (fig:ler_discriminators left, §6.1 numbers) | `docker run --rm -v $PWD/work/ecc_out:/qec/experiment_results -v $PWD/results:/results:ro -e ORAQLE_RESULTS=/results -e LER_DIST=5 eccentric python -u run_ler_discriminators.py` | `work/ecc_out/discriminators/ler_discriminators_d5.csv` | no (reads results/discriminators/master_fidelity.csv) | eccentric | 13 CPU-h; longest 24 min | tier2 smoke: surface d=5 MCMit-T 600 ns, 20k shots: 0.0717 vs 0.07115 (z = 0.21); color d=5 HERQULES 400 ns (BP-OSD, 5k): 0.1876 vs 0.1842 (z = 0.44); missing master_fidelity.csv gives a clear error |
| Discriminator sweep d=12 (fig:ler_discriminators right) | same with `-e LER_DIST=12 -e LER_CODES=surface,bacon,hh -e LER_INCLUDE_CNN=1` (env reconstructed from the CSV and log config count) | `..._d12.csv` | no | eccentric | 5.8 CPU-h; longest 11 min | not run |
| Simulator comparison, ECCentric half (fig:ler_current/futuristic (a)) | `qec/eccentric/fetch_eccentric.sh --with-7858c2c`, then `docker run --rm -v $PWD/work/eccentric-7858c2c/compare_readout_eccentric.py:/qec/compare_readout_eccentric_7858c2c.py:ro -v $PWD/work/ecc_out:/qec/experiment_results eccentric python -u compare_readout_eccentric_7858c2c.py` | `.../readout_compare/eccentric_readout_ler_herqules_nmr1.csv` (the plotted d+1 variant needs MEMORY_ROUNDS=d+1, a code state that was never committed) | no | eccentric | 50 configs, 15k-100k shots each, hours | tier2 smoke: d=5 600 ns current, 100k shots: 0.3631 vs nmr1 CSV 0.36084 (z ≈ 1.0), 23 s |
| Simulator comparison, lattice-sim half (fig:ler_current/futuristic (b)) | `docker run --rm -v $PWD/work/eccentric-7858c2c/synchronization-artifact:/art -w /art eccentric python -u compare_readout_artifact.py` (lattice-sim copy inside eccentric @7858c2c; writes `artifact_readout_ler_herqules.csv` in that directory, copy it into `experiment_results/readout_compare/`) | `work/eccentric-7858c2c/synchronization-artifact/artifact_readout_ler_herqules.csv` | no | eccentric (stim 1.14, PyMatching 2.3.1) | 50 configs × 500k shots; ~16 s per config at d=5 | tier2 smoke: d=5 600 ns current, 500k shots: 0.006478 vs paper 0.006688 (z ≈ −1.3), 17 s |

Sampling in the ECCentric drivers is unseeded, so re-runs agree statistically, not bit for bit.


## 5. discussion.tex

| Paper item | Source | Verified |
|---|---|---|
| discussion:33 "readout can be cut to ~600 ns ... reclaiming ~40% of readout latency" | derived from the §4/§5 plateau (docs/reproduce/discriminators.md); 40% = (1000 − 600)/1000; consistent with the §7 optimum (600-800 ns, `verify`) | derived, no separate check |
| discussion:33 "limited by device physics ... (mostly qubit state changes and SNR overlap)" | qualitative, from the error-mode analysis (docs/reproduce/leakage.md) | see leakage.md |

## 6. Known content issues (paper text vs data; recorded, not fixed; do not fail verification)

§7 (listed with measured values by `verify` from `results/system/paper_values.json` `known_issues`):

1. **§7.6 "flagging 20% ... lowers Λ to ~2.0 everywhere"** (system_bottleneck:150) is false at 400 ns: Λ = 1.66
   (2.00-2.05 at 600-1000 ns).
2. **§7.2 "HERQULES stays within 1.0-1.3× of the CNN at 600-1000 ns"** (:38) holds only relative to the CNN at
   1000 ns (0.96/1.00/1.28); at equal readout length HERQULES costs 1.28-1.63× the CNN's spacetime.
3. **§7.2 "every discriminator crosses [Λ=2 / 6%] between 400 and 600 ns"** (:31 caption, :36): QubiCML at 600 ns
   has p = 6.15% and Λ = 1.96; it reaches Λ ≥ 2 only at 800 ns (2.00) and drops to 1.79 at 1000 ns.
4. **§7.2 "12.4× (QubiCML)"** (:38): 12.35, which rounds to 12.3.
5. **§7.4 "QubiCML ... 2.2-2.5× the qubits"** (:105): 2.19-2.45 (2.2-2.4).
6. **§7.6 "(600 ns, 5% flagged: 5.1% to 3.7%)"** (:150): the retrained CNN's unflagged error is 5.045% (5.0%);
   5.06% is the Table-fidelity value.
7. **§7.7 "16 weight levels stay within 0.03 of full weights"** (:177): true at the four hard-readout points (max
   0.019), not at the 600 ns / 5% soft point (0.036). **"4 levels ... cost 1.5-2.2× the spacetime"**: 1.17× at that
   soft point.
8. **fig:sys_decoders caption "9-12× the qubits"** (:172): 8.50-12.03 (800 ns gives 8.498).
9. **§7.1 "308 ns at p99 (..., 168 ns per round)"** (:13): 168 ns per round belongs to the 1.51 µs whole-shot path
   (incl. serialization), not to the 308 ns decode stage (≈34 ns per round); unused in any computation.
10. **§7.1 Micro Blossom "772 ns at d=9 on an AMD Alveo V80"** (:13): no bench, log or script exists in any
    repository (transcribed from CXL-Quantum Table tab:req-decoders, commit 3307a11). Micro Blossom is charged a
    whole-shot latency, Helios only its decode stage (results/external/COMPARISON_qshell_vs_sec7.md).
11. **§7.3 Helios latency mapping** (:80, tab:sys_reaction): the equivalent p is matched with *uniform* circuit noise
    but qShell measured Helios under SI1000; with an SI1000 match Helios's p99 would be ~180-210 ns lower and
    "p≈0.7-1.2%" would read ≈0.4-0.7%; unweighted Helios would wait 1 round at 800 ns, weighted Helios 4 at 400 ns and
    2 at 800 ns. §7 is pessimistic for Helios there; its conclusion (unweighted Helios is Λ-limited) stands.

§6 and the QEC appendix (from `results/qec/paper_values.json` `known_issues`):

12. **fig:ler_current / fig:ler_futuristic (readout_ler_herqules_{current,futuristic}.pdf) use the old mixed-shot
    HERQULES fidelities** (eccentric @7858c2c `HERQULES_ACC`, measure_error 0.221/0.051/0.034/0.034/0.033, the set
    later moved to `buggy_herqules/`), not the MCMit-CNN readout that qec_impact:20 states. CNN reruns exist for both
    halves (lattice-sim 49/50 points); `make -C paper_figures/qec extras` draws the CNN version (same qualitative
    picture).
13. **§6.1 "the best discriminator is always an MCMit variant"** (qec_impact:27, 32) is false at 400 ns (Baseline FNN
    best on surface 0.1218 and color 0.1546; KLiNQ on Bacon-Shor); **"the worst QubiCML or the Linear threshold"** is
    false at 200 ns (HERQULES worst on surface and heavy-hex).
14. **"ECCentric has the LER saturated near 0.5"** (qec_impact:20, appendix_herqules:409) for the lattice-surgery CNOT:
    the plotted CSV saturates at ~0.73-0.75 for d ≥ 7 (three logical observables).
15. **The plotted ECCentric comparison CSV** (`eccentric_readout_ler_herqules.csv`, d+1 memory rounds) came from an
    uncommitted state of `compare_readout_eccentric.py`; the 7858c2c file as committed writes the `_nmr1` control.
16. §6 statistics at d=12 are thin (5k BP-OSD / 20k MWPM shots, Gross 1k); the paper itself calls those spreads
    sampling noise. "~415%" is 416.0%.

Other appendix item requested to be recorded (outside the QEC verification, from the provenance audit):

17. **tab:fidelity_linear** (appendix_herqules:224-240): Q5 at 800 ns is printed 0.936 but `master_fidelity.csv`
    ("Linear Threshold") has 0.9633; F4Q at 800 ns is printed 0.934 (computed from the typo), correct 0.941, which also
    moves the bold F4Q from 600 to 800 ns (see docs/reproduce/discriminators.md for the discriminator tables).

Narrative note (not an error): §6 and the appendix adopt ECCentric as the faithful simulator ("lattice-sim under-models
noise"), while §7 uses lattice-sim because ECCentric's circuits sit above threshold; a reviewer will notice.

## 7. qShell vs §7 decoder numbers

Full table: `results/external/COMPARISON_qshell_vs_sec7.md`. Summary:
* **Match exactly:** `QSHELL_P99_NS` (1680...2592 ns = nearest-rank p99 of the qShell histograms), `LUT_D9` (Helios
  116,575, Micro Blossom 155,399), V80 budget 2,574,208 and shell 63,989 LUT, "largest Helios on one V80 = d=21".
* **The qShell p99 (1680 ns at p=1e-4) is 1508 ns of fixed interface/serialization plus the Helios kernel**; §7 uses
  only its growth over p=1e-3 on top of 308 ns, i.e. §7's Helios p99 = qShell kernel p99 + 12 ns at every p:
  consistent.
* **Not in qShell:** Micro Blossom 772 ns (qShell has only an illustrative 240/480 ns), Helios 184/308 ns
  (CXL-Quantum table), GPU 2.1 µs (CUDA-Q). See issues 9-11 above.
* K_LUT = 3.39 comes from Helios Table I (d=9..21), not from qShell's d3/d9 pair (which would give 2.47), by design.

## 8. UNIFORM_FRACTION

Derived: `system/derive_uniform_fraction.py` = the inline stim run of 2026-10-26 (transcript 94019386, line 1039):
mean detection fraction of `stim.Circuit.generated("surface_code:rotated_memory_z", distance=9, rounds=9)` with
uniform noise p ∈ {1e-3, 3e-3, 5e-3, 1e-2, 2e-2} (after-Clifford depolarization, before-round data depolarization,
before-measure flip, after-reset flip, all = p), 20,000 shots, seed 1. Reproduces all five constants exactly with
stim 1.16.0 (laptop MCMit venv and oraqle-latsim on steve). Represents: the detection-event density that uniform
circuit noise p produces at d=9, used to place each readout length's measured density (d=9 timing runs) on qShell's
Helios-latency-vs-p curve.

## 9. Tier-2 smoke tests (steve, 2026-11-05 13:49-14:05 UTC; repo rsynced to ~/oraqle-unify-smoke/qec/repo)

| Test | Command (in `~/oraqle-unify-smoke/qec`, `U=$(id -u):$(id -g)`) | Outcome |
|---|---|---|
| exact replay, original image | `docker run --rm --name unify-qec-replay --user $U -e HOME=/tmp -v $PWD/repo:/repo:ro -w /repo oraqle-latsim:dec python system/replay_points.py --n 2` | 15/15 identical (grid 2, disc_hi 2, soft_params 5, soft 2, decoders 2 with all 7 decoders, timing 2); 54 s |
| tiny lattice-sim LER sweep | `docker run --rm --name unify-qec-latsim-quick ... oraqle-latsim python system/sweep_ler.py --set disc --distances 5 --quick --workers 16 --out /out/disc_quick.jsonl` | 30 points, 7 s; e.g. HERQULES 1000 ns d=5: 107/5000 |
| tiny decoder sweep | `docker run --rm --name unify-qec-dec-quick ... oraqle-latsim:dec sh -c "python system/sweep_decoders.py --out /out/dec_quick.jsonl --quick --workers 32 && python system/analyze_decoders.py /out/dec_quick.jsonl"` | 32 jobs, 28 s; d=5/7 MWPM failures at 400 ns (420/343) equal the published ones |
| verify + stim | `docker run --rm --name unify-qec-verify ... -e ORAQLE_OUT=/out oraqle-latsim python verify/qec_system.py --stim` | 15/15 checks passed (no poppler in the image, so no `--figures`) |
| documented recipes (docker/README.md `$COMMON`), final state | replay, `verify --stim`, quick disc sweep, and the analysis chain with the default `$ORAQLE_WORK/system` | 0 differing jobs; 15/15; ok; `decoders_summary.txt` identical, summaries equal up to the last float digit |
| soft-CNN plumbing with the dataset | `docker run --rm --name unify-qec-softcnn-plumb ... -v ~/qubit_readout_klinq/data/five_qubit_data:/data:ro -e ORAQLE_DATA=/data oraqle-torch-gpu python -c "...import soft_readout_cnn; require_dataset(); P.calibrate(...)"` | imports from `discriminators/` OK, dataset found, calibrate on the real traces 0.7 s, CNN 3,061 parameters |
| union-find helper | `docker run --rm --name unify-qec-testuf ... oraqle-latsim:dec python system/helpers/test_uf.py` | ran, 5 s |
| rebuilt pinned images | `docker build -t unify-qec-latsim repo/docker/latsim`; `docker build -f repo/docker/latsim/Dockerfile.dec -t unify-qec-latsim-dec repo/docker/latsim` | `pip freeze` == locks (Python 3.11.17); replay in the rebuilt images: 13/13 + 1/1 identical; images removed afterwards |
| ECCentric (graham) | see section 4 (simulation rows) and `qec/eccentric/README.md` | 6 one-point runs agree with the paper CSVs (|z| ≤ 1.3); image `unify-qec-eccentric` (2.87 GB) left on graham for re-tests: `docker rmi unify-qec-eccentric` |
