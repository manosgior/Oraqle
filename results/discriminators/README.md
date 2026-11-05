# results/discriminators/ — published discriminator and readout results

Top level: a verbatim copy of `graham:~/oraqle_reports` (the directory every runner wrote to
as `/app/optimization_reports`), so file names are the original ones. `legacy_training_results/`
is the older git snapshot `Discriminators/training_results` (May 2026; includes KLiNQ
teacher/student Optuna runs and older Transformer runs not used by the paper).

Files the paper depends on:

| File(s) | Paper item | Producer |
|---|---|---|
| `master_fidelity.csv` | tab:fidelity, tab:fidelity_linear, fig:pareto, tab:model_comparison parameters; ECCentric/lattice-sim readout errors | `discriminators/tables/build_table2.py` (byte-identical rebuild) |
| `FNN_len*.csv`, `Arxiv240618807FNN_q*_len*.csv` | Baseline FNN, QubiCML rows | `runners/hyper_optimize_paper.py` |
| `herqules_length_faithful_vs_demux.csv` | HERQULES rows, tab:herqules-pertrace, App. 21% / 31-35% | `runners/_herqules_length_both.py` |
| `KLiNQ_paper_repro_F5Q_20260601_141220.csv` (+ per-qubit `KLiNQ_paper_repro_20260601_141220.csv`) | KLiNQ rows | `runners/reproduce_klinq_paper.py` |
| `Transformer_len*.csv` | MCMit-T rows | `runners/hyper_optimize_transformer.py` |
| `cnn_length_{200..1000}ns.csv`, `cnn_length_deployment.csv` | MCMit-CNN rows, F4Q 0.957 | `runners/_cnn_length_sweep.py`; aggregate rebuilt by build_table2.py |
| `Baseline_Threshold_len*.csv`, `Baseline_Threshold_dFdT_analysis_20260514_131943.csv`, `Baseline_Threshold_sliding_window_{250ns_20260514_204419,500ns}.csv` | fig:readout_sweeps, §4 prose, Linear Threshold rows | `readout/run_baseline_sweeps.py` |
| `iq_clouds_vs_length_q5.npz` | fig:iq_clouds | `readout/iq_clouds_vs_length.py` (extract) |
| `herqules_truncation_pertrace_deployment.csv` | tab:herqules-noretrain | `runners/_herqules_truncation_pertrace.py` |
| `herqules_full_dataset.csv` | App. "seed-to-seed variance ≈ 0.0002" (a std, on the released assembly) | `runners/_herqules_full.py` |
| `iq_error_gallery_q5.npz`, `preds_MCMitCNN_{pb,tb}.npy`, `preds_HERQULESdeploy_{pb,tb}.npy`, `rmf_len{200,500}.pkl` | leakage study (§5.3, tab:error_modes, fig:iq_taxonomy; see `leakage/`) | `runners/iq_error_gallery.py`, `_xtalk_cnn.py`, `_xtalk_herqules_deploy.py`, `_train_and_analyze_errors.py` |
| `paper_values.json` | expected values for verify/discriminators.py (new) | `discriminators/tables/extract_paper_values.py` |
| `dataset_metadata.json` | dataset facts in overview.tex (new) | probed on graham |

History kept on purpose: `master_fidelity.csv.bak` (2026-06-09 intermediate: HERQULES rows from
the no-mix seed sweep, CNN rows from the CNN_m16 Optuna runs), `fidelity_table.tex` (stale
2026-06-01 output of build_master_and_table.py), `herqules_*_snapshot.csv[.bak]`.
`CNN_m16_len*.csv` belong to a different CNN run (hyper_optimize_cnn.py) that the paper does
not use. PDFs/PNGs are the plots as they were on the cluster.
