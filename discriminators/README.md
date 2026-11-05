# discriminators/ — ML qubit-state discriminators (paper §5, Table 2)

The discriminator framework (formerly `Discriminators/`, and the newer non-git copy on the
cluster, `~/Oraqle/Discriminators`), reorganised so that it runs from a clone. The full
reproduction guide is `docs/reproduce/discriminators.md`; the dataset is described in
`dataset/README.md`; provenance per file is in `PROVENANCE.md`.

## Layout

| Path | Contents |
|---|---|
| `disc_paths.py` | maps the old Docker mounts (`/data/five_qubit_data`, `/app/optimization_reports`, `/app/saved_models`, `/data/cnn`) to `oraqle_env.py` paths |
| `networks/` | model definitions: `SingleQubitFNN` (Baseline FNN, KLiNQ teacher), `Qubic.Arxiv240618807FNN` (QubiCML), `HERQULES.Net_rmf`, `KLiNQ_StudentModel`, `Transformer` (MCMit-T), `CNN` (MCMit-CNN), plus `TransformerMF`, `HERQULESPlus`, `HybridCNN` used by extras |
| `trainers/` | `HERQULES_original.py` (HERQULES port; loads and demodulates the data at import), `train_HERQULES.py`, KD trainers |
| `helpers/` | data loading, CNN preprocessing, `extract_data*.py` (derived CNN inputs into `$ORAQLE_WORK/cnn`) |
| `matched_filter.py` | MF / RMF code used by HERQULES |
| `runners/` | producers of the paper numbers (table below) |
| `tables/` | `build_table2.py` (rebuilds master_fidelity.csv and tab:fidelity), `build_master_and_table.py` (the original builder, kept for provenance), `extract_paper_values.py` |
| `smoke/smoke_table2.py` | tier-2 smoke test of every producer (tiny configs) |
| `plot_fidelities.py`, `fetch_results.py` | fig:pareto and older per-model plots |
| `docs/framework_overview.md` | the original framework README (HERQULES stages, models, KLiNQ story) |
| `data/` | KLiNQ upstream MF/RMF envelopes and notebooks (small) |

## Paper producers

| Paper item | Runner | Output (in `$ORAQLE_OUT/discriminators/reports`) |
|---|---|---|
| tab:fidelity Baseline FNN, QubiCML | `runners/hyper_optimize_paper.py` (paper protocol wrapper around `runners/hyper_optimize.py`) | `FNN_len*.csv`, `Arxiv240618807FNN_q*_len*.csv` |
| tab:fidelity HERQULES; tab:herqules-pertrace | `runners/_herqules_length_both.py` | `herqules_length_faithful_vs_demux.csv` |
| tab:fidelity KLiNQ | `runners/reproduce_klinq_paper.py` | `KLiNQ_paper_repro[_F5Q]_*.csv` |
| tab:fidelity MCMit-T | `runners/hyper_optimize_transformer.py` | `Transformer_len*.csv` |
| tab:fidelity MCMit-CNN | `CNN_LEN=<samples> runners/_cnn_length_sweep.py` (one length per process) | `cnn_length_<ns>ns.csv` |
| tab:fidelity Linear Threshold (appendix tab:fidelity_linear), §4 sweeps | `readout/run_baseline_sweeps.py` | `Baseline_Threshold_*.csv` |
| tab:herqules-noretrain | `runners/_herqules_truncation_pertrace.py` | `herqules_truncation_pertrace_deployment.csv` |
| App. seed-to-seed variance | `runners/_herqules_full.py` | `herqules_full_dataset.csv` |
| master_fidelity.csv, tab:fidelity LaTeX | `tables/build_table2.py` | `$ORAQLE_OUT/discriminators/tables/` |
| fig:pareto | `plot_fidelities.py` (via `paper_figures/discriminators/fig_pareto.py`) | `pareto_params_vs_fidelity_600ns_1000ns.pdf` |
| leakage-study inputs (used by `leakage/`) | `runners/_xtalk_cnn.py` (`preds_MCMitCNN_{pb,tb}.npy`), `runners/_xtalk_herqules_deploy.py` (`preds_HERQULESdeploy_*`), `runners/_train_and_analyze_errors.py` (`HERQULES_len{200,500}.pth`, `rmf_len*.pkl`), `runners/iq_error_gallery.py` (fig:iq_taxonomy npz + plot) | as named |

Run scripts with the repository's Python environment, for example

    export ORAQLE_DATA=/path/to/five_qubit_data
    python discriminators/runners/hyper_optimize_paper.py
    CNN_LEN=500 python discriminators/runners/_cnn_length_sweep.py
    cd $ORAQLE_WORK && python $REPO/discriminators/runners/_herqules_length_both.py   # HERQULES writes rmf.pkl etc. to the cwd

Re-runs never write into `results/` or `checkpoints/`: reports go to
`$ORAQLE_OUT/discriminators/reports`, new checkpoints to `$ORAQLE_WORK/checkpoints`
(override with `ORAQLE_REPORTS_OUT`, `ORAQLE_MODELS_OUT`).

Not used by the paper but kept: `extras/discriminators/` (three-level / leakage trainers,
synthetic data, seed sweeps, crosstalk variants, HybridCNN, truncation evaluation, ...).
