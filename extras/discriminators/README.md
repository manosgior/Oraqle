# extras/discriminators/ — discriminator work the paper does not use

Real code from the cluster workspace (`~/Oraqle/Discriminators`) and the old git tree that
no paper number depends on. It is kept so nothing is lost; it is not maintained and was only
path-plumbed (raw-data, report and checkpoint paths go through `discriminators/disc_paths.py`;
every edited line is tagged `[unify]`), not re-run. Scripts import the shared framework from
`discriminators/` (both directories are put on `sys.path`; `runners` is a namespace package).

| Group | Files (runners/ unless noted) | Notes |
|---|---|---|
| Three-level / leakage discrimination survey | `build_three_level_labels.py`, `three_level_common.py`, `train_three_level_{arxiv,cnn,fnn,herqules,klinq,transformer}.py`, `assemble_three_level_table.py`, `inflate_leakage.py`, `error_analysis_sweep.py` | the 3-level pseudo-labels are circular (leakage audit 2026-11): do not train on them |
| Synthetic traces | `generate_synthetic.py`, `synth_to_cnn.py`, `synth_transfer_sweep.py`, `_debug_synth.py` | results `synth_transfer_sweep_*.csv` |
| Seed / subset variance | `_seed_sweep.py`, `_seed_sweep_csv.py`, `_hybrid_seed_sweep.py`, `herqules_subset_variance.py` | |
| HybridCNN (an earlier CNN, not MCMit-CNN) | `train_hybrid_cnn.py`, `_train_and_analyze_hybridcnn.py`, `_xtalk_hybridcnn.py` | `networks/HybridCNN.py` stays in discriminators/ (used by the leakage study) |
| HERQULES analyses | `analyze_*` lives in discriminators/; here: `herqules_error_report.py`, `_herqules_binary_heads.py`, `_herqules_error_breakdown_full.py`, `_herqules_mw_features.py`, `_herqules_truncation_eval.py` (paper-subsample variant of the no-retrain table), `_test_as_published.py`, `_train_herqules_once.py`, `_deploy_compare.py`, `_xtalk_herqules.py` (scrambled variant) | |
| CNN variants | `hyper_optimize_cnn.py` (Optuna CNN_m16 on ds20 data: the `CNN_m16_*` results/checkpoints, **not** the Table 2 CNN), `_cnn_truncation_eval.py`, `_colleague_cnn_tf.py` (the colleague's TensorFlow reference), `_train_cnn_once.py`, `train_CNN.py` (older; `/home/sandra` path replaced by `$ORAQLE_WORK/cnn`) | |
| Other | `evaluate_truncated.py` (truncation evaluation of all published checkpoints; reads `checkpoints/`), `hyper_optimize_transformer_mf.py` (TransformerMF; the MCMit sync copied this one, but the paper's MCMit-T came from discriminators/runners/hyper_optimize_transformer.py), KLiNQ-upstream runners `train_KD_with_KLiNQ_TeacherStudent.py`, `train_KD_with_SingleQubitFNN.py`, `train_SingleQubitFNN.py` | |
| `docs/` | `MIGRATION_GUIDE.md`, `QUICK_REFERENCE.md`, `README_REFACTORING.md`, `REFACTORING_CHECKLIST.md`, `REFACTORING_SUMMARY.md` (framework refactoring notes, May 2026), `execute.md` (the old docker recipe) | historical |
| `legacy/` | `HERQULES.py` (early top-level HERQULES; imports `herqules_demodulation`, which exists only in the precursor repo `qubit_readout_klinq`, so it does not run; kept verbatim), `test.py` (scratch test) | |
| `scratch/` | table-making scripts from the old repo root (May 2026), reading `results/discriminators/legacy_training_results/` | |
