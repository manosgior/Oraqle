# checkpoints/ — trained models

Top level: verbatim copy of `graham:~/oraqle_models` (112 files, 48 MB; best Optuna trial
per model/length, KLiNQ paper-reproduction students, HERQULES per-trace checkpoints).
`INDEX.csv` maps every checkpoint to the result CSV that references it (`model_path`
column) and to its paper use. `legacy_trained_models/` is the older git snapshot
`Discriminators/trained_models` (185 files), including the KLiNQ teacher/student Optuna
models (`KLiNQ_{teacher,student}_q*_len*`) that are **not** used by the paper.

Which checkpoints feed paper numbers (Table 2 numbers were measured during training and are
stored in the result CSVs; the checkpoints let you re-evaluate the models):

| Paper row | Checkpoints |
|---|---|
| Baseline FNN 200..1000 ns | `FNN_len{100,200,300,400,500}_trial*.pth` (as named in `FNN_len*.csv`) |
| QubiCML | `Arxiv_q{0..4}_len{100..500}_trial*.pth` (as named in `Arxiv240618807FNN_q*_len*.csv`) |
| KLiNQ | `KLiNQ_student_paper_q{0..4}_{200,400,600,800,1000}ns.pth` (as named in `KLiNQ_paper_repro_20260601_141220.csv`; teachers were not saved). **Resolved:** the KLiNQ row comes from these cluster checkpoints, not from the local `legacy_trained_models/KLiNQ_{teacher,student}_*` (those belong to the older Optuna KLiNQ runs in `legacy_training_results/KLiNQ_*`) |
| MCMit-T | `Transformer_len{100..500}_trial*.pth` (as named in `Transformer_len*.csv`; e.g. len500_trial37, 19.6 MB) |
| HERQULES, MCMit-CNN | none: `_herqules_length_both.py` and `_cnn_length_sweep.py` retrain per length with seed 1 and save no checkpoint |
| §5.3 per-trace HERQULES (leakage study), tab:fpga_resources HERQULES | `HERQULES_len{200,500}.pth` (June 2, `_train_and_analyze_errors.py`, with `results/discriminators/rmf_len*.pkl`) |
| tab:fpga_resources QubiCML / KLiNQ | `Arxiv_q0_len500_trial3.pth`, `KLiNQ_student_paper_q{0,2}_1000ns.pth` (byte-identical to the files in `fpga/hls4ml/checkpoints/`) |

`CNN_m16_*` (Optuna CNN on ds20 data) and `HybridCNN_*` are not used by the paper
(`CNN_m16_len500_trial36.pth` was synthesized as `fpga/hls4ml/reports/cnn-m16.txt`).
