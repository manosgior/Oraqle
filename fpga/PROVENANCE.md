# fpga/ provenance

Sources (all read-only):

* **mcmit-model**: `/home/manosgior/Documents/GitHub/mcmit-model` (laptop), git, origin `git@github.com:gustifix/mcmit-model.git`, HEAD `56cfcc010f48af7696b8bc5df46f4463703afc97` (2026-06-15), clean working tree. Author of every commit: **Felix Gust** <gustifix@users.noreply.github.com>. Copied into `fpga/hls4ml/` on 2026-11-05.
* **Oraqle cluster tree**: graham `~/Oraqle/Discriminators/runners/` (not git; laptop mirror `/home/manosgior/Documents/Oraqle`), copied into `fpga/hls_input_prep/` (see the table at the end).

New files: `README.md`, `PROVENANCE.md`, `parse_reports.py`, `hls4ml/.gitignore`.

| File | Origin | Last commit touching it (mcmit-model) | Change |
|---|---|---|---|
| `fpga/hls4ml/.gitignore` | new (copy of mcmit-model/.gitignore entries hls-project*/, __pycache__/) | - | new |
| `fpga/hls4ml/checkpoints/CNN_1000ns.pth` | mcmit-model/micro-rebuttal/checkpoints/CNN_1000ns.pth | b3c6133 | verbatim |
| `fpga/hls4ml/checkpoints/CNN_1000ns_singlehead.pth` | mcmit-model/micro-rebuttal/checkpoints/CNN_1000ns_singlehead.pth | 0d016af | verbatim |
| `fpga/hls4ml/checkpoints/CNN_small_m8_ds40_1000ns_singlehead.pth` | mcmit-model/micro-rebuttal/checkpoints/CNN_small_m8_ds40_1000ns_singlehead.pth | fc054b6 | verbatim |
| `fpga/hls4ml/checkpoints/HERQULES_1000ns.pth` | mcmit-model/micro-rebuttal/checkpoints/HERQULES_1000ns.pth | b3c6133 | verbatim |
| `fpga/hls4ml/checkpoints/KLiNQ_student_LARGER_q2_1000ns.pth` | mcmit-model/micro-rebuttal/checkpoints/KLiNQ_student_LARGER_q2_1000ns.pth | f98fd18 | verbatim |
| `fpga/hls4ml/checkpoints/KLiNQ_student_paper_q0_1000ns.pth` | mcmit-model/micro-rebuttal/checkpoints/KLiNQ_student_paper_q0_1000ns.pth | b3c6133 | verbatim |
| `fpga/hls4ml/checkpoints/QubiCML_q0_1000ns.pth` | mcmit-model/micro-rebuttal/checkpoints/QubiCML_q0_1000ns.pth | b3c6133 | verbatim |
| `fpga/hls4ml/inputs/CNN_small_m8_ds40_input.npy` | mcmit-model/micro-rebuttal/inputs/CNN_small_m8_ds40_input.npy | fc054b6 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/CNN_small_m8_ds40_labels.npy` | mcmit-model/micro-rebuttal/inputs/CNN_small_m8_ds40_labels.npy | fc054b6 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/CNN_small_m8_ds40_output.npy` | mcmit-model/micro-rebuttal/inputs/CNN_small_m8_ds40_output.npy | fc054b6 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/HERQULES_input.npy` | mcmit-model/micro-rebuttal/inputs/HERQULES_input.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/HERQULES_labels.npy` | mcmit-model/micro-rebuttal/inputs/HERQULES_labels.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/HERQULES_output.npy` | mcmit-model/micro-rebuttal/inputs/HERQULES_output.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/KLiNQ_student_LARGER_q2_input.npy` | mcmit-model/micro-rebuttal/inputs/KLiNQ_student_LARGER_q2_input.npy | f98fd18 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/KLiNQ_student_LARGER_q2_labels.npy` | mcmit-model/micro-rebuttal/inputs/KLiNQ_student_LARGER_q2_labels.npy | f98fd18 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/KLiNQ_student_LARGER_q2_output.npy` | mcmit-model/micro-rebuttal/inputs/KLiNQ_student_LARGER_q2_output.npy | f98fd18 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/KLiNQ_student_q0_input.npy` | mcmit-model/micro-rebuttal/inputs/KLiNQ_student_q0_input.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/KLiNQ_student_q0_labels.npy` | mcmit-model/micro-rebuttal/inputs/KLiNQ_student_q0_labels.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/KLiNQ_student_q0_output.npy` | mcmit-model/micro-rebuttal/inputs/KLiNQ_student_q0_output.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/MCMitCNN_input.npy` | mcmit-model/micro-rebuttal/inputs/MCMitCNN_input.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/MCMitCNN_labels.npy` | mcmit-model/micro-rebuttal/inputs/MCMitCNN_labels.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/MCMitCNN_output.npy` | mcmit-model/micro-rebuttal/inputs/MCMitCNN_output.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/QubiCML_q0_input.npy` | mcmit-model/micro-rebuttal/inputs/QubiCML_q0_input.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/QubiCML_q0_labels.npy` | mcmit-model/micro-rebuttal/inputs/QubiCML_q0_labels.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/QubiCML_q0_output.npy` | mcmit-model/micro-rebuttal/inputs/QubiCML_q0_output.npy | 79b2a94 | not shipped (dataset slice; regenerate with hls_input_prep) |
| `fpga/hls4ml/inputs/README.txt` | mcmit-model/micro-rebuttal/inputs/README.txt | 79b2a94 | verbatim |
| `fpga/hls4ml/micro-submission/cnn-hls4ml.py` | mcmit-model/micro-submission/cnn-hls4ml.py | 200a242 | verbatim |
| `fpga/hls4ml/micro-submission/cnn-model-quantized.py` | mcmit-model/micro-submission/cnn-model-quantized.py | 200a242 | BASE_DIR path plumbed to $ORAQLE_CNN_DATA / $ORAQLE_WORK/cnn ([unify] tag) |
| `fpga/hls4ml/micro-submission/cnn-model.py` | mcmit-model/micro-submission/cnn-model.py | 200a242 | BASE_DIR path plumbed to $ORAQLE_CNN_DATA / $ORAQLE_WORK/cnn ([unify] tag) |
| `fpga/hls4ml/micro-submission/example.py` | mcmit-model/micro-submission/example.py | 200a242 | verbatim |
| `fpga/hls4ml/networks/CNN.py` | mcmit-model/micro-rebuttal/networks/CNN.py | fc054b6 | verbatim |
| `fpga/hls4ml/networks/HERQULES.py` | mcmit-model/micro-rebuttal/networks/HERQULES.py | b3c6133 | verbatim |
| `fpga/hls4ml/networks/KLiNQ_StudentModel.py` | mcmit-model/micro-rebuttal/networks/KLiNQ_StudentModel.py | b3c6133 | verbatim |
| `fpga/hls4ml/networks/Qubic.py` | mcmit-model/micro-rebuttal/networks/Qubic.py | b3c6133 | verbatim |
| `fpga/hls4ml/reports/cnn-m16.txt` | mcmit-model/micro-rebuttal/reports/cnn-m16.txt | 0d016af | verbatim |
| `fpga/hls4ml/reports/cnn-m8-parallelizationfactors.txt` | mcmit-model/micro-rebuttal/reports/cnn-m8-parallelizationfactors.txt | d35b94e | verbatim |
| `fpga/hls4ml/reports/cnn-m8-reuse2.txt` | mcmit-model/micro-rebuttal/reports/cnn-m8-reuse2.txt | 6cde6f9 | verbatim |
| `fpga/hls4ml/reports/cnn-m8-reuse4.txt` | mcmit-model/micro-rebuttal/reports/cnn-m8-reuse4.txt | 6cde6f9 | verbatim |
| `fpga/hls4ml/reports/cnn-m8.txt` | mcmit-model/micro-rebuttal/reports/cnn-m8.txt | fc054b6 | verbatim |
| `fpga/hls4ml/reports/herqules.txt` | mcmit-model/micro-rebuttal/reports/herqules.txt | d2d474f | verbatim |
| `fpga/hls4ml/reports/klinq-larger-reuse4.txt` | mcmit-model/micro-rebuttal/reports/klinq-larger-reuse4.txt | 56cfcc0 | verbatim |
| `fpga/hls4ml/reports/klinq-larger.txt` | mcmit-model/micro-rebuttal/reports/klinq-larger.txt | f98fd18 | verbatim |
| `fpga/hls4ml/reports/klinq.txt` | mcmit-model/micro-rebuttal/reports/klinq.txt | 06a4dd4 | verbatim |
| `fpga/hls4ml/reports/qubic.txt` | mcmit-model/micro-rebuttal/reports/qubic.txt | 7ca629d | verbatim |
| `fpga/hls4ml/requirements.txt` | mcmit-model/requirements.txt | b865386 | verbatim |
| `fpga/hls4ml/run-cnn.py` | mcmit-model/micro-rebuttal/run-cnn.py | 6cde6f9 | verbatim |
| `fpga/hls4ml/run-herqules.py` | mcmit-model/micro-rebuttal/run-herqules.py | d2d474f | verbatim |
| `fpga/hls4ml/run-klinq-larger.py` | mcmit-model/micro-rebuttal/run-klinq-larger.py | f98fd18 | verbatim |
| `fpga/hls4ml/run-klinq.py` | mcmit-model/micro-rebuttal/run-klinq.py | 06a4dd4 | verbatim |
| `fpga/hls4ml/run-qubic.py` | mcmit-model/micro-rebuttal/run-qubic.py | 06a4dd4 | verbatim |

| File | Origin (host graham, not git) | Change |
|---|---|---|
| `fpga/hls_input_prep/_make_example_inputs.py` | `~/Oraqle/Discriminators/runners/_make_example_inputs.py` (mtime 2026-06-12/15) | `sys.path.append("/app/Discriminators")` -> repo bootstrap; raw-data paths -> `disc_paths.RAW_TRAIN/RAW_TEST`; `/data/cnn/...` -> `$ORAQLE_WORK/cnn`; `MODELS="/m"` -> `checkpoints/`; `OUT="/out"` -> `$ORAQLE_OUT/fpga/example_inputs` (all edits tagged `[unify]`) |
| `fpga/hls_input_prep/_make_klinq_large_input.py` | `~/Oraqle/Discriminators/runners/_make_klinq_large_input.py` (mtime 2026-06-12/15) | repo bootstrap; raw-data paths; `/m` -> `checkpoints/`; `/out` -> `$ORAQLE_OUT/fpga/example_inputs` (all edits tagged `[unify]`) |
| `fpga/hls_input_prep/_cnn_small_hls.py` | `~/Oraqle/Discriminators/runners/_cnn_small_hls.py` (mtime 2026-06-12/15) | repo bootstrap; raw-data paths; `/ckpt`, `/ex` -> `$ORAQLE_OUT/fpga/{checkpoints,example_inputs}` (all edits tagged `[unify]`) |
| `fpga/hls_input_prep/_cnn_shallow_hls.py` | `~/Oraqle/Discriminators/runners/_cnn_shallow_hls.py` (mtime 2026-06-12/15) | repo bootstrap; raw-data paths; `/ckpt`, `/ex` -> `$ORAQLE_OUT/fpga/{checkpoints,example_inputs}` (all edits tagged `[unify]`) |
