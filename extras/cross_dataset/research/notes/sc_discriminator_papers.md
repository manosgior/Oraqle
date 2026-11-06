# Superconducting readout-discriminator papers: who released raw time-domain traces (paper-first sweep, as of 2026-11-06)

How to read these notes. Tiers T1 to T6 are as defined in the assignment. "None in arXiv text" means I downloaded the arXiv PDF, extracted the text and searched it for availability wording and repository names (`availab`, `request`, `zenodo`, `figshare`, `github`, `dataverse`, and similar), and found nothing relevant. That is weaker than reading the journal version, and APS, ACM and IEEE full texts were mostly paywalled. "not verified" means I did not see the fact in a primary source. Each table row's facts come from the paper linked in that row unless another link is given. Dates are the journal or OpenAlex publication date where I saw one, otherwise arXiv v1.

## Q1. Which papers from 2015 to November 2026 evaluate a discriminator on time-domain traces, and whose hardware data is it?

### Takeaway
I identified about 20 papers that used their own lab's superconducting hardware, 11 that reuse the Lienhard five-qubit data (3 more probably do, unconfirmed), 2 that reuse Di Guglielmo's data, and a smaller group of integrated-only and simulation-only papers. The reuse is real but not universal: LBNL, Purdue/Fermilab, Oxford, Rochester, Princeton/Pittsburgh, USTC, SUSTech/Macau and UQ all ran discriminators on their own traces.

### Cited Findings

**Group A: own-lab hardware, time-domain input (or claimed time-domain input).**

| # | Paper (date) | Platform and lab | Qubits | Shots | Samples and rate | Readout duration | Labels | Tier |
|---|---|---|---|---|---|---|---|---|
| A1 | [Magesan et al., PRL 114, 200501](https://arxiv.org/abs/1411.4994) (2015-05-18) | Superconducting qubit, IBM | not verified | 51,200 single-shot trajectories, half each state | not verified | not verified | prepared 0/1 | T6 |
| A2 | [Flurin et al., PRX 10, 011006](https://arxiv.org/abs/1811.12420) (2020-01-09) | Dispersive qubit-cavity, UC Berkeley (Siddiqi) | 1 | 2×10^6 traces | 40 ns time steps | variable, 0 to 4 µs | 6 preparation and 6 measurement settings, final projective outcome | T6 |
| A3 | [Martinez et al., PRA 102, 062426](https://arxiv.org/abs/2006.00109) (2020-12-24) | LLNL | not verified | 25,250 per state | not verified | 20 µs rectangular pulse | prepared 0/1 | T6 |
| A4 | [Navarathna et al., APL 119, 114003](https://arxiv.org/abs/2107.05857) (2021-09-13) | Transmons A and B, University of Queensland | 1 per sample | 16,384 traces per state (Sample A) | 50 samples at 100 MHz, from a 500 MSa/s digitizer | 2 µs | prepared g/e/f | T6 provisional (see Q2) |
| A5 | [Chen et al., arXiv 2103.08961](https://arxiv.org/abs/2103.08961) (2021-03-16); apparently published as [Duan et al., PR Applied 16, 024063](https://doi.org/10.1103/PhysRevApplied.16.024063) | USTC (Guo group) | 6-qubit chip | 8,000 records, 1,000 per each of 8 prepared states | 320 time points at 1.6 GS/s | 200 ns (computed) | prepared states | T6 |
| A6 | [Koolstra et al., PRX 12, 031017](https://arxiv.org/abs/2108.12023) (2022-07-26) | Transmon, weak continuous measurement, UC Berkeley QNL (Siddiqi) | 1 | see Q2 card | binned 40 ns, filtered 20 ns, in the file I opened; ADC 1 GS/s | up to 5 µs in the file I opened | preparation state, measurement axis, final readout result | T1 |
| A7 | [L. Chen et al., npj QI 2023](https://www.nature.com/articles/s41534-023-00689-6) (2023-03-16) | Chalmers | 3 | not verified | not verified | 140 ns per [search summary only](https://arxiv.org/abs/2208.05879), not verified | 3-state | T6 |
| A8 | [Luchi et al., PR Applied 20, 014045](https://arxiv.org/abs/2212.00080) (2023-07-20) | Transmon qutrit, LLNL hardware, Trento analysis | 1 | 16,000 (2-state) and 24,000 (3-state), 8,000 per state | sliced demodulation, rate not verified | not verified | prepared 0/1/2 | T6 |
| A9 | [Reuer et al., Nat. Commun. 14, 7138](https://www.nature.com/articles/s41467-023-42901-3) (2023-11-06) | ETH Zurich | not verified | not verified | not verified | not verified | not verified | T6 for traces |
| A10 | [Cao et al., arXiv 2402.09532](https://arxiv.org/abs/2402.09532) (2024-02-14, v4 2025-05-23) | Five hardware setups; affiliations Oxford, LBNL/Berkeley, RIKEN | not verified per setup | 5,000 traces per state (15,000 total) in the reported training run | 1 Gsps per ADC, 125 MHz carrier | not verified | prepared 0/1/2 | T6 |
| A11 | [You et al., arXiv 2312.07958](https://arxiv.org/abs/2312.07958) (2023-12-13) | Xmon in a 6-qubit lattice, SUSTech and Macau | 1 measured | 5,000 per state per SNR setting | not verified | not verified | prepared g/e, plus Rabi | T6 |
| A12 | [Khan et al., PRX Quantum 5, 020364](https://arxiv.org/abs/2310.18519) (2024-06-21) | Transmon, Hatridge lab (Pittsburgh) with Türeci (Princeton) | 1 | not verified | digitized at 1 GS/s, 50 MHz IF | not verified | prepared g/e/f | T6 (code only) |
| A13 | [Vora et al., QubiCML, arXiv 2406.18807](https://arxiv.org/abs/2406.18807) (2024-06-27); venue versions [QCE 2024](https://doi.org/10.1109/qce60285.2024.10332) and [GLSVLSI 2026](https://doi.org/10.1145/3787109.3816383) | QubiC RFSoC, LBNL | not verified | about 5,000 per class; training set "10,000 readout trajectories" per qubit | 2 GSPS ADC | not verified | prepared g/e/f | T6 |
| A14 | [Wang et al., PR Applied 23, 034046](https://arxiv.org/abs/2407.17407) (2025-03-20) | High-EJ/EC transmon, Rochester (Blok) | 1 | 8,000 + 2,000 traces per state, 10 states | not verified | not verified | prepared 0 to 9 | T6 |
| A15 | [Gunyhó et al., Nat. Electron.](https://www.nature.com/articles/s41928-024-01147-7) (2024-04-10) | Transmon read by a thermal detector (bolometer), Aalto | 1 | see Q2 card | see Q2 card | see Q2 card | pi-pulse on/off | T2 |
| A16 | [Di Guglielmo et al., IEEE TQE 2025](https://arxiv.org/abs/2501.14663) (2025; [DOI](https://doi.org/10.1109/TQE.2025.3604712)) | Frequency-tunable multi-transmon device, one transmon read, QICK on ZCU216, Purdue (Ma) and Fermilab | 1 | 1,000,000 | see Q2 card | see Q2 card | prepared 0/1 | T1 |
| A17 | [Jung et al., arXiv 2601.19057](https://arxiv.org/abs/2601.19057) (2026-01-27; also [Research Square](https://www.researchsquare.com/article/rs-9228881/v1)) | 8 fixed-frequency transmons, LBNL | 8 | about 25,000 per state, 75,000 per qubit | 2 samples per ns, 1,500 to 2,000 steps | 750 to 1,000 ns | prepared 0/1/2 | T6 |
| A18 | [Le et al., arXiv 2610.33843](https://arxiv.org/abs/2610.33843) (2026-10-24) | 8 fixed-frequency transmons, LBNL, QubiC | 8 | 25,000 per prepared state | 1,408 complex samples at 2 GSPS | 704 ns window | prepared 0/1/2, heralded | T6 |
| A19 | [Johnson et al., arXiv 2610.00783](https://arxiv.org/abs/2610.00783) (2026-09-30) | Transmon Q1 and a 3D ancilla, Fermilab SQMS with Illinois Tech | not verified | 50,000 per class at 8 µs | 307.2 MS/s decimated stream | 4, 6, 8, 13 µs | prepared g/e | T6 |
| A20 | [Gonzalez-Guerrero et al., npj Unconv. Comput.](https://www.nature.com/articles/s44335-026-00085-1) (2026-08-04) | Data from a separate study under review; co-authors include QubiC authors | not verified | not verified | not verified | not verified | not verified | T6 |

Abstract-level only, because the full text was paywalled, so all fields are not verified and each is T6 by default: [C. Chen et al., RSI 2025, dynamic adaptive readout](https://doi.org/10.1063/5.0239413); [W. Wang, EIECC 2024, deep learning readout of tunable qubits](https://doi.org/10.1109/eiecc64539.2024.10929552); [Lin and Chen, PRA 2026, transient response exclusion](https://doi.org/10.1103/hbz5-2fp9); [Cryo-CMOS readout IC with FCNN, VLSI 2026](https://doi.org/10.1109/vlsitechnologyandcir65830.2026.11577211); [Yesilyurt et al., QICK poster 2023, 92% vs 84% fidelity](https://www.osti.gov/biblio/1974720), which appears to be an earlier stage of A16.

**Group B: "reuses Lienhard" (not counted as new).**

- [HERQULES, Maurya et al., ISCA 2023](https://arxiv.org/abs/2212.03895): "Access to ADC data traces on IBM Quantum Cloud is restricted. In this paper, we use the dataset of a custom five-qubit chip provided by the authors of [26]."
- [Maurya et al., side-channel, QCE 2024](https://arxiv.org/abs/2405.08962): "a dataset of one-microsecond-long readout traces obtained from a five-qubit superconducting quantum computer, as described in Reference [8]".
- [Mude et al., DAC 2025](https://arxiv.org/abs/2405.08982): "We obtained datasets containing the readout time traces collected directly from the ADC originating from a five-qubit chip used in Ref. [1]". Three-level, 243-class analysis, so possibly further data from the same chip, but the same device and lab.
- [Gautam et al., 2024, later titled "Quantized NN Workflow", DOI 10.1145/3814576.3814580](https://arxiv.org/abs/2407.03852): five-qubit processor, "consult Ref. [13]" (Lienhard), 500 samples each of I and Q at 500 MHz.
- [KLiNQ, Guo et al., DAC 2025](https://arxiv.org/abs/2503.03544): "real-measured experimental data from a superconducting quantum information processor comprising five qubits [3]".
- [Kent et al., PR Applied 25, 044009](https://arxiv.org/abs/2506.15771): "the five-qubit dataset analyzed by Lienhard et al." and also "three single-qubit datasets analyzed by Khan et al.".
- [MCMit, Giortamis et al.](https://arxiv.org/abs/2604.25863): "real readout traces from a five-qubit QPU described by Lienhard et al.".
- [Otting et al., Mamba](https://arxiv.org/abs/2607.11668): 1.6 M traces, 50,000 per state, 500 MHz, 2 µs, first 1 µs used.
- [Oraqle, Giortamis et al.](https://arxiv.org/abs/2608.01939): uses Lienhard for all readout and discriminator experiments.
- [Dillon et al., spiking networks](https://arxiv.org/abs/2610.02129): "the multiplexed-readout dataset introduced by Lienhard et al.".
- [Etim and Szefer, fault attacks](https://arxiv.org/abs/2512.20077): test input is "IQ data from [2]" where [2] is HERQULES. How they obtained it is not stated.
- Probable but not verified (Lienhard or Guo co-authors, paper text not accessible or silent): [Liu et al., ISCAS 2025](https://doi.org/10.1109/iscas56072.2025.11043488), [Du et al., NIAQ, DAC 2026](https://doi.org/10.1145/3770743.3808043), [Guo and Schulz, SE 2026](https://dl.gi.de/handle/20.500.12116/48328).

**Group C: reuses Di Guglielmo.** Both independently confirm the T1 record is usable.

- [LUNA, Farooq et al., 2025](https://arxiv.org/abs/2512.07808): "We use the publicly available superconducting-qubit readout dataset (time-series I/Q traces, labeled 0/1) used by Guglielmo et al.", citing 10.5281/zenodo.14427490. It reports a 90,000/10,000 split, smaller than the 900,000/100,000 in the record, so it presumably subsampled.
- [SNAC-Pack 2.0, Weitz et al., 2026](https://arxiv.org/abs/2605.16138): "superconducting qubit I/Q signal data Guglielmo et al. (2025)".
- [AI Benchmark Democratization and Carpentry, 2025](https://arxiv.org/abs/2512.11588) lists "Single Qubit Readout on QICK System" as a benchmark entry.

**Group D: integrated I/Q only, real hardware, no public data found.** These are not time-domain evaluations. They would be T3 if the data were public, so they are T6.

- [Quiroga et al., 2021, ibmq_bogota](https://arxiv.org/abs/2112.00313). The paper notes that "Level 0" trajectories exist as an input type.
- [Azad and Zhang, 2022, IBM five-qubit systems](https://arxiv.org/abs/2210.08574).
- [Utt et al., DATE 2023](https://www.cise.ufl.edu/research/cad/Publications/date23quantum.pdf) and [ACM TQC 2024](https://doi.org/10.1145/3644823): works on post-measurement I/Q distributions. The hardware source is not verified.
- [Zhang et al., Lifting, 2026, ibm_fez, 10,000 shots per class shown](https://arxiv.org/abs/2610.18482).
- [Acampora et al., ACM TQC 2026](https://doi.org/10.1145/3847660): abstract only, data source not verified.

**Group E: simulated traces (T5).**

- [Rinaldi et al., 2021, QuTiP stochastic trajectories, 1,000 per state](https://arxiv.org/abs/2112.05332).
- [Angelatos et al., PRX 2021](https://arxiv.org/abs/2011.09652).
- [Lema et al., 2025, code at GitHub with `--simulate 1`, no data files listed](https://github.com/vadimoganesyan/2512.14583).
- [Awad, Research Square 2026, 40,000 simulated shots of N=100 samples at 20 ns](https://www.researchsquare.com/article/rs-10619196/v1).
- [Awad, IEEE Access 2026, "a controlled simulation study"](https://doi.org/10.1109/access.2026.3722871).
- [Barillas Rodriguez, Barcelona MSc thesis 2026, synthetic raw ADC traces built from IQ statistics](https://diposit.ub.edu/items/80b41c3e-bcf0-4299-9461-d07106d5d66f/full).

**Checked and excluded (not discriminator evaluations on traces).**

- [Senanian et al., analog quantum reservoir](https://arxiv.org/abs/2312.16166): classifies input microwave signals, not qubit states. Its notebook names the axes (classes, data, shots, bits) and prints shapes such as (10, 100, 400, 10, 4), which points to per-shot bits rather than analog traces. Public data is at [Zenodo 10432778](https://zenodo.org/records/10432778): 12 files, 12.86 GB, CC-BY-4.0, 2023-12-26. The record describes itself as data and code to reproduce the figures, and I did not open the large tar files beyond the first file name.
- [Carles et al., quantum reservoir](https://arxiv.org/abs/2506.22016): data at [Zenodo 15745370](https://zenodo.org/records/15745370), 3.05 GB, file shapes not verified.
- [Cüce et al.](https://arxiv.org/abs/2601.12168), [Bello et al.](https://arxiv.org/abs/2602.14333) and [Artag et al.](https://arxiv.org/abs/2605.15882): theory or simulation.
- [Hoang et al., KAN](https://arxiv.org/abs/2602.02056): streaming single I/Q points, dataset not identified.
- [Butko et al.](https://arxiv.org/abs/2601.08213): says real-qubit validation is future work, dataset not identified.
- Readout hardware papers with no trace link in the arXiv text and no ML discriminator as the focus: [Hazra et al.](https://arxiv.org/abs/2407.10934), [Spring et al.](https://arxiv.org/abs/2409.04967), [Swiadek et al.](https://arxiv.org/abs/2307.07765), [Castellanos-Beltran et al.](https://arxiv.org/abs/2501.01185), and [Di Giovanni et al.](https://arxiv.org/abs/2502.08589) ("experimental data are available upon request").

For context, the excluded dataset itself: [Lienhard et al., PR Applied 17, 014024](https://arxiv.org/abs/2102.12481), 500 MSa/s, MIT. It is described in [Oraqle's repository](https://github.com/manosgior/Oraqle/blob/main/dataset/README.md) as 1.6 M shots, 1,000 samples, 2 µs, 32 labels, in two HDF5 files of 7.68 GB and 17.92 GB.

### Inferences
- A16 and A6 show that single-qubit raw traces have been public for some time: the Koolstra Drive folder dates to 2021 and the Di Guglielmo record to December 2024.
- A17 and A18 describe matching hardware (8 fixed-frequency transmons, 25,000 shots per state, 2 GSPS, LBNL), so they may be the same acquisition. That is my inference; neither paper says so in the text I read. A13 (QubiCML) and A20 likely draw on the same group.
- The Oraqle sentence that using Lienhard is "consistent with all SOTA qubit-state discriminator evaluations" is too strong as a general statement, given Group A and Group C. It may still hold for the specific discriminators Oraqle compares, apart from QubiCML, whose own evaluation used LBNL data.

### Gaps
- Group A rows carry many "not verified" fields because I extracted targeted passages, not full readings. Fields could be filled in by reading each paper in full.
- I screened citers of [Flurin](https://api.openalex.org/works/W2902080782), [Reuer](https://api.openalex.org/works/W4388410078) and [Angelatos](https://api.openalex.org/works/W3099427004) by title keywords only, so a relevant paper with an unusual title could have been missed.
- I did not open [Accurate Leakage Speculation for QEC, MICRO 2025](https://doi.org/10.1145/3725843.3756053), which cites Mude and Luchi, so its data source is unknown.

## Q2. What does each data-availability statement say, and what actually exists at the linked location?

### Takeaway
Only three discriminator-relevant records hold real, per-shot, time-resolved superconducting traces that I could confirm without asking anyone: Di Guglielmo (T1), Koolstra (T1) and Gunyhó (T2). The claim "no other raw-trace dataset is publicly available" is therefore false as written. None of the three is multi-qubit and frequency-multiplexed, so a narrower claim limited to that setting survives this sweep. One more candidate (Navarathna at UQ) says it is open, but I could not confirm the files.

### Cited Findings

**T1 card: Di Guglielmo et al. 2025 (counterexample).**
- Statement: "we have made our dataset available on Zenodo [44] and the code to reproduce the algorithm training and implementation available at Ref [21]. We hope that this will be useful for future studies and encourage more datasets and benchmarks to be made available" — [arXiv 2501.14663](https://arxiv.org/abs/2501.14663). The repository README points to the same record — [GitHub](https://github.com/fastmachinelearning/ml-quantum-readout).
- Record: [Zenodo 14427490](https://zenodo.org/records/14427490), DOI 10.5281/zenodo.14427490, creator Du, Botao, published 2024-12-12, licence CC-BY-4.0, empty description, 4 files, 12,328,000,512 bytes — [Zenodo API](https://zenodo.org/api/records/14427490).
- Shapes, read from `.npy` headers with HTTP range requests, no full download: `0528_X_train_0_770.npy` (900000, 1540) `<f8`; `0528_X_test_0_770.npy` (100000, 1540) `<f8`; `0528_y_train_0_770.npy` (900000,) `<f8`; `0528_y_test_0_770.npy` (100000,) `<f8` — [Zenodo files](https://zenodo.org/records/14427490).
- Labels: values are 0 and 1, block-ordered with the change at index 50,000 in test and 450,000 in train (sampled by range read around those points). First X row holds integer-valued ADC units, for example 284, 1301, -670, 374 — [Zenodo files](https://zenodo.org/records/14427490). The paper says "500,000 shots each for qubits prepared in 0 and 1" — [arXiv](https://arxiv.org/abs/2501.14663). The repo's [`generate_window.py`](https://github.com/fastmachinelearning/ml-quantum-readout/blob/dev/data/qick_data/generate_window.py) slices columns `lo*2:hi*2`, which suggests alternating I and Q values (my inference; 770 × 2 = 1540).
- Timing: "the readout pulse length of 2.5 µs corresponds to 770 ADC clock cycles" and 100 clock cycles is about 325 ns; ADCs run at 6.88 GS/s and are demodulated on the FPGA to I(t) and Q(t) — [arXiv](https://arxiv.org/abs/2501.14663). That gives 3.25 ns per sample. [Johnson et al.](https://arxiv.org/abs/2610.00783) separately name a "307.2-MS/s decimated stream" on QICK, which matches.
- Conflict: the [repository data README](https://raw.githubusercontent.com/fastmachinelearning/ml-quantum-readout/dev/data/README.md) describes QICK data as "2000 ns with 2.6 ns sampling rate" and X shape (909000, 1440). That describes a different, older version from the Zenodo files, so trust the paper for the 770-sample record.
- Preparation quality: thermal excited population about 6%; a cooling step brings ground-state preparation below 1.6% thermal population — [arXiv](https://arxiv.org/abs/2501.14663).
- Other data links in the same README: a Dropbox "Partitioned" link for Version 20230529 returned HTTP 200 but I could not see its contents; a Purdue SharePoint link returned HTTP 403; a "Quantum Machine" dataset (single and 2-qubit multiplexed, `00002_IQ_plot_raw.h5`, X (9000, 2000), 1 ns sampling) sits behind a wrapped SharePoint link that I did not reach; "Version 20240501" links to `#` — [README](https://raw.githubusercontent.com/fastmachinelearning/ml-quantum-readout/dev/data/README.md). The repo reports no licence — [GitHub API](https://api.github.com/repos/fastmachinelearning/ml-quantum-readout).

**T1 card: Koolstra et al. 2022 (counterexample, with caveats).**
- Statement: arXiv v2 says "A link to the training data will be available here later" and links only code — [arXiv 2108.12023](https://arxiv.org/abs/2108.12023). The PRX page shows no data statement, only the code reference — [PRX](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.031017). The repository README, however, says "Download the dataset" with a Google Drive link — [GitHub](https://github.com/qnl/trajectories_lstm).
- Location: a [Google Drive folder titled "Public Data Repository"](https://drive.google.com/drive/folders/1FNnrNAwe4NaeiUjSvbZOvkd-el_4MsxL) with 8 `cts_rabi_qubit_ge_amp_*` folders and `rabi_modulation_data`, modified 3/28/21. Range requests returned HTTP 206 with no login.
- File sizes seen: 3,017,198,368 B (`amp_0.0000`, `two_prep_states_dt_40ns_meas_XYZ.h5`), 1,101,001,976 B (`amp_0.0103`), 1,471,962,968 B (`amp_0.0171`), 2,516,011,016 B (`rabi_modulation_data`). Total for all folders is not verified. The folders also hold `settings.yaml` and PNGs.
- Structure of the `amp_0.0103` file, read as HDF5 metadata over range requests (about 2 MB fetched): `prep_g/meas_{X,Y,Z}/t_{0..50}/` with `I_binned`, `Q_binned`, `I_binned_filtered`, `Q_binned_filtered`, `final_ro_results`, `time_axis_*`, `dt_*`, `cutoff_freq`. Example `t_50`: `I_binned` (2357, 126), `I_binned_filtered` (2357, 252), `final_ro_results` (2357,) with values 0 and 1, `dt_binned` 40 ns, `dt_filtered` 20 ns, `cutoff_freq` 7.5 MHz, last time point 5 µs. Other slices I saw hold between 2,335 and 2,366 shots.
- `settings.yaml` for `amp_0.0000`: `n_reps` 4000, `unbinned_dt` 1 ns, `binned_dt` 40 ns, `prep_states` [+Y, -Y], `t_stop` 8 µs, `t_step` 200 ns, `target_qubit` 6 — [Drive](https://drive.google.com/drive/folders/1FNnrNAwe4NaeiUjSvbZOvkd-el_4MsxL).
- Hardware: voltage records digitized at 1 GS/s; "largest dataset of 0.8 × 10^5 voltage records" — [arXiv](https://arxiv.org/abs/2108.12023). GitHub reports no licence for the code; the Drive folder states none — [GitHub API](https://api.github.com/repos/qnl/trajectories_lstm).
- Caveat: this is weak continuous measurement under a Rabi drive, used to reconstruct trajectories with an LSTM, not a standard projective dispersive readout benchmark. The labels are preparation state, measurement axis and a final strong readout result. Binned I/Q, not raw ADC samples.

**T2 card: Gunyhó et al. 2024.**
- Statement: "The data that support the findings of this study are available via Zenodo at https://doi.org/10.5281/zenodo.7773980" — [Nature Electronics](https://www.nature.com/articles/s41928-024-01147-7).
- Record: that DOI resolves to [Zenodo 10000511](https://zenodo.org/records/10000511), published 2023-10-13, CC-BY-4.0, one zip of 922,898,009 B holding 35 entries and 1.89 GB uncompressed: 22 NetCDF files plus notebooks and scripts — [Zenodo API](https://zenodo.org/api/records/10000511).
- Shape, read from one member extracted by range and then deleted: `data/bolometer-characterization-timetrace-76.nc` has `readout_meta_CH0_re` and `readout_meta_CH0_im` of shape (2, 1000, 781) float64, with `qubit_pi_pulse_amplitude` [0, 1], `readout_meta_shot_index` 0 to 999, and `readout_meta_time` from 0 to 49.92 µs in 64 ns steps (15.625 MS/s). The bundled notebook calls this "run 76: full timetraces" and computes fidelity for readout windows.
- Why T2 rather than T1: only 1,000 shots per state in this run, and the format is documented only through notebooks. Larger `qubit-bolometer-readout-*.nc` files (up to 431 MB uncompressed) were not opened, so they might contain more shots.
- Caveat: the detector is a bolometer, not a parametric-amplifier chain, and the discriminator is a threshold on integrated windows, not an ML classifier.

**T3 records (real hardware, per-shot, no time axis).**
- [Arnold et al., all-optical readout](https://arxiv.org/abs/2310.16817), [Zenodo 14033026](https://zenodo.org/records/14033026), 32,029,038 B, CC-BY-4.0, 2024-11-06. `Fig_4a/IQblobs_0Hz.mat` holds `I_g`, `Q_g`, `I_e`, `Q_e` and second-measurement copies, each (1, 50000) float64 (member opened, then deleted). Not an ML paper.
- [Xin et al., PRX Quantum 2026](https://arxiv.org/abs/2511.17460), [4TU record](https://data.4tu.nl/datasets/902d7f9e-38bf-48a2-a2f7-52bbf7aeeedf), `Data.zip` 6,494,638,141 B, CC BY 4.0, 2026. README: "Data provided for Figure 3 and S1 are analog voltages (I-Q...)", other figures hold "classified measurement outcomes (0, 1, or 2)" and probabilities. 17-qubit processor. I did not open the HDF5 files, so per-shot status of the I-Q files is not verified.

**T4 records.** This paper-first sweep verified none. The ETH Reuer `.dat` files might include averaged traces, but I saw only file names and sizes.

**T6 and unresolved: statement versus what exists.**

| Paper | Statement seen | What exists |
|---|---|---|
| Navarathna 2021 | "openly available in UQ eSpace at" [10.48610/9d3c2eb](https://doi.org/10.48610/9d3c2eb) ([arXiv](https://arxiv.org/abs/2107.05857)) | [RDA record "APL paper files"](https://researchdata.edu.au/apl-paper-files/3369753): Access Open, UQ reuse terms. Second record [10.14264/ab6f55f](https://researchdata.edu.au/data-supporting-neural-state-classification/3369705): "Contains data for Sample A, fidelities for the 24 hour plot and data for the phase robustness". eSpace returned 403 to scripted clients and WebFetch; the RDM API returned 401. File list **not verified**. Could be T1 or T2 if the Sample A traces are there |
| Reuer 2023 | arXiv: "available from the corresponding authors upon reasonable request" ([arXiv](https://arxiv.org/abs/2210.16715)). Published: deposited at [10.3929/ethz-b-000637125](https://www.nature.com/articles/s41467-023-42901-3) | [ETH record](https://doi.org/10.3929/ethz-b-000637125): `SourceData.zip`, 672,801 B, 24 `.dat` files named after figures, CC BY 4.0, 2023-10-18. Figure source data only |
| Khan 2024 | "Source code available at" Zenodo ([arXiv](https://arxiv.org/abs/2310.18519)); [PRX Quantum page](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020364) shows no data link | [Zenodo 10976215](https://zenodo.org/records/10976215): 2 zips, 185,178 B, software, CC-BY-4.0. Code only. Kent et al. still used its single-qubit data, so it was shared privately or by another route (not stated) |
| Vora 2024 | "All data relevant to the study are available from the corresponding authors upon request." ([arXiv](https://arxiv.org/abs/2406.18807)) | Nothing public found |
| Gonzalez-Guerrero 2026 | "not publicly available due to originally generated for and presented in a separate study... available from the corresponding author on reasonable request" ([Nature](https://www.nature.com/articles/s44335-026-00085-1)) | Nothing public |
| L. Chen 2023 | "available on request" ([Nature](https://www.nature.com/articles/s41534-023-00689-6)) | Nothing public found |
| Di Giovanni 2025 | "experimental data are available upon request" ([arXiv](https://arxiv.org/abs/2502.08589)) | Nothing public found |
| Magesan, Flurin, Martinez, Chen/Duan, You, Wang, Cao, Jung, Le, Johnson, Mude, Maurya (side-channel), Gautam, HERQULES, KLiNQ, Kent, Dillon, Otting, Etim | None in arXiv text ([Jung abs page](https://arxiv.org/abs/2601.19057), [Cao abs page](https://arxiv.org/abs/2402.09532), [You abs page](https://arxiv.org/abs/2312.07958), [Johnson abs page](https://arxiv.org/abs/2610.00783), [Dillon abs page](https://arxiv.org/abs/2610.02129) also list no data links) | Nothing found; journal versions not read |
| Luchi 2023 | None in arXiv text; [APS page](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.20.014045) shows none but the full text is paywalled | Nothing found |
| Kent 2026 | [APS page](https://journals.aps.org/prapplied/abstract/10.1103/bnwn-d2p4) shows none; full text paywalled | Nothing found |
| MCMit, Oraqle | Code artifacts only | [Zenodo 21485912](https://zenodo.org/records/21485912): 32 MB software zip. [Oraqle README](https://github.com/manosgior/Oraqle): "not public and is not redistributed here" |

**Public re-host of the Lienhard dataset: none found.**
- [Oraqle's `dataset/README.md`](https://github.com/manosgior/Oraqle/blob/main/dataset/README.md) states "no Zenodo/figshare/GitHub release exists" and that HERQULES and KLiNQ obtained it from the authors. [HERQULES](https://arxiv.org/abs/2212.03895) confirms "provided by the authors".
- Web searches for the exact filenames `DRaw_C_Tr_v0-001` and `DRaw_C_Te_v0-002`, for KLiNQ and HERQULES repositories, and for "32 basis states 50000 traces" on Hugging Face, Kaggle or Zenodo returned no host. These are search-engine results, so a private or unindexed copy cannot be excluded.
- [`manosgior/MCMit`](https://github.com/manosgior/MCMit) has `qubit_state_discrimination/data/single_qubit_data/MF_RMF_q{1..5}.pkl` at 16,287 B each, too small to hold traces (not opened). The largest non-firmware files in the tree are notebooks and PDFs.
- [`manosgior/Oraqle`](https://github.com/manosgior/Oraqle) holds predictions and checkpoints (largest about 19.6 MB), not traces.

**Places and queries that found nothing new.**
- arXiv API: 36 phrase queries combining "qubit readout" with neural network, machine learning, transformer, recurrent, autoencoder, hidden Markov, matched filter, reservoir, spiking, knowledge distillation, FPGA, hls4ml, QICK, QubiC, leakage, qudit, fluxonium, RFSoC and others. 60 unique results; several queries returned 0. The API then started answering HTTP 429.
- Web search for: public raw time-domain traces plus Zenodo; "publicly available" time-resolved I/Q benchmark; GitHub and Google Drive trace datasets; fluxonium, cat qubit and dual-rail with Zenodo; `meas_level 0` raw kerneled traces at IBM. The IBM query surfaced no public level-0 dataset, consistent with [HERQULES](https://arxiv.org/abs/2212.03895) saying that access is restricted.
- OpenAlex keyword queries: "readout traces superconducting qubit dataset" (1 hit), "qubit readout hidden Markov", "qutrit readout neural network", "mid-circuit measurement discriminator superconducting" (0), and others, with no new public dataset.
- Ancillary files: arXiv abs pages for 2610.02129, 2601.19057, 2402.09532, 2312.07958, 2108.12023 and 2610.00783 show none.

### Inferences
- The draft claim should change. It is false in general. A defensible narrower form, based on this sweep, is along the lines of "to our knowledge, no other *frequency-multiplexed multi-qubit* raw-trace dataset is publicly available; single-qubit datasets exist (Di Guglielmo et al.; Koolstra et al.)". The separate repository sweep and other platforms could still overturn the narrower form.
- The same Oraqle sentence also implies that Lienhard's data is itself public: "(no other raw-trace dataset is publicly available)" — [arXiv 2608.01939](https://arxiv.org/abs/2608.01939). Oraqle's own [README](https://github.com/manosgior/Oraqle/blob/main/dataset/README.md) says it is "not publicly downloadable". The wording is internally inconsistent regardless of the counterexamples.
- Di Guglielmo has 1 M single-qubit shots, roughly 10× the per-qubit trace data of many own-lab papers here, but it is a single qubit at 96% fidelity, so it cannot test crosstalk claims.

### Gaps
- Navarathna at UQ is the most important open item. Someone should open [https://doi.org/10.48610/9d3c2eb](https://doi.org/10.48610/9d3c2eb) in a browser. I stopped trying after scripted access was refused, and the permission system blocked further probing of the RDM API.
- Data-availability sections of paywalled APS, ACM and IEEE full texts could not be read: Luchi, Khan (only the abstract page), Kent, Martinez, Flurin, Liu, Du, Acampora, Wang (PR Applied), Mude and KLiNQ published versions. A statement there could differ from the arXiv text.
- Koolstra: total size and number of shots across all nine folders are not verified. From what I saw, each `amp` file holds roughly 2,350 shots per slice, 51 slices and 3 axes in one preparation state, around 360,000 records, which is my arithmetic and well above the paper's "0.8 × 10^5" for its largest dataset. I have not resolved that mismatch; slices at different durations may not be independent training records.
- I did not open the QICK repository's in-repo CSVs `data/fpga_testing/malab_{e,g}_state_A_fpga.csv` (38 MB each, described as "Mock readout data") — [tree](https://github.com/fastmachinelearning/ml-quantum-readout/tree/dev/data/fpga_testing).

## Q3. Which papers citing Lienhard 2022, HERQULES, KLiNQ or QubiCML from 2024 to 2026 introduce a new dataset of their own?

### Takeaway
At least 10 citing papers from 2024 to 2026 use new hardware data, but only two of those datasets are publicly downloadable: Di Guglielmo (T1) and Gunyhó (T2). Everything else is on request, unstated or code-only. Citing sets came from [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2102.12481/citations) (Lienhard 68, HERQULES 51, KLiNQ 7, QubiCML 17) and [OpenAlex](https://api.openalex.org/works?filter=cites:W3133392172) (Lienhard 62, HERQULES 33, KLiNQ 4, QubiCML 4).

### Cited Findings

**New own-hardware data, with public status.**

| Paper | Cites | New data | Public? |
|---|---|---|---|
| [Di Guglielmo 2025](https://arxiv.org/abs/2501.14663) | Lienhard, HERQULES | Purdue/Fermilab single transmon, 1 M shots | **Yes, T1**, [Zenodo 14427490](https://zenodo.org/records/14427490) |
| [Gunyhó 2024](https://www.nature.com/articles/s41928-024-01147-7) | Lienhard | Aalto bolometer readout | **Yes, T2**, [Zenodo 10000511](https://zenodo.org/records/10000511) |
| [Khan 2024](https://arxiv.org/abs/2310.18519) | Lienhard, HERQULES | Pittsburgh transmon | No, code only |
| [Vora 2024](https://arxiv.org/abs/2406.18807) | Lienhard, HERQULES | LBNL | No, on request |
| [Cao 2024](https://arxiv.org/abs/2402.09532) | Lienhard, QubiCML | Oxford and others, five setups | None stated |
| [Wang 2025](https://arxiv.org/abs/2407.17407) | Lienhard | Rochester 10-level transmon | None stated |
| [Jung 2026](https://arxiv.org/abs/2601.19057) | Lienhard, QubiCML | LBNL 8 transmons | None stated |
| [Le 2026](https://arxiv.org/abs/2610.33843) | Lienhard, HERQULES | LBNL, possibly same as Jung (my inference) | None stated |
| [Gonzalez-Guerrero 2026](https://www.nature.com/articles/s44335-026-00085-1) | Lienhard, QubiCML | From a separate study under review | No, on request |
| [Zhang 2026, Lifting](https://arxiv.org/abs/2610.18482) | Lienhard, HERQULES, QubiCML | ibm_fez integrated I/Q, not traces | None stated |

Not verified, abstract only: [C. Chen RSI 2025](https://doi.org/10.1063/5.0239413), [W. Wang EIECC 2024](https://doi.org/10.1109/eiecc64539.2024.10929552), [Acampora 2026](https://doi.org/10.1145/3847660), and [Utt 2024](https://doi.org/10.1145/3644823), the last of which appears to use IBM I/Q.

**Citing papers that do not add new hardware data.**
- Reuses Lienhard: Maurya 2024, Mude 2025 (same chip), Gautam 2024, KLiNQ 2025, Kent 2026 (plus Khan's data), MCMit, Otting, Oraqle, Dillon, Etim (all in Q1 Group B).
- Reuses Di Guglielmo: LUNA, SNAC-Pack 2.0 (Q1 Group C).
- Simulation: [Awad IEEE Access 2026](https://doi.org/10.1109/access.2026.3722871), which is in the Lienhard, KLiNQ and Di Guglielmo citing sets.

Forward citers of [Di Guglielmo](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2501.14663/citations) (Semantic Scholar 23, OpenAlex 12) added one own-hardware paper, [Johnson 2026](https://arxiv.org/abs/2610.00783) (Fermilab SQMS), with no data link.

### Inferences
- The public-data contribution rate is low: 2 out of at least 10 new-data citing papers. LBNL-affiliated authors appear on at least 4 of the others (Vora, Jung, Le, Cao), so that is the most obvious group to ask.
- The field's two reuse clusters (Lienhard, Di Guglielmo) follow availability pressure: HERQULES names restricted IBM access as its reason, and LUNA names public availability.

### Gaps
- Google Scholar was not accessed, so its larger citing set may contain papers neither Semantic Scholar nor OpenAlex lists. Semantic Scholar was rate-limited (HTTP 429) for most of the session, but all six seeds eventually returned. OpenAlex warned of a shared daily budget, and I ran about 15 keyword queries rather than the 22 I planned.
- Non-arXiv conference papers (ACM, IEEE) could mostly not be opened, so data origin for Liu, Du and Acampora remains open.
