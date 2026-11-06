# Public raw single-shot readout datasets on non-superconducting platforms

Checked 2026-11-06. Tiers follow the assignment (T1 to T6). Three conventions fill gaps in that scheme:
- **T6 (figure-level)**: a deposit exists but holds only plotted or aggregate values, with no per-shot data and no time axis, so raw traces are not actually obtainable from it.
- **T3-image**: per-shot camera frames with no time axis. Neutral-atom images are listed separately under their own question, as requested.
- **Not classified**: I could not open the deposit, so no tier is claimed.

"How verified" terms: *header* means an npy, HDF5 or TIFF header read through an HTTP range request. *Sample* means a small member was opened (`allow_pickle=False`, nothing unpickled or executed; samples deleted afterwards). *Listing* means a repository API or ZIP central directory. No file above about 50 MB was downloaded in full.

## Trapped ions: is there public time-resolved photon-arrival, PMT or per-shot camera data?

### Takeaway
I found no public per-shot, time-resolved ion readout record, so nothing at T1 or T2. The best deposits are two T3 sets of per-shot integrated counts. The machine-learning readout papers (Seif, Ding) have no public data, and Sandia QSCOUT's deposit holds outcome probabilities only.

### Cited Findings

**Jeon et al. 2025: T3**
- Paper: "Two-Mode Bosonic State Tomography with Single-Shot Joint-Parity Measurement of a Trapped Ion", PRX Quantum 6, 040352 (2025) — [Crossref](https://api.crossref.org/works/10.1103/wsqr-j9f4)
- Deposit: DOI 10.5281/zenodo.17626177 — [Zenodo](https://zenodo.org/records/17626177)
- Tier evidence: the record says each dataset is "a N × (2 + m) matrix. The first and the last columns represent the photon count and the shot index". A sample `datasets/cat` had shape (4200, 4) float64 (how verified: sample). That is one count per shot and no time axis — [Zenodo](https://zenodo.org/records/17626177)
- Platform and lab: single trapped ion with two motional modes; creators Jeon, Kang, Choi, Kim, You, Kim. Ion species and lab: not verified.
- Shots: 3,807 HDF5 files; 4,200 rows in the sampled dataset. Samples per trace: 1. Sample rate: not applicable. Detection duration: not verified.
- Labels: scan-parameter columns; state assigned by photon-count threshold; calibration in `detection_contrast.npz`. No per-shot prepared-state column — [Zenodo](https://zenodo.org/records/17626177)
- Format, size, licence, date: ARTIQ HDF5 in `dataset.zip`, 41.5 MB (357 MB uncompressed), CC-BY-4.0, 2025-11-17 — [Zenodo](https://zenodo.org/api/records/17626177)

**Canteri et al. 2025 ("Photon-Interfaced Ten-Qubit Register of Trapped Ions"): T3**
- Deposit: DOI 10.5281/zenodo.17748444 — [Zenodo](https://zenodo.org/records/17748444)
- Tier evidence (sample): `data/ion_data/181626/camera_ion1.txt` has 100 rows by 105 columns with header `scan point, MeanExc, ion1(sum), ion1(0), ion1(1), ...`, holding values around 1.5e5 to 3e5. I read these as per-shot, per-ion camera counts. Column meaning is not documented; I found no README. `data/photon_data/*.txt` are `Time stamps, Channel` lists (40,368 events in the file opened) that `load_data.m` calls time tags. They appear to record emitted photonic qubits rather than ion-state readout — [Zenodo](https://zenodo.org/records/17748444)
- Platform and lab: string of ten ions; creators Canteri, Koong, Bate, Winkler, Krutianskii, Lanyon. Species and lab: not verified.
- Shots: about 102 per scan point (inferred from columns), 6 runs. Samples per trace: 1. Readout duration: not verified.
- Labels: measurement-basis and scan settings only.
- Format, size, licence, date: text, `.mat`, MATLAB scripts; 116 MB zip (132 MB uncompressed); CC-BY-4.0; 2025-11-28 — [Zenodo](https://zenodo.org/api/records/17748444)

**Figure-level or unavailable (T6)**
- Seif et al. 2018, J. Phys. B 51, 174006: the data format matches the requested type ("photon counts and their arrival times on M≥N PMT channels", "sub-μs resolution", 80,000 samples per label for 3 qubits, 50,000 per label for 5), but the text has no data statement or link, and DataCite returns only the arXiv record for the title. Prepared-state labels exist in the paper's experiment, not in any public file — [ar5iv](https://ar5iv.labs.arxiv.org/html/1804.07718)
- Ding et al. 2019 (arXiv:1810.07997): single 171Yb+ ion, 2×10⁵ training samples, 100 sub-bins of 3 μs in simulation. No data statement. The only link is a code repository, and that URL now returns 404 — [arXiv](https://arxiv.org/html/1810.07997v2); [GitHub](https://github.com/quantumira-cle/On_board_FNN_qubit_discrimination)
- NIST trap-integrated SNSPD deposits mds2-2956 and mds2-3216: both READMEs say "Numerical values of all data points shown in figures", not raw records. Figure-level — [mds2-2956 README](https://data.nist.gov/od/ds/mds2-2956/2956_README.txt); [mds2-3216 README](https://data.nist.gov/od/ds/mds2-3216/3216_README.txt)
- Sandia QSCOUT: the one dataset found is "Solovay-Kitaev Algorithm and Randomized Compilation Data Availability" (833 KB, JSON). The page says the 3D arrays hold "outcome probabilities". Figure-level, no per-shot record — [Sandia](https://www.sandia.gov/research/datasets/details/solovay-kitaev-algorithm-and-randomized-compilation-data-availability-2026-04-13/); [DataCite](https://api.datacite.org/dois/10.11578/3815430)
- Balaniuk MSc thesis (Waterloo 2025): uses a "Raw fluorescence dataset" from the Rajibul Islam group, with 8, 16 and 32 ions and "mainly unlabelled" data. Keyword search of the full text found no repository link — [UWSpace](https://uwspace.uwaterloo.ca/bitstreams/ee16b006-69a4-45c0-a7ca-35f3643b7eba/download)
- Site-AIT, "300-ion fluorescence readout": its README calls it a "code-and-checkpoint release", and users must supply `data/intersection_train_data`. Code only — [GitHub](https://github.com/JUNGJEIHUANG/Trapped-Ion-Qubits-readout)
- `createamazing/Trapped-Ion-Qubits-readout`: the page lists only code files — [GitHub](https://github.com/createamazing/Trapped-Ion-Qubits-readout)

### Inferences
- Ion groups publish count-level or probability-level data; time-tagged readout records seem to stay private. Seif and Ding show such data exists internally.
- An ML-readout paper on ions cannot currently be benchmarked against public data at the photon-arrival level.

### Gaps
- Harty et al. 2014, Myerson et al. 2008, Todaro et al. 2021 and Reens et al. 2022: publisher pages were not readable (APS blocked, no Europe PMC full text), so their data statements are not verified. DataCite and Zenodo title searches found no deposit for them.
- Crain et al. 2019 (SNSPD readout): not verified; my arXiv title lookup returned nothing.
- Zhukas et al. (arXiv:2006.12801): my keyword scan of the HTML found no data statement or repository link — [arXiv](https://arxiv.org/abs/2006.12801). The published version was not checked.

## Semiconductor spins: are there public single-shot charge-sensor, RF-SET or dispersive time traces?

### Takeaway
Yes. One deposit meets T1: Mądzik et al. (UNSW), with raw SET current traces whose prepared electron state is derivable from record position. Struck, Zwerver and Zheng are real per-shot time-resolved sets at T2, totalling about 1.37 million traces. Further UNSW Dryad deposits describe raw traces I could not open. The "reasonable request" statement in Struck's paper is wrong in practice, because the data is on Zenodo.

### Cited Findings

| Record | Tier | Traces × samples | Rate |
|---|---|---|---|
| Mądzik 2022, Figshare | T1 | ED Fig. 5: 601 files × 160 readouts × 208 | 100 kHz |
| Struck 2021, Zenodo | T2 | 10⁶ × 491, plus 25,000 × 495 | 62.5 kHz |
| Zwerver, Zenodo | T2 | 336,650 × 80 | 7.629 kHz |
| Zheng 2019, 4TU | T2 | 10,000 × 768 | 1 MHz |
| Stemp 2026 and other UNSW Dryad | T2 | not verified | 500 kS/s (README) |
| Krzywda, Kelly, Philips | T3 | one value per shot | n/a |
| Takeda 2024 | T4 | 2 averaged traces × 31,220 | 6.25 MHz |

**Mądzik et al. 2022: T1**
- Paper: "Precision tomography of a three-qubit donor quantum processor in silicon", Nature (2022). Its data statement points to Figshare — [Nature](https://www.nature.com/articles/s41586-021-04292-7)
- Deposit: Figshare collection, DOI 10.6084/m9.figshare.c.5471706 — [Figshare](https://doi.org/10.6084/m9.figshare.c.5471706)
- Tier evidence (ED Fig. 5, sample): the included `Nuclear_flipping.m` says the "data was aquired at 100kHz sampling rate and each current trace represents 160 electron spin readouts", and "the 81-120th records shoud all show an electron tunneling event as the electron is ... prepared in spin up state, while 121-160th records should mostly show no tunneling events, as electron is not excited and stays in the spin down state". Each `.bin` is 66,560 bytes of `uint16`, which is 33,280 samples, or 208 per readout. Scale and offset are in the script — [Figshare file](https://ndownloader.figshare.com/files/26691152); [article](https://api.figshare.com/v2/articles/14160887)
- Tier evidence (ED Fig. 3, sample): `#035_ESR_Rabi_UU/traces/ESR_0_0_1_0.hdf5` has `traces/output` with nominal shape (10, 4, 101, 50, 352) float64, root attributes `sample_rate` 200000.0 and `samples` 50, and the full pulse sequence stored as HDF5 attributes. The sweep axis is the π-pulse duration. The acquired read pulse lasts 1.7 ms. The file is only 17.3 MB, far below the 568 MB that shape implies, so most of the nominal array is unwritten and the stored shot count is not verified — [Figshare file](https://ndownloader.figshare.com/files/28627419); [article](https://api.figshare.com/v2/articles/14871087)
- Also present (listing): `CNOT_Zero-CNOT.zip` has about 2,560 raw `.bin` files plus a script describing "120 electron spin readouts" at 100 kHz; the GHZ and GST archives have `traces/*.hdf5` — [ED Fig. 6](https://api.figshare.com/v2/articles/14160893); [Fig. 4](https://api.figshare.com/v2/articles/14898981)
- Platform and lab: ³¹P donors in silicon, read by spin-dependent tunnelling to a SET. Morello group, UNSW (lab name from memory; authors not re-read). Three qubits: two nuclei plus one electron.
- Shots: ED Fig. 5 has 601 files × 160 = 96,160 readout records, of which 48,080 carry a derivable electron prepared state. Totals across the collection: not verified.
- Samples and duration: 208 samples at 100 kHz (2.08 ms) in ED Fig. 5; 352 samples at 200 kHz (1.76 ms) in ED Fig. 3.
- Labels: position within the file (`.bin` sets); sweep parameter plus stored pulse sequence (HDF5 sets).
- Format, size, licence, date: `.bin`, HDF5, QCoDeS `.dat`, MATLAB scripts; 14 articles totalling 8.96 GB; CC BY 4.0; 2021-07-06 — [Figshare API](https://api.figshare.com/v2/collections/5471706/articles)

**Struck et al. 2021: T2 (closest to T1 of the rest; the Rabi subset alone arguably meets T1)**
- Paper: "Robust and fast post-processing of single-shot spin qubit detection events with a neural network", Sci. Rep. (2021). Its statement says data are "available from the corresponding author on reasonable request" — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8355192/fullTextXML)
- Deposit: DOI 10.5281/zenodo.7232127, linked `isSupplementTo` the paper — [Zenodo](https://zenodo.org/records/7232127)
- Tier evidence (header): `train_data_set_1..4.npy` are each `<f8`, (250000, 491); `train_label_set_1.npy` is `<f8`, (1000000,) with 0/1 values — [file](https://zenodo.org/api/records/7232127/files/train_data_set_1.npy/content). The deposited script loads these under "train a model on real data" and sets `ms_to_samples=62.5` — [script](https://zenodo.org/api/records/7232127/files/code%20examples%20script.py/content)
- Why not T1: the paper says "The labels |↑⟩ and |↓⟩ for these data are assigned by the Bayesian filter", so they are classifier outcomes rather than prepared states. `rabi_oszillation_data.plk` holds real Rabi traces: the script reshapes 25,000 predictions into 100 drive times × 250 repeats, so drive time is derivable from order — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8355192/fullTextXML); [script](https://zenodo.org/api/records/7232127/files/code%20examples%20script.py/content)
- Platform and lab: single electron in a ²⁸Si/SiGe dot, SET current, 668 mT; authors Struck, Lindner, Hollmann, Schauer, Schmidbauer, Bougeard, Schreiber. One qubit. Affiliations: not verified — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8355192/fullTextXML)
- Shots and duration: 10⁶ traces "collected over two months of continuously running experiments with one device", plus 25,000 Rabi traces. The paper's synthetic traces use 495 points over 8 ms. The real training arrays have 491 samples and the Rabi arrays 495, at 62.5 samples per ms per the script — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8355192/fullTextXML)
- Format, size, licence, date: `.npy`, Python pickles (`.plk`, `.pkl`), Keras `.h5`; 4.05 GB; CC-BY-4.0; 2022-10-20. The Rabi set is a pickle (100.3 MB); I inspected bytes only, and first bytes show arrays of shape (495,) — [Zenodo API](https://zenodo.org/api/records/7232127)

**Zwerver et al.: T2**
- Paper: "Qubits made by advanced semiconductor manufacturing" (arXiv:2101.12650), whose statement cites this deposit — [arXiv](https://arxiv.org/html/2101.12650)
- Deposit: DOI 10.5281/zenodo.4478855. The record title is only "Data and analysis", so keyword searches miss it — [Zenodo](https://zenodo.org/records/4478855)
- Tier evidence (sample and member headers): QCoDeS `.dat` columns `Rabi_Frequency, N, t_set, digitizer_dig_1` with dimensions `# 41 150 80`; `Time_train_N_t_set.dat` `# 82 1000 80`; `...manip_mw_bp0_on_set_N_set_t_set.dat` `# 35 71 100 80`. Together that is 336,650 traces of 80 samples — [Zenodo](https://zenodo.org/records/4478855)
- Sample rate and duration: `Figure_3.py` sets `digitizer_sample_rate = 7629` and plots a ms axis, so 80 samples span about 10.5 ms — [Zenodo](https://zenodo.org/records/4478855)
- Labels: only sweep parameters. The scripts threshold traces, and `Figure_3.py` hand-picks one trace each as "spin up" and "spin down". Hence T2.
- Platform and lab: silicon dot with industrial patterning, charge-sensor spin-dependent tunnelling; three devices studied per the paper's text. Lab: not verified from the deposit.
- Format, size, licence, date: `.dat`, HDF5, Python; 302 MB zip (2.41 GB uncompressed); CC-BY-4.0; 2021-01-29 — [Zenodo API](https://zenodo.org/api/records/4478855)

**Zheng et al. 2019: T2**
- Paper: "Rapid gate-based spin read-out in silicon using an on-chip resonator", Nat. Nanotechnol. (2019) — [Nature](https://www.nature.com/articles/s41565-019-0488-9)
- Deposit: DOI 10.4121/uuid:8df1a6fa-9230-400f-a790-1b7714b1aad5 — [4TU](https://data.4tu.nl/articles/_/12707408/1)
- Tier evidence (header): `data/17-11-59_qtt_pulsescheme1D_timetrace/single-shot_traces.h5` holds `S21mag` and `S21phase`, each (10000, 768) float64. `plot_figures.m` uses a 1 μs time axis — [4TU API](https://data.4tu.nl/v2/articles/12707408/files). The paper says "We perform 10000 repetitions of this single-shot cycle, and record time traces of the transmitted signal with an integration time of 1 μs" — [arXiv](https://arxiv.org/html/1901.00687)
- Labels: none. The sequence loads "an electron with a random spin orientation", so no prepared-state label exists — [arXiv](https://arxiv.org/html/1901.00687)
- Platform and lab: double quantum dot, on-chip superconducting resonator, Pauli-blockade two-electron spin readout; Vandersypen group, QuTech per author list. One two-electron spin state. Duration about 768 μs including 50 μs before the read pulse.
- Format, size, licence, date: HDF5, QCoDeS `.dat`, MATLAB; 117.5 MB zip; CC0; 2019-05-28 — [Figshare API mirror](https://api.figshare.com/v2/articles/12707408)

**UNSW donor deposits on Dryad: T2 (shapes not verified)**
- Stemp et al., "Electron readout contrast enhancement in the parallel nuclear regime...", PRB 113, 245409 (2026): the README describes a `traces` subfolder with "raw SET traces" in HDF5, sampling 500 kS/s (2 μs), and `Blip_analysis_code.ipynb`. 561 MB zip, CC0 per the API, 2026-03-09 — [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.1c59zw4b2); [Crossref](https://api.crossref.org/works/10.1103/1jc4-v2zt)
- Stemp et al., "Scalable entanglement of nuclear spins mediated by electron exchange": `traces` folder "where present", 500 kS/s; 18.1 GB; 2025-07-21 — [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.brv15dvm5)
- Stemp et al., "Tomography of entangling two-qubit logic operations...": raw traces "for some datasets", HDF5; 27.1 GB; 2024-03-13 — [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.w3r2280zm)
- Yu et al., "Schrödinger cat states of a nuclear spin qudit in silicon": only one SET trace dataset is listed, for Fig. S14. Mostly processed; 138 MB; 2024-05-27 — [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.931zcrjtf)
- Vaartjes et al. ("Certifying the quantumness..."): the README does not mention traces; 260.6 MB; not classified — [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.547d7wmj0)
- Access: Dryad's API returned "Unauthorized, must have current bearer token", and direct file links return an Anubis bot check, so I could not open these archives — [Dryad API](https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.1c59zw4b2)

**Krzywda et al. 2026: T3 (despite the record's wording)**
- Deposit: "Coherence Protection for Mobile Spin Qubits in Silicon (Raw Data)". The record says files "contain single-shot readout traces" — [Zenodo](https://zenodo.org/records/18470200)
- Evidence (two samples): `_m0` (240, 700) and (130, 700) float64 in mV, with axes sweep × `repetition`. That is one value per repetition and no time axis in the files opened. About 189 of those HDF5 files exist.
- Platform and lab: ²⁸Si/SiGe linear dot device, QuTech, spin shuttling; PSB readout per the file names. Shots: 700 per sweep point. Format, size, licence, date: HDF5, 1.16 GB zip (4.54 GB uncompressed), CC-BY-4.0, 2026-02-10 — [Zenodo API](https://zenodo.org/api/records/18470200); [GitHub](https://github.com/jaq-lab/mobile-qubit-protection)

**Kelly et al.: T3**
- Paper: "Identification and mitigation of errors in hole spin qubit readout" (arXiv:2504.06898), IBM Research Europe – Zurich, germanium double dot, two qubits. Integration time 50 μs, "thresholding of 10,000 shots", OPX+ demodulation — [arXiv](https://arxiv.org/html/2504.06898)
- Evidence (npy headers only): `Fig_2e_ud.npz` holds `I` and `Q` as object arrays with shape (19, 51, 1000), which I read from the notebook as conversion steps × Rabi time × shots. The file names `uu`, `ud`, `du` give prepared configurations. The object dtype needs pickle to load, so I did not load the values.
- Format, size, licence, date: `.npz`, notebook; 287 MB zip (0.80 GB uncompressed); CC-BY-4.0; 2026-07-29 — [Zenodo](https://zenodo.org/records/21671989)

**Philips et al. 2022: T3**
- Paper: "Universal control of a six-qubit quantum processor in silicon", Nature (2022), six qubits — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9519456/fullTextXML)
- Evidence (sample): core-tools HDF5 with float32 datasets of 144,000 elements alongside 72- and 2,000-element arrays. I read this as 72 points × 2,000 shots of integrated values; no sample-level time axis was seen in the one file opened. 336 HDF5 files — [Zenodo](https://zenodo.org/records/6138474)
- Format, size, licence, date: HDF5 plus `.p` pickles; 660 MB zip; CC-BY-4.0; 2022-02-18.

**Takeda et al. 2024: T4**
- Paper: npj Quantum Information, published 2024-02-13. Fig. 3a: "The blue (red) trace shows the measurement result for the initial state T− (T+)", normalised Vrf — [Nature](https://www.nature.com/articles/s41534-024-00813-0)
- Evidence (sample): `Figure3a.csv` holds 2 series of 31,220 points, 3.2 μs to 4998 μs at 160 ns. They are smooth (std of first differences 0.0002), so I read them as averaged relaxation curves. The rest is figure data. 10 files, about 1 MB, CC-BY-4.0, 2024-01-09 — [Zenodo](https://zenodo.org/records/10473513)

**Eriksson group "Single-shot latched readout": T4**
- `figure_data.hdf5` (1.2 MB) has Fig. 2d as 30 points of time (ms) with |0⟩ and |1⟩ I_SD, plus histograms. CC-BY-4.0, 2024-08-23 — [Zenodo](https://zenodo.org/records/13368027)

**T6 (not obtainable or figure-level)**
- Matsumoto et al. 2021: "available from the authors upon reasonable request". The GitHub repository holds only `DNN_Classifier.py` and a one-line README — [Nature](https://www.nature.com/articles/s41534-021-00470-7); [GitHub](https://github.com/ymatsumoto65/dnn_spin_readout)
- Oakes et al. 2023: "available from the corresponding authors upon reasonable request" (arXiv v1; PRX version not checked) — [arXiv](https://arxiv.org/html/2203.06608v1)
- Mills et al. 2022: "All data needed ... are present in the paper and/or the Supplementary Materials" — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8986105/fullTextXML)
- West et al. 2019 and Urdampilleta et al. 2019: on reasonable request — [West](https://www.nature.com/articles/s41565-019-0400-7); [Urdampilleta](https://www.nature.com/articles/s41565-019-0443-9)
- Berritta et al. 2024: on request — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10891052/fullTextXML)
- Noiri et al. 2022: Zenodo holds 44 `fig_*.h5` figure files, 1 MB in total. Figure-level — [Zenodo](https://zenodo.org/records/5508362)
- Huang et al. 2024: three figure zips, 3.8 MB — [Zenodo](https://zenodo.org/records/10452860)
- Tanttu et al. 2024: GST HTML reports, analysis files, `raw_figure_data.zip` 0.4 MB — [Zenodo](https://zenodo.org/records/12537838)
- Steinacker et al. 2025: figure CSVs. Fig. 1b is a 401×401 map. `ExtData_Fig3.csv` (296 MB) is a lab-time series whose first row exceeds 15,000 columns; I saw no per-shot trace structure — [Zenodo](https://zenodo.org/records/17518472)

**Screened by listing only (not classified)**
- Xue et al. 2022: 181 MB of GST, VQE and fitting zips. The record states "The data must not be reused in other studies or publications without approval of the authors" — [Zenodo](https://zenodo.org/records/5044450)
- Xue et al. 2019 CRB: two HDF5 files totalling 1.84 GB uncompressed with a `Data Arrays` group; shape not verified — [Zenodo](https://zenodo.org/records/2613798)
- Neyens et al. 2024: CSV and pickles by figure, 72 MB; pickles not opened — [Zenodo](https://zenodo.org/records/10601293)
- Hendrickx et al. 2021: 6.86 MB zip — [4TU](https://doi.org/10.4121/13663442)
- John et al. 10-qubit Ge: `code_repo.zip`, 1.72 GB, CC BY-SA 4.0 — [4TU](https://doi.org/10.4121/5ee5b0d3-e838-478e-990d-02c50b75eeab)
- Bichromatic Rabi Ge: 337 MB — [4TU](https://doi.org/10.4121/bb43fe1d-f503-49e8-9f17-ce7d734f015d)
- "SNR in Spin Qubit Gate-Based Readout": 88 MB, mixed measured and simulated — [4TU](https://doi.org/10.4121/ad8cbe46-abd8-4d16-9672-18737177e096)
- Jirovec et al. 2021: Labber HDF5 "for figures", 222 MB — [DataCite](https://api.datacite.org/dois/10.15479/at:ista:9323)
- "Triplet-assisted leakage ... quantum point contact": "processed numerical data" and a "raw validation subset", 633 MB; numerics, so likely T5, not opened — [Zenodo](https://zenodo.org/records/20050549)

### Inferences
- The draft's sentence is false for spin qubits if read as covering all platforms. Mądzik meets T1 and three other deposits hold about 1.37 million real per-shot traces.
- Availability statements are unreliable in both directions. Struck says "on request" yet has a full deposit. Krzywda's record says "traces" but the sampled arrays have one value per shot. Only array shapes settle the tier.
- Signal models differ from IQ readout: Struck, Zwerver and Mądzik record current or voltage telegraph blips at 7.6 to 200 kHz. Only Zheng has magnitude and phase (S21 at 1 MHz).
- Struck's 10⁶ labels inherit the Bayesian filter's errors, so they cannot serve as ground truth without that caveat.

### Gaps
- Dryad archive contents could not be opened, so trace shapes and shot counts for the Stemp deposits are not verified. Mądzik's Figshare files use the `traces/*.hdf5` layout the Dryad READMEs describe, which suggests the same format but does not prove it.
- `Hu et al.` (arXiv:2206.03650): the abstract page mentions no dataset; full text not checked — [arXiv](https://arxiv.org/abs/2206.03650).
- `VNCiriano-Tejel/Fidelity-of-qubit-readout` (1 MB, GitHub): not opened because of API rate limits.
- Other QuTech core-tools deposits probably look like Philips (T3), but I did not sample them all.
- `Dataset for 'Identifying Pauli spin blockade using deep learning'` (Zenodo 7948852, 310 MB): by title this concerns blockade identification, not per-shot readout. Not opened.

## Neutral atoms (listed separately): are there public per-shot fluorescence images used for state readout?

### Takeaway
Yes, at large scale. CaltechDATA (Manetsch et al., 171.7 GB of raw qCMOS frames) and Figshare+ (Kim et al., 424 GB of raw EMCCD stacks) are downloadable now, and two smaller deposits on the Max Planck Edmond repository add to them. These are images with no time axis, so they are T3-image, not traces. The Phuttitarn ML-readout data, by contrast, shows no sign of being public.

### Cited Findings

**Manetsch et al. 2025 (Caltech): T3-image**
- Paper: "A tweezer array with 6,100 highly coherent atomic qubits", Nature (2025), DOI 10.1038/s41586-025-09641-4; its statement cites CaltechDATA — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12589140/fullTextXML)
- Deposit: — [CaltechDATA](https://data.caltech.edu/records/4m9sp-yzr58)
- Evidence (header plus record text): "raw qCMOS camera images (in units of photons collected per pixel, with type float16)". `fig1-2_part02.npy` is `<f2`, (2000, 4, 1041, 1041), one of 8 such files. `fig3_repeated_imaging.npy` is `<f2`, (15, 1001, 1041, 1041) — [record API](https://data.caltech.edu/api/records/4m9sp-yzr58)
- Platform: caesium atoms, 11,998-site array, about 6,139 atoms loaded per iteration — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12589140/fullTextXML)
- Shots: 16,000 iterations × 4 images, plus 15 × 1,001 images. Exposure for the main set: not verified (20 ms appears for one Extended Data set).
- Labels: none per image. Occupancy comes from analysis; the first image is omitted for the 3-image procedure. `fig4b_ramsey_presence.npy` (431.9 MB) is processed per-site boolean presence.
- Size, licence, date: 171.7 GB, MIT per the record's rights field, 2025-09-08.

**Kim et al. 2024 (Jaewook Ahn group; KAIST affiliation from memory): T3-image**
- Paper: "Quantum computing dataset of maximum independent set problem on king lattice of over hundred Rydberg atoms", Sci. Data (2024), ⁸⁷Rb, up to 141 atoms, 45 experiments of 4,000 to 30,000 repetitions, 582,916 events, two EMCCD photos per repetition (iXon Ultra 888) — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10805775/fullTextXML)
- Deposits: Figshare+ 24503680 (275.8 GB), 24523198 (106.6 GB), 24540988 (SPAM and calibration, 42.1 GB) — [Exp 1-30](https://doi.org/10.25452/figshare.plus.24503680); [Exp 31-45](https://doi.org/10.25452/figshare.plus.24523198); [SPAM](https://doi.org/10.25452/figshare.plus.24540988). Per-site fluorescence and digitised arrays, 0.78 GB — [Figshare](https://doi.org/10.6084/m9.figshare.23828004)
- Evidence (sample): `SPAM2_X12.tif` is 541×301, 16-bit, PackBits, 18 frames. Record text: "a three-dimensional array... the third dimension represents the stack of images across experimental repetitions" — [Figshare API](https://api.figshare.com/v2/articles/24540988)
- Labels: digitised `floudigreshape` from thresholding, and SPAM sets labelled by |g⟩→|r⟩ condition. Exposure: not verified.
- Format, licence, date: TIFF, `.mat`; CC BY 4.0; 2023-11-20.

**Impertro et al. 2023 (MPQ): T3-image**
- Paper: Commun. Phys. 6, 166 (2023); the statement cites Edmond — [Nature](https://www.nature.com/articles/s42005-023-01287-w)
- Evidence (remote HDF5 headers): `images` (310, 1, 475, 490) int16; `snippets` (310, 1, 3025, 256, 256) and (310, 1, 441, 256, 256) int16. README: "experimental fluorescence image data". Caesium quantum gas microscope, for site occupation rather than qubit-state readout. No labels (unsupervised method) — [Edmond](https://doi.org/10.17617/3.RNICHV)
- Size, licence, date: 11.43 GB, CC0, 2023-06-29.

**"Realization of a cavity-coupled Rydberg array" (Edmond, 2026; lab not verified): T3-image**
- Evidence (listing): `imaging_fidelity_survival_images.zip` has 4,616 `.jxl` images in `loop-*_img/tweezers_N.jxl` folders (about 1,154 loops). The README defines `shot_id` for per-shot, per-site CSVs such as `sqrt_N_PZT_vs_GND.csv` (17.2 MB), but a search of it found no description of the image zip. Image dimensions were not decoded — [Edmond](https://doi.org/10.17617/3.60MFFH)
- Paper DOI 10.1103/jgdt-d61y, arXiv:2602.12152. Size, licence, date: 0.52 GB, CC0, 2026-10-14. Species: not verified.

**Per-shot but digitised, not raw images**
- Chiu et al. 2019, "String patterns in the doped Hubbard model": per-shot "atom matrices" (1: atom, 0: hole or doublon) as Python 2.7 pickles, 0.259 GB, CC0 — [Dataverse](https://doi.org/10.7910/DVN/1CSVBV)
- "Realization of 1D anyons with arbitrary statistical phase": "Binary atom matrices", 5,317 files, 11 MB, CC0 — [Dataverse](https://doi.org/10.7910/DVN/HF5ONJ)
- "Bad metallic transport in a cold atom Fermi-Hubbard system": one raw FITS frame (801×801×2, Andor Zyla) plus 478 `.mat` files not opened, 1.64 GB, CC0 — [Dataverse](https://doi.org/10.7910/DVN/1FUZ0S)

**T6**
- Phuttitarn et al., Phys. Rev. Applied 22, 024011 (2024): the accepted manuscript has no data statement or repository link (full-text search). Described data: 3×3 Cs array, 3,000 single-shot images per dataset, readout 10 to 100 ms, labels from a high-SNR path — [accepted manuscript](https://link.aps.org/accepted/10.1103/PhysRevApplied.22.024011); [arXiv](https://arxiv.org/html/2311.12217v2)
- Kent et al., Phys. Rev. Applied 25, 044048 (2026): reuses that data, 6,002 images per readout time at 442×62 px, and the arXiv text gives no data link. The published statement is not verified (APS blocked) — [arXiv](https://arxiv.org/html/2504.08170v1); [Crossref](https://api.crossref.org/works/10.1103/x3h2-nz8y)
- Mude et al. (arXiv:2510.25982): same source; the portion read shows no data statement — [arXiv](https://arxiv.org/html/2510.25982)
- Bluvstein et al. 2024 and Evered et al. 2023: on request — [Bluvstein](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10830422/fullTextXML); [Evered](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10567572/fullTextXML)
- Su et al. 2025 ("Fast single atom imaging"): on request — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11762994/fullTextXML)
- Ma et al. (metastable Yb erasure): figure tables only — [Dataverse](https://doi.org/10.7910/DVN/TJ6OIF)
- La Rooij et al. 2023: 39 KB of source data — [DataCite](https://api.datacite.org/dois/10.15129/90d99ae4-35cb-4bb7-9e0e-f9e3194538d9)

### Inferences
- For image-based readout benchmarks, Manetsch and Kim are the usable public sets. Neither gives prepared hyperfine-state labels for the raw frames; the task they support is atom presence or Rydberg-versus-ground detection, and Kim's SPAM sets are the closest to calibration data.
- The data behind the Phuttitarn line of ML work would need a request to the authors.

### Gaps
- "A strontium quantum-gas microscope" (10.34810/data1343): the CORA API failed with a TLS error, so not classified.
- Winklmann et al. "Realistic neutral atom image simulation" (would be T5): my title lookup returned nothing, so no deposit was verified.
- `coldkevin1991/QuantumAtomProcessor` (1 MB): not opened because of GitHub rate limits.
- Several Rydberg-array Zenodo records looked figure-level from listing alone (for example [17538671](https://zenodo.org/records/17538671) and [15203442](https://zenodo.org/records/15203442)); I did not open them.

## Colour centres and other optically read solid-state spins: is any per-shot time series public?

### Takeaway
Yes, but only at T2 or below. The strongest is Fischer et al. (NV in a microcavity), with 1,465 raw HydraHarp time-tag files, followed by Brevoord et al. (SnV event-level arrival data, optical probing rather than spin readout). Chen et al. (Er³⁺) give prepared-label per-shot counts at T3. NV and SiV single-shot readout papers otherwise offer data on request or figure data only.

### Cited Findings

**Fischer et al. (NV in open microcavity): T2**
- Paper: "Spin-Photon Correlations from a Purcell-enhanced Diamond Nitrogen-Vacancy Center Coupled to an Open Microcavity", arXiv:2506.20722, Nat. Commun. article s41467-025-66722-8 — [4TU API](https://data.4tu.nl/v2/articles/6f8031ae-dd61-4b4f-adf4-d20bc88ed9ac)
- Deposit: — [4TU](https://doi.org/10.4121/6f8031ae-dd61-4b4f-adf4-d20bc88ed9ac)
- Evidence (listing plus sample): 1,004 `general_readout/hydra_harp.hdf5` and 461 `dark_esr_spin_pump_resonant_readout/hydra_harp.hdf5`, up to 1.59 GB each. A 0.9 MB sample holds `x0` uint64, `y0` uint8, `y1` uint64 of length 186,222. `y0` takes values 0 and 64 with 93,111 each. Attributes: `n_readout_pulses` 600, `HH400_base_resolution` 1.0, `HH400_binning` 16, sync aperture 0 to 100 ns. I read `x0` as picosecond ticks and `y0` as sync versus detector channel, since sync-to-click gaps of about 40 ns match the JSON's 40.2 ns pulse centre.
- Why T2: the top-level README covers environment setup only, so the format is undocumented, and I saw no per-shot prepared-state labels. Experiment type gives spin pumping and microwave settings at run level.
- Platform: single NV, one spin plus photonic time-bin qubits. Shots: `repetitions_todo` 100000 in the sampled run's JSON. Readout: 600 pulses, with a fit window of 43 to 70 ns in that JSON. Total readout duration: not verified.
- Format, size, licence, date: HDF5, JSON, Python; 60.4 GB zip (74.3 GB uncompressed); CC BY 4.0; 2026-01-21.

**Brevoord et al. 2023 (SnV): T2**
- Deposit: "Heralded initialization of charge state and optical transition frequency of diamond tin-vacancy centers" — [4TU](https://doi.org/10.4121/39356576-6144-465a-869c-e3e9c35e2dc6)
- Evidence (header): `20230610 probe rabi (all_sync_t1_flatten, all_probe_per_sync_flatten).npy` is `<f8`, (2, 17481704); `all_probe_counts.npy` is `<i4`, (5000000,); `20230531_CR_charac_all_pulses.npy` is `<i4`, (225, 2, 10000). Two others are object arrays that I did not load. These records concern optical and charge-state probing, not spin-state readout. Labels and array meaning come only from notebook variable names — [4TU API](https://data.4tu.nl/v2/articles/39356576-6144-465a-869c-e3e9c35e2dc6/files)
- Size, licence, date: 0.63 GB, CC BY 4.0, 2023-11-21. Journal citation: not verified.

**Antoniadis et al. 2023 (InAs quantum dot spin, optical): T2 (very few shots)**
- Paper: Nat. Commun. (2023), "Cavity-enhanced single-shot readout of a quantum dot spin within 3 nanoseconds" — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10322905/fullTextXML)
- Evidence (sample): `datafig2a.mat` has `shot1..shot5` uint8 (1000) on `time` 0 to 99.9 at 0.1 steps (units not stated in the file); `datafig1c.mat` has `Z_reps` (100, 201); `datafig3b.mat` has one record of 10,001 points; `datafig3c.mat` has 3,050 flip times. CC-BY-4.0, 2023-07-05, 0.74 MB — [Zenodo](https://zenodo.org/records/7937818)

**Chen et al. (six Er³⁺ ions in a silicon photonic crystal cavity): T3**
- Evidence (sample): `Fig2D.mat` holds `Ion1_initToHigh`, `Ion1_initToLow`, `Ion2_initToHigh`, `Ion2_initToLow`, each 3,000 int32 values between -14 and 15. These are per-shot integrated values with prepared-state names. Other files are figure data. CC0, 1 MB, 2020-09-09 — [Dataverse](https://doi.org/10.7910/DVN/5XMCMU); [file](https://dataverse.harvard.edu/api/access/datafile/4062374)

**Degen et al. 2021 (P1 centres near NV): T4**
- Paper: 10.1038/s41467-021-23454-9. Evidence (sample): `Fig2a_TimeTrace.txt` has 23,136 rows of `Time (s), N` at about 0.53 s per point, so one series with many repetitions inside each bin. Listing also shows `.npz` figure files. 618 MB zip including a 611 MB environment tarball, CC BY 4.0, 2021-05-28 — [4TU](https://doi.org/10.4121/14376611)

**T6 (not obtainable or figure-level)**
- Irber et al. 2021 (NV all-optical): on reasonable request — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7822820/fullTextXML)
- Zhang et al. 2021 (NV spin-to-charge): "may be requested from the authors" — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7943573/fullTextXML)
- Qian et al. (ML NV readout; time-resolved traces used, 10⁹ repetitions): on reasonable request — [arXiv](https://arxiv.org/html/2102.08558v1)
- Kindem et al. 2020 (Yb): on request — [Nature](https://www.nature.com/articles/s41586-020-2160-9)
- Anderson et al. 2022 (SiC): all data in paper and supplement — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8809532/fullTextXML)
- Abobeih et al. 2022: JSON and notebook for plots, about 0.3 MB in total — [Zenodo](https://zenodo.org/records/6461872)
- Song et al. (T centre register): 0.07 MB source data, rest on request — [Zenodo](https://zenodo.org/records/17239007)

**T5 (theory or simulation)**
- "Cavity-assisted single-shot T-center spin readout": the record says "two theoretical protocols"; `.mat` figure data — [Dataverse](https://doi.org/10.7910/DVN/KCVKYF)
- `snv-tfln-cavity-interface`: "simulation data and validation testbench"; not opened — [Zenodo](https://zenodo.org/records/23507013)

**Not classified**
- Gritsch et al. 2025 (Er in silicon, "Optical single-shot readout of spin qubits in silicon"): the paper's statement says deposited in mediaTUM, and the DOI resolves, but the landing page returned "Access Denied" from Anubis. Contents not verified — [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11695859/fullTextXML); [DataCite](https://api.datacite.org/dois/10.14459/2024mp1760064)
- Screened by listing only, possibly holding event-level data: SnV remote entanglement with `dataset_tpqi_big.hdf5` (1.73 GB) — [4TU](https://doi.org/10.4121/91bea2d1-437e-4796-bd21-5a1753609e77); 7-qubit NV register, 1.13 GB, CC BY-NC-SA — [4TU](https://doi.org/10.4121/cd297506-13ea-49ee-8d23-907f3ba01339); SnV cavities with `multi_harp.hdf5` — [4TU](https://doi.org/10.4121/df7ca98d-f2f3-485c-a91c-12810555321f); SnV thesis repository, 0.47 GB — [4TU](https://doi.org/10.4121/ea06e1ba-61f6-4d78-a88a-eb78c71eb4b4); "Deep three-dimensional solid-state qubit arrays" (laser-written NV), 1.89 GB — [Warwick](https://api.datacite.org/dois/10.82444/warw.33794839.v1); T centre source, `fig1c_data.mat` 154 MB — [Dataverse](https://doi.org/10.7910/DVN/XCS15A).

### Inferences
- Photon time tags are the natural raw format here. The Delft 4TU deposits are the place to look, since they often include raw detector output. They need reverse engineering, because formats are lightly documented.
- No public SiV per-shot series turned up.

### Gaps
- Sukachev 2017, Nguyen 2019 and Stas 2022 (SiV): publisher pages not readable. Title and keyword searches found no deposit, so this is an absence of evidence only.
- Rosenthal et al. (SnV single-shot, arXiv:2403.13110) and Hesselmeier et al. (arXiv:2401.04470): my scan of the arXiv text found no data statement. Published versions not checked.
- Rovny et al. NV covariance data ([XMEIHR](https://doi.org/10.7910/DVN/XMEIHR), [23TSVE](https://doi.org/10.7910/DVN/23TSVE)): small `.mat` figure files, not opened.

## Majorana and parity-readout devices: is any per-shot time series public?

### Takeaway
Yes. Four deposits from Microsoft and Delft hold real complex time traces of parity readout, all T2. In the three long-trace sets parity is not prepared per shot, so states are inferred afterwards from telegraph switching; for the 2026 Delft qubit set, per-shot labels are not verified. Together they exceed 340 GB of compressed data, and the Delft Kitaev deposit alone has 297 GB of uncompressed raw databases.

### Cited Findings

**Aghaee et al. 2025 (Microsoft): T2**
- Paper: "Interferometric single-shot parity measurement in InAs-Al hybrid devices", Nature 638, 651–655 (2025) — [GitHub README](https://github.com/microsoft/azure-quantum-parity-readout)
- Deposit: DOI 10.5281/zenodo.14804380 (2025-02-04); updated version 10.5281/zenodo.21847809 (2026-08-08), which changes `tgp2_tuneup` — [v1](https://zenodo.org/records/14804380); [update](https://zenodo.org/records/21847809)
- Evidence (header via partial inflate): `raw_data/mpr_A1/mpr_A1_Vrf.h5` has `Vrf_1` (71, 26, 69, 14853) complex128 and `time` (14853). The product times 16 bytes equals the 30.27 GB file size. That gives 127,374 traces of 14,853 samples. Dimension order: `B_perp` is 71 and `V_qd_1_plunger_gate` is 26; the axis of size 69 is not verified — [Zenodo](https://zenodo.org/records/14804380)
- Duration and rate: "nearly 1.5×10⁴ consecutive measurements of the resonator response, each with an integration time of 4.5 μs, thereby recording a time trace of total length 67 ms" — [arXiv](https://arxiv.org/html/2401.09549v4)
- Documentation: "Some of the raw datasets ... contain timetraces that were left uncoarsened" — [GitHub README](https://github.com/microsoft/azure-quantum-parity-readout)
- Labels: none prepared. The repository provides GMM thresholding and dwell-time analysis — [code](https://github.com/microsoft/azure-quantum-parity-readout/blob/main/measured_analysis_figures/analysis_code/timetrace_analysis.py)
- Size and contents: 96.5 GB; raw HDF5 `mpr_A1`, `mpr_A2`, `mpr_B1`, `trivial_A`, `trivial_B`, `cut_loop_A`, `cut_loop_B`, `injector`; plus `converted_data` and `simulated` (1 GB).
- Licence conflict: Zenodo metadata says CC-BY-4.0, while the bundled manifest says CDLA-Permissive-2.0 — [Zenodo API](https://zenodo.org/api/records/14804380); [manifest](https://zenodo.org/api/records/14804380/files/azure-quantum-parity-readout.json/content)
- Context: an arXiv comment disputes the topological gap protocol behind this work, which concerns device interpretation and not the time traces — [arXiv](https://arxiv.org/html/2503.08944v1)

**Microsoft 2025 tetron: T2**
- Paper: "Distinct Lifetimes for X and Z Loop Measurements in a Majorana Tetron Device", arXiv:2507.08795. "long time traces of 120 ms length" and "effective integration time of 2 μs" after coarsening — [arXiv](https://arxiv.org/html/2507.08795)
- Evidence (listing): `long_time_record_Vrf.h5` 1.21 GB, `xmpr_Vrf.h5` 0.77 GB, `zmpr_Vrf.h5` 68.5 MB, `wpwp_0..7.zip` about 10.9 GB each. Raw shape not verified: metadata lay beyond a 25 MB partial read — [Zenodo](https://zenodo.org/records/17058315)
- Size, licence, date: 92.9 GB, CC-BY-4.0, 2025-09-04. Code: [GitHub](https://github.com/microsoft/microsoft-quantum-tetron-lifetimes)

**van Loo, Zatelli et al. 2026 (Delft): T2**
- Paper: "Single-shot parity readout of a minimal Kitaev chain", Nature 650, 334–339 (2026), published 2026-02-11. Its statement cites this deposit — [Nature](https://www.nature.com/articles/s41586-025-09927-7)
- Deposit: DOI 10.4121/227fd419-fded-4a96-ab62-421a0cd57fa5 — [4TU](https://doi.org/10.4121/227fd419-fded-4a96-ab62-421a0cd57fa5)
- Evidence (samples): QCoDeS SQLite `U289_04_longtraces_10.db` holds `resonatorM` complex64 (25000) with `time` step 0.2 ms, 5 s total, one blob per run. `analyzed_timetraces/*_hmm.nc` holds `assigned_time_series` and `avg_resonator_X` of 66,666 bins at 150 μs, about 10 s; its PSD extends to 10 kHz.
- Rates from the paper: captions cite 50 kHz and 10 kHz sampling and a 150 μs integration time for reported measurements. The README says time traces were collected in separate databases — [Nature](https://www.nature.com/articles/s41586-025-09927-7); [README](https://data.4tu.nl/file/227fd419-fded-4a96-ab62-421a0cd57fa5/83dd440d-60d2-43d0-9228-68dc3b553ad2)
- Labels: HMM-assigned states in the analysed set; no prepared state.
- Counts: about 310 time-trace databases, the largest 11.3 GB; 15,270 analysed files.
- Format, size, licence, date: SQLite, netCDF, notebooks; `raw_data.zip` 141 GB (297 GB uncompressed), `analyzed_timetraces.zip` 16.5 GB, 157.6 GB in total; CC BY 4.0; v3 2025-11-12.

**Zatelli, Roovers, van Loo et al. 2026 (Delft): T2**
- Paper: "Majorana parity qubit in coupled minimal Kitaev chains", arXiv:2607.09511 — [arXiv API](https://export.arxiv.org/api/query?search_query=ti:%22Majorana%20parity%20qubit%20in%20coupled%20minimal%20Kitaev%20chains%22&max_results=1)
- Evidence (sample): `ed_singlechain_left_timetrace.nc`, named `PulsedReadoutTimeTrace`, has `resonatorM` complex128 (12437) with `time` step 402 μs, 5 s total. Oscillation files hold processed probabilities such as `p_0_cond_0` with 4,141 points. Raw is 33 QCoDeS databases. Figures distinguish even and odd initialisation, but I did not verify per-shot labels in raw data — [Zenodo](https://zenodo.org/records/23189687)
- Size, licence, date: 4.9 GB (raw 20.5 GB uncompressed), CC-BY-4.0, 2026-10-06.

**T6 and T5**
- Hinderling et al. 2024 (flip-chip inductive parity readout): the arXiv text says "will be available on Zenodo" and "upon reasonable request". Zenodo title search found nothing — [arXiv](https://arxiv.org/abs/2307.06718). T6.
- Theory or simulated: Benjadi and Egger [Zenodo](https://zenodo.org/records/20623730); Liu [Zenodo](https://zenodo.org/records/19450116); Stanescu [Figshare](https://doi.org/10.6084/m9.figshare.31739833); Miles et al. [Zenodo](https://zenodo.org/records/23321749). T5 at most, with no readout traces seen.

**Out of scope, flagged for the superconducting researchers**
- Wesdorp et al., nanowire junction in a superconducting LC resonator, 43 GB, not opened — [4TU](https://doi.org/10.4121/17876240)
- "Probing a minimal Kitaev chain with gate reflectometry", 11.7 GB, not opened — [Zenodo](https://zenodo.org/records/15848622)

### Inferences
- These are the closest non-superconducting analogues to complex IQ-like records, since `Vrf_1` and `resonatorM` are complex-valued. They differ in that they are continuous random telegraph records with no per-shot preparation, so readout error can only be inferred from models.
- Their "qubit count" is ill-defined: these are parity-measurement devices, and only the 2026 Delft set is presented as a qubit.

### Gaps
- Exact raw sample rate of Microsoft's digitised time traces beyond the 4.5 μs integration, and shape of the tetron raw files.
- Delft raw shot counts: only small databases were opened; the large ones were sized from the listing.

## What did the negative searches cover, and what does this mean for the draft's claim?

### Takeaway
The draft's sentence "no other raw-trace dataset is publicly available" is not true across qubit platforms; one T1 deposit and over a dozen T2 records contradict it, although several T2 entries rest on README text and not on arrays I opened. If it is meant only for superconducting IQ readout, it needs that scope stated, and this note cannot confirm that narrower claim. Keyword searches have poor recall here: the T1 deposit and Zwerver were found only through papers' data statements.

### Cited Findings
- Counterexamples: Mądzik (T1) — [Figshare](https://doi.org/10.6084/m9.figshare.c.5471706); Struck — [Zenodo](https://zenodo.org/records/7232127); Zwerver — [Zenodo](https://zenodo.org/records/4478855); Zheng — [4TU](https://data.4tu.nl/articles/_/12707408/1); Aghaee — [Zenodo](https://zenodo.org/records/14804380); van Loo — [4TU](https://doi.org/10.4121/227fd419-fded-4a96-ab62-421a0cd57fa5); Fischer — [4TU](https://doi.org/10.4121/6f8031ae-dd61-4b4f-adf4-d20bc88ed9ac).

**Zenodo API** (`type=dataset`) queries returning zero — [API](https://zenodo.org/api/records):
- Ions: `"trapped ion" AND readout AND ("photon arrival" OR PMT OR photomultiplier)`; `"ion" AND qubit AND "state detection" AND (fluorescence OR "time-resolved")`; `"photon arrival times" AND (ion OR qubit)`; `"trapped-ion" AND "machine learning" AND readout`; `"trapped ion" AND (EMCCD OR camera OR sCMOS) AND readout`; `"trapped ion" AND SNSPD AND (readout OR detection)`; `"ion trap" AND "state discrimination"`; `"mid-circuit" AND "trapped ion" AND measurement`; `"ion qubit" AND readout`; `"state detection" AND ion AND qubit`; `"photon counts" AND "trapped ion"`; `"ion trap" AND EMCCD`.
- Spins: `"spin" AND readout AND traces AND "neural network"`; `"RF-SET" OR "radio-frequency single electron transistor" OR "rf-SET"`; `"RF-SET"`; `"rf-SET"`; `"charge sensor" AND ("time trace" OR "time traces" OR "traces")`; `"charge sensor" AND qubit`; `"Elzerman"`; `"Elzerman" AND readout`; `"Pauli spin blockade" AND "single-shot"`; `"spin blockade" AND readout`; `"dispersive" AND spin AND "single-shot" AND (silicon OR germanium)`; `"germanium" AND "hole" AND ("single-shot" OR "single shot")`; `"donor" AND "single-shot" AND spin AND silicon`; `"donor" AND "spin qubit"`; `"single electron box" OR "single-electron box" AND spin`; `"singlet-triplet" AND "single-shot"`; `"spin readout" AND ("hidden Markov" OR "Bayesian" OR "machine learning")`; `"single-electron transistor" AND spin AND readout`; `"Fast high-fidelity single-shot readout"`.
- Atoms: `"neutral atom" AND readout`; `"neutral atom" AND readout AND images`; `"tweezer" AND fluorescence AND (images OR image) AND "single-shot"`; `tweezer AND "single-shot" AND images`; `"atom array" AND images AND dataset`; `"atom array" AND fluorescence`; `"quantum gas microscope" AND images`; `"optical lattice" AND "fluorescence images"`; `"Rydberg" AND "fluorescence images"`; `"Rydberg" AND snapshots`; `"neutral atom" AND "machine learning" AND (readout OR detection OR imaging)`; `"single-site" AND reconstruction AND atoms`; `"Enhanced measurement of neutral"`.
- Colour centres: `"nitrogen-vacancy" AND "single-shot"`; `"NV centre" AND readout`; `"silicon-vacancy" AND diamond AND readout`; `"spin-to-charge" AND diamond`; `"silicon carbide" AND "single-shot"`; `erbium AND "single-shot"`; `ytterbium AND "single-shot" AND readout AND crystal`.
- Parity: `"Kitaev chain" AND readout`; `"Cooper pair box" AND nanowire`; `"Flip-chip-based fast inductive parity readout"`; `"inductive parity readout"`.

**DataCite** (`query`) with zero or only irrelevant results — [API](https://api.datacite.org/dois):
- Zero: `photon arrival times ion qubit`; `neutral atom readout images`; `silicon-vacancy single-shot`; `"133Ba+" qubit`; `"time-tagged" photons qubit`; `sCMOS ion qubit`; `PMT photon counts ion qubit`; `"poor man" Majorana`.
- Only irrelevant dataset hits: `RF-SET traces`; `charge sensor time trace`; `spin readout neural network`; `"QSCOUT"` apart from the Sandia record above; `EMCCD ion qubit`; `ion chain camera readout`; `"trapped ion" fluorescence images`.
- Title checks found only arXiv records or nothing for Seif, Oakes, Matsumoto, Phuttitarn, Kent and Mude.

**Other services**
- Figshare search, zero results: `trapped ion qubit readout`; `photon arrival times ion qubit`; `spin qubit single-shot traces`; `RF-SET charge sensor time traces`; `neutral atom fluorescence images readout` — [API](https://api.figshare.com/v2/articles/search)
- Dryad, zero results: `trapped ion qubit readout`; `charge sensor quantum dot`; `neutral atom fluorescence imaging`; `nitrogen vacancy single-shot`; `Majorana parity`. `spin qubit` returned the UNSW set above — [API](https://datadryad.org/api/v2/search)
- Harvard Dataverse: `trapped ion qubit readout`, `photon arrival ion qubit` and `Majorana parity` gave large noisy totals with nothing relevant in the top 15 — [API](https://dataverse.harvard.edu/api/search)
- Hugging Face: `qubit readout`, `spin qubit`, `trapped ion`, `neutral atom`, `quantum dot readout` all returned 0; `qubit` gave unrelated sets and `jwt625/sc-qubit-dataset` (superconducting by name, not opened) — [API](https://huggingface.co/api/datasets)
- OSF SHARE: five readout queries returned 0, except one irrelevant tetron project — [API](https://share.osf.io/api/v3/index-card-search)
- GitHub repository search: `qubit readout dataset` and `ion trap readout neural` returned 0, and longer phrases returned 0. Simple terms found only code repositories already listed — [API](https://api.github.com/search/repositories)

### Inferences
- Zero-hit Zenodo queries do not show absence: Struck's record was found, but Zwerver's title ("Data and analysis") and Mądzik's Figshare collection were invisible to platform keyword searches. The ion negatives rest on the same weak search plus paper checks, so they are the least certain of the conclusions.
- Reusing these deposits needs care, independent of tier: Struck `.plk`, Kelly object arrays, Chiu `.pkl`, Neyens `.pickle` and Philips `.p` files all require unpickling, which should only be done in isolation. Xue 2022 carries an explicit reuse restriction despite its CC-BY-4.0 tag.

### Gaps
- Blocked or unavailable: mediaTUM (Anubis access denied), Dryad file downloads (Anubis) and API (token required), CORA (TLS error), APS article pages (401 or 404, accepted-manuscript route worked for Phuttitarn only), GitHub API tree calls (rate limit).
- 4TU's native search (`search_for`) returned 0 for all 14 queries, which suggests the parameter was wrong and does not indicate absence. I reached 4TU through DataCite and paper links instead.
- Kaggle, IEEE DataPort and Mendeley Data were not searched natively; the latter two only through DataCite.
- University repositories (Oxford ORA, UMD DRUM, ETH) surfaced theses through DataCite, but I did not search them directly.
