# Public qubit readout trace datasets: repository-first sweep

Sweep date: 2026-11-06. Tiers follow the assignment: T1 real hardware, per-shot, time-resolved, downloadable now, labels present or derivable. T2 same but labels missing or unclear, undocumented format, or very few shots. T3 per-shot, integrated only. T4 time axis, averaged only. T5 simulated. T6 not obtainable.

How arrays were checked: `.npy` headers and zip central directories were read with HTTP range requests, and HDF5 metadata through a range-backed file object. Small samples went to a scratch directory, were read with `allow_pickle=False` or h5py, and were deleted. Nothing was unpickled or executed. Pickle files (`.pkl`, `.plk`) were never opened.

## What do searches for terms such as "qubit readout traces", "single-shot readout", "IQ traces", "time traces", "heterodyne", "dispersive readout", "readout transients", "quantum trajectories", "measurement record", "state discrimination" and "discriminator dataset" return in each repository?

### Takeaway
Exact phrases return little, and the phrases closest to the need return almost nothing: "IQ traces", "readout transients", "discriminator dataset" and "qubit state discrimination" return zero or one record on Zenodo and DataCite. The useful hits came from broader terms (`transmon`, `single-shot`, `raw data`) followed by inspecting file listings, because deposits are usually titled "Data for [paper]" and rarely say "traces".

### Cited Findings

Hit counts for phrase and AND queries. Zenodo counts are from its [records API](https://zenodo.org/api/records), DataCite counts from its [DOI API](https://api.datacite.org/dois). "DataCite (datasets)" is filtered to resource type Dataset.

| Query | Zenodo | DataCite (datasets) | DataCite (all types) |
|---|---|---|---|
| `"qubit readout"` | 36 | 66 | not run |
| `"readout traces"` | 1 | 3 | not run |
| `"qubit readout traces"` | not run | 0 | not run |
| `"single-shot readout"` | 13 | 31 | not run |
| `"IQ traces"` | 1 | 0 | not run |
| `"I/Q traces"` | 1 | 2 | not run |
| `"dispersive readout"` | 15 | 13 | not run |
| `"readout transients"` | 0 | 0 | not run |
| `"discriminator dataset"` | 0 | 0 | not run |
| `"qubit state discrimination"` | 0 | 0 | not run |
| `"readout dataset"` | 0 | not run | 14 (none qubit trace data) |
| `qubit AND readout AND (trace OR traces)` | 24 | not run | 197 |
| `transmon AND (trace OR traces)` | 31 | not run | 158 |
| `qubit AND (timetrace OR timetraces OR "time trace" OR "time traces")` | 3 | not run | 45 |
| `"time traces" AND qubit` | not run | 16 | not run |
| `qubit AND heterodyne` | 7 | not run | 66 |
| `qubit AND "quantum jump"` | 4 | not run | 94 |
| `"quantum trajectories" AND qubit` | not run | 6 | not run |
| `"measurement record" AND qubit` | not run | 1 | not run |
| `"state discrimination" AND qubit` | not run | 5 | not run |
| `discriminator AND qubit AND readout` | not run | 4 | not run |
| `"raw shots"` | 7 | not run | 34 |
| `qubit AND "I/Q"` | 18 | not run | 83 |

Other repositories, using the same keyword list in plain-word form:

| Repository | Query results | What the hits were |
|---|---|---|
| [Figshare API](https://api.figshare.com/v2/articles/search) | `qubit readout` 92, `single-shot readout` 84, `IQ traces` 20, `time traces qubit` 12, `heterodyne qubit` 3, `dispersive readout` 85, `quantum trajectories qubit` 24, `measurement record qubit` 38, `state discrimination qubit` 5, `qubit discriminator` 13, `transmon readout` 4 (page cap 100) | Mostly journal supplements. Relevant: records 24147135, 30636059, 33997593, 34803777, plus mirrored 4TU records. |
| [Dryad API](https://datadryad.org/api/v2/search?q=qubit+readout) | `qubit readout` 2, `single-shot readout` 2, `spin qubit readout` 1, `superconducting qubit` 10, `neutral atom readout` 1, all others 0 | One relevant record, 1c59zw4b2. |
| [Harvard Dataverse API](https://dataverse.harvard.edu/api/search?q=qubit+readout&type=dataset) | `qubit readout` 77, `transmon readout` 37, `heterodyne qubit` 50, `superconducting qubit` 139, `spin qubit readout` 279 | Multi-word queries behave as OR, so totals are inflated (`state discrimination qubit` gave 73,798). Relevant: DPMID3, 5XMCMU, RGTLP7, TJ6OIF, L9YIYC. |
| [4TU.ResearchData API](https://data.4tu.nl/v2/articles/search) | Returned 0 or the cap of 100 per query | The native search is unreliable. DataCite and title filtering found the relevant records, including several titled "Raw timetraces". |
| [Mendeley Data via DataCite prefix 10.17632](https://api.datacite.org/dois?prefix=10.17632&query=qubit) | `qubit` 74, `transmon` 3, `qubit AND readout` 0 | One relevant: vzs49gddsm. |
| [ETH via DataCite prefix 10.3929](https://api.datacite.org/dois?prefix=10.3929&query=qubit) | `qubit` 180, `transmon` 31, `qubit AND readout` 27, `"single-shot" AND (qubit OR spin)` 13 | Mostly papers and theses. One relevant data collection, ethz-b-000551461. |
| [GitHub repository search](https://api.github.com/search/repositories?q=qubit+readout) | `qubit readout` 78, `single-shot readout` 10, `dispersive readout` 11, `transmon readout` 17, `qubit discriminator` 3, `qubit state discrimination` 5, `readout transients` 4, `qubit readout dataset` 0, `readout traces qubit` 0, `IQ traces qubit` 0 | Mostly code and simulators. Real data found in a few repositories, listed below. |

Search-quality limits that affect these numbers:
- Zenodo treats unquoted words as OR, for example `qubit readout traces` returned 30,471 and `qubit readout machine learning` 341,896. I reviewed only the top 100 by relevance for the first wave and the top 50 for the second, 1,576 unique records in the first wave. ([Zenodo API](https://zenodo.org/api/records?q=qubit+readout+traces))
- The OSF search index returned HTTP 500 when filtered by resource type and worked without the filter. ([SHARE index](https://share.osf.io/trove/index-card-search?cardSearchText=qubit+readout&acceptMediatype=application/json))

### Inferences
- Keyword search alone would have missed the best deposits. The Harvard record DPMID3 says only that it holds "data and processing code" for a paper on engineering a bath for transmons, and its per-shot arrays are visible only from the file contents.
- Record descriptions are not reliable in either direction, which is why tiers below rest on array shapes where possible. The Krzywda record says "single-shot readout traces" but its files have no time axis (T3).

### Gaps
- GitHub code search needs authentication and `gh` is not installed here, so I could not run the requested code search for `.h5`, `.hdf5`, `.npz`, `.npy` and `.mat` files alongside readout names. Repository search only was used.
- Four GitHub second-wave queries were rate-limited with HTTP 403: `qubit readout autoencoder`, `qubit measurement classification lstm`, `NV center single shot readout data`, `ibm_brisbane meas_level 1 discriminator`.

## Among the hits, which ones actually contain per-shot time-resolved arrays, as shown by file listings and documentation?

### Takeaway
Three deposits are T1, all single-qubit superconducting: the Fermilab/Purdue QICK set (1,000,000 shots), the Aalto bolometer set (2,000 shots), and the Hatridge-lab three-level set (310 GB). Eight more are T2: four confirmed from array shapes (Struck, Zheng, Watson, Microsoft) and four from documentation or file names only (Stemp, Zatelli, Sett, Severin). I found no public re-host of the Lienhard dataset and no public multi-qubit per-shot time-resolved superconducting dataset.

### Cited Findings

#### T1

**1. QICK / hls4ml transmon readout, Zenodo [10.5281/zenodo.14427490](https://zenodo.org/records/14427490)**
- Citation: Di Guglielmo, Du et al., "End-to-end workflow for machine learning-based qubit readout with QICK and hls4ml", [arXiv:2501.14663](https://arxiv.org/pdf/2501.14663). A later code archive cites the journal version as IEEE Trans. Quantum Eng. 2025, doi:10.1109/TQE.2025.3604712. ([Zenodo 23777889 README](https://zenodo.org/records/23777889))
- Tier evidence: `.npy` headers read directly from the [record files](https://zenodo.org/api/records/14427490): `0528_X_train_0_770.npy` (900000, 1540) float64, `0528_X_test_0_770.npy` (100000, 1540), `0528_y_train_0_770.npy` (900000,), `0528_y_test_0_770.npy` (100000,). The [paper](https://arxiv.org/pdf/2501.14663) says the signal is "digitized by the ADC to a discrete-time series of quadrature values {I_n, Q_n}", with n = 1 to 770. A sample of 500 rows I read showed integer values with a per-sample standard deviation near 825 ADC units, matching the single-shot scale in the paper's Fig. 3a.
- Platform and lab: one transmon on a frequency-tunable multi-transmon device, read with QICK on a ZCU216 RFSoC. Authors from Fermilab, Purdue and the University of Chicago. ([paper](https://arxiv.org/pdf/2501.14663))
- Qubits: 1. Shots: 1,000,000 (the paper states 500,000 per prepared state). ([paper](https://arxiv.org/pdf/2501.14663))
- Samples and rate: 770 I/Q pairs per shot, stored as 1,540 values. The ADC runs at 6.88 GS/s and is demodulated on the FPGA; 770 clock cycles correspond to 2.5 µs, about 3.25 ns per sample. ([paper](https://arxiv.org/pdf/2501.14663)) The interleaved I,Q layout is taken from a [third-party config](https://zenodo.org/records/23777889) (`"layout": "interleaved_IQ", "n_time": 770`), and my sample agrees: the two ends of the test file, whose checked labels are 0 and 1, differ in both the even and the odd columns. I checked the first 40 and last 5 test labels, not every label in those 500 rows.
- Readout duration: 2.5 µs. Labels: prepared |0⟩ or |1⟩, with cooling before preparation and a π pulse for |1⟩. ([paper](https://arxiv.org/pdf/2501.14663))
- Format and size: NumPy `.npy`, 12.33 GB. Licence: CC-BY-4.0. Published: 2024-12-12. ([record](https://zenodo.org/records/14427490))
- Independent reuse: the [LUNA paper](https://arxiv.org/html/2512.07808) calls it "the publicly available superconducting-qubit readout dataset (time-series I/Q traces, labeled |0⟩/|1⟩)", and its [setup instructions](https://raw.githubusercontent.com/advent-lab/LUNA/HEAD/logicnets/experiments/binary_classification/README.md) download it.

**2. Thermal-detector readout, Zenodo [10.5281/zenodo.7773981](https://zenodo.org/records/7773981), later version [10.5281/zenodo.10000511](https://zenodo.org/records/10000511)**
- Citation: Gunyhó et al., "Single-Shot Readout of a Superconducting Qubit Using a Thermal Detector", arXiv:2303.03668, doi:10.1038/s41928-024-01147-7. ([record metadata](https://zenodo.org/records/10000511))
- Tier evidence: I extracted `data/bolometer-characterization-timetrace-76.nc` (25 MB) and read it with h5py: `readout_meta_CH0_re` and `readout_meta_CH0_im` are (2, 1000, 781) float64 in volts, `qubit_pi_pulse_amplitude` is [0, 1], `readout_meta_shot_index` has 1000 entries, `readout_meta_time` has 781 entries. ([record](https://zenodo.org/records/7773981))
- Platform and lab: superconducting qubit read by a bolometer. Described as Aalto data by a [later user](https://zenodo.org/records/23777889); lab affiliation not verified in the record itself. Qubits: 1 per the title.
- Shots: 2,000 in this run, 1,000 per label. Samples and rate: 781 at 64 ns spacing, which works out to 15.625 MS/s. Duration: 49.92 µs. Labels: with or without a π pulse. (from the same file)
- Format and size: NetCDF4/HDF5 plus scripts. 873 MB zip (1.81 GB uncompressed); later version 923 MB. Licence: CC-BY-4.0. Published: 2023-03-27, later version 2023-10-13. ([record](https://zenodo.org/records/7773981))
- Caveat: only this one run is confirmed as T1, and 2,000 shots is small. The file `qubit-bolometer-readout-MC6-I06-126.nc` exposes a time axis of 1,465 points and sweep axes but no shot axis in the metadata I could read, so it looks T4. Larger files in the set were not opened.

**3. Three-level transmon readout, Harvard Dataverse [doi:10.7910/DVN/DPMID3](https://doi.org/10.7910/DVN/DPMID3)**
- Citation: Cao, Mucci, Liu, Pekker, Hatridge, "Engineering a multi-level bath for transmons with three-wave mixing and parametric drives", [arXiv:2407.21765](https://arxiv.org/abs/2407.21765v2); the search result gives PRL 135, 230403 (2025), which I did not check at the publisher.
- Tier evidence: remote h5py read of [`001_prep_e_natural_decay_0`](https://dataverse.harvard.edu/api/access/datafile/13103194): `I_data` (220000, 80), `Q_data` (220000, 80) float64, `x_data` (121,). The [analysis notebook](https://dataverse.harvard.edu/api/access/datafile/13099454) confirms the second axis is time within a readout: `# the readout signal integration window is from index 20 to index 80`, followed by `np.mean(raw_I_data[:, 20:80], axis=1)`. Its docstring calls this "the raw I and Q data" and says each record alternates "a first measurement for preparing state" and a second measurement after the pump.
- Platform and lab: transmon coupled to a SNAIL, Hatridge group per the paper's author list. Deposited by Xi Cao, University of Illinois Urbana-Champaign. ([record](https://doi.org/10.7910/DVN/DPMID3)) Qubits: 1.
- Shots: 220,000 records per file in the natural-decay groups (110,000 measurement pairs); 60 files each for `prep_g`, `prep_e`, `prep_f`. Other groups use other file sizes, and I did not compute a total. ([file list](https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DPMID3))
- Samples: 80 per readout. Sample rate and readout duration: not verified.
- Labels: prepared state g, e or f in the file name, plus a herald measurement in each record. The second measurement follows a variable wait or pump, so its label is reliable only at short delays. There is no README in the record; the notebook is the only documentation.
- Format and size: HDF5 without file extensions, 310 GB in 1,598 files, none restricted. Licence: CC0 1.0. Published: 2025-10-31. ([record](https://doi.org/10.7910/DVN/DPMID3))

#### T2

**4. Si/SiGe spin qubit, Zenodo [10.5281/zenodo.7232127](https://zenodo.org/records/7232127)**
- Citation: Struck et al., Sci. Rep., doi:10.1038/s41598-021-95562-x. ([record](https://zenodo.org/records/7232127))
- Why T2: real traces and per-shot time-resolved, but the labels come from a classifier, not from preparation. The [paper](https://ar5iv.labs.arxiv.org/html/2012.04686) says 10⁶ measured traces trained network C and that a Bayesian filter "was used for this classification".
- Evidence: `train_data_set_1..4.npy` are each (250000, 491) float64 and `train_label_set_1.npy` is (1000000,). The [script](https://zenodo.org/api/records/7232127/files/code%20examples%20script.py/content) labels this block "a model on real data" and sets 62.5 samples per ms. A sample of 50 rows shows each trace is normalised (median standard deviation 1.0), so these are not raw ADC values.
- Platform: single electron spin in a Si/SiGe quantum dot, charge-sensor readout. ([paper](https://ar5iv.labs.arxiv.org/html/2012.04686)) Lab: creators include Schreiber and Bougeard; affiliation not verified. Qubits: 1.
- Shots: 1,000,000. Samples: 491 at 16 µs (computed from 62.5 per ms). Duration: about 7.86 ms (computed). Size: 4.05 GB in `.npy`, `.h5`, pickle. Licence: CC-BY-4.0. Published: 2022-10-20. `rabi_oszillation_data.plk` (100 MB) is a pickle and was not opened. The paper's text says data are on request, but this deposit is open.

**5. Gate-based spin readout, 4TU [10.4121/uuid:8df1a6fa-9230-400f-a790-1b7714b1aad5](https://doi.org/10.4121/uuid:8df1a6fa-9230-400f-a790-1b7714b1aad5)**
- Citation: Zheng et al., "Rapid gate-based spin read-out in silicon using an on-chip resonator", Nature Nanotechnology 2019. ([README in zip](https://data.4tu.nl/v2/articles/d94c064e-2cee-4ca4-b2a1-2852fb21d174))
- Why T2: no state labels in the file. The analysis histograms all shots and fits a double Gaussian.
- Evidence: `data/17-11-59_qtt_pulsescheme1D_timetrace/single-shot_traces.h5` has `S21mag` and `S21phase`, both (10000, 768) float64. `plot_figures.m` has a section "T1 and histogram from single-shot traces" that sets a time axis of 1 µs per sample. ([record](https://data.4tu.nl/v2/articles/d94c064e-2cee-4ca4-b2a1-2852fb21d174))
- Platform and lab: spin qubits in silicon, read through an on-chip resonator; Vandersypen at TU Delft is the correspondence contact in the README. Qubit count: not verified. Shots: 10,000. Samples: 768 at 1 µs. Duration: 768 µs. Size: 117.5 MB zip. Licence: CC0. Published: 2019-05-28.

**6. Two-qubit silicon processor, Zenodo [10.5281/zenodo.1135014](https://zenodo.org/records/1135014)**
- Citation: Watson et al., "A programmable two-qubit quantum processor in silicon", doi:10.1038/nature25766. ([record](https://zenodo.org/records/1135014))
- Why T2: undocumented format. The file attribute says `MEAS_INFO: No description provided`, and I found no label mapping.
- Evidence: the record calls this "the raw single-shot data for the state tomography experiments". One sampled `digitizerdata.h5` has `ch0` and `ch1` at (100, 28272) float64, with `FREQ(Hz)` 61035 and segment time 0.463 s. Each row is therefore a long segment, presumably holding many shots; shot boundaries are not verified.
- Platform: silicon spin qubits. Lab: not verified in the record. Qubits: 2 per the title. Samples: 28,272 per segment at 61.035 kHz. Size: 500 HDF5 files, 22.6 GB uncompressed in a 5.48 GB zip. Licence: CC-BY-4.0. Published: 2018-02-08.

**7. Donor spin qubits, Dryad [10.5061/dryad.1c59zw4b2](https://doi.org/10.5061/dryad.1c59zw4b2)**
- Citation: Stemp et al., doi:10.1103/1jc4-v2zt per [DataCite](https://api.datacite.org/dois/10.5061/dryad.1c59zw4b2).
- Why T2, and from documentation only: the [README text](https://api.datacite.org/dois/10.5061/dryad.1c59zw4b2) says each folder has a `traces` subfolder with "the raw SET current traces" in HDF5, sampled at 500 kS/s (2 µs). Labels come from thresholds in `analysis.json`, not from stated preparation. The Dryad API returned 401 and the download route 403 to my requests, so shapes, shots and qubit count are not verified.
- Platform: exchange-coupled donors in silicon, Elzerman readout. Size: 561 MB zip. Licence: CC0. Published: 2026-03-09.

**8. InAs-Al parity readout, Zenodo [10.5281/zenodo.14804380](https://zenodo.org/records/14804380)** (newer copy [21847809](https://zenodo.org/records/21847809))
- Citation: Microsoft Azure Quantum, Nature 638, 651 (2025). ([repository](https://github.com/microsoft/azure-quantum-parity-readout))
- Why T2: time-resolved RF records with no prepared-state labels. The device is a parity readout demonstration, not a qubit in the usual sense; it is included because any platform counts.
- Evidence: `qpp_Vrf.h5` has `Vrf_1` (100001,) complex128 and `time` (100001,). `mpr_A1_Vrf.h5` is 30 GB. The [README](https://github.com/microsoft/azure-quantum-parity-readout) says some raw datasets "contain timetraces that were left uncoarsened". The full array shape of the large files is not verified.
- Size: 96.5 GB. Licence: CC-BY-4.0. Published: 2025-02-04.

**9. Minimal Kitaev chain, 4TU [10.4121/227fd419-fded-4a96-ab62-421a0cd57fa5](https://data.4tu.nl/datasets/227fd419-fded-4a96-ab62-421a0cd57fa5)**
- Why T2, from documentation only: the README describes "raw time traces" and `timetraces_analysis.ipynb`, with `raw_data.zip` at 141 GB and `analyzed_timetraces.zip` at 16.5 GB. There are no prepared-state labels; parity is the measured quantity. Shapes not verified.
- Lab: QuTech, Kouwenhoven. Format: `.nc`. Licence: CC-BY-4.0. Published: 2025-11-12 (v3).

**10. Qubit-induced cavity bistability, Zenodo [10.5281/zenodo.10518321](https://zenodo.org/records/10518321)**
- Why T2, from documentation and file names: the record says "the raw data of PBB single shot measurements at all kappas are included". `Figure3.zip` holds files such as `PBB_dec100_5PIQ_..._Channel_1_sweep_1_...mat` at about 1.1 GB each. This is photon-blockade breakdown, not qubit-state readout, so there are no state labels. Shapes not verified.
- Citation: Sett et al., "Emergent Macroscopic Bistability Induced by a Single Superconducting Qubit". Size: 13.4 GB. Licence: CC-BY-4.0. Published: 2024-01-16.

**11. Elzerman classification project, GitHub [JohannSeverin/Elzermann_readouts](https://github.com/JohannSeverin/Elzermann_readouts)**
- Why T2: undocumented. `small_example_implementation/traces_high.txt` is 55 MB with 220 values per row, and there is no README. Hardware versus simulated origin is not verified. `common_data/train_test_data.dat` is a pickle by its first bytes and was not opened.

#### T3 (per-shot, no time axis)

| Dataset | Evidence for tier | Platform and lab | Shots and labels | Size, licence, date |
|---|---|---|---|---|
| [GitHub Quasar-UniNA/FCM-based-QubitDiscriminator](https://github.com/Quasar-UniNA/FCM-based-QubitDiscriminator) | `data/ibm_brisbane/.../0s.npy` is (8192, 127) complex64 | ibm_brisbane, University of Naples group by organisation name (not verified) | 8,192 shots × 127 qubits per file; label in file name | 4.1 GB, 1,572 files; licence not verified; pushed 2026-02-04 |
| [Zenodo 22719098](https://zenodo.org/records/22719098) | `I1_final`, `Q1_final` (100000,) in one sampled `.npz` | Transmons, Murch among creators; 1 to 5 qubits | 100,000 shots in that file | 4.6 GB, CC-BY-4.0, 2026-09-12 |
| [Zenodo 15364358](https://zenodo.org/records/15364358) | Description: soft and hard measurements per qubit, one row per repetition; `hard_measurements` (1000, n) read | Rigetti Ankaa-2, arXiv:2410.05202 | Per round; no prepared-state labels | 1.23 GB, CC-BY-4.0, 2024-10-22 |
| [Zenodo 18470200](https://zenodo.org/records/18470200) | `_m0` (60, 700) with a `repetition` axis, no time axis, despite "single-shot readout traces" in the text | Silicon spin qubits, QuTech | 700 repeats × 60 sweep points in sampled file | 4.5 GB, CC-BY-4.0, 2026-02-10 |
| [Zenodo 21671989](https://zenodo.org/records/21671989) | `Fig_2e_ud.npz`: `I`, `Q`, `t` are object arrays (19, 51, 1000); notebook uses `t[:,0]` as a sweep axis | Hole spin qubits, arXiv:2504.06898 | 1,000 shots per point; prepared uu, ud, du in file names | 800 MB, CC-BY-4.0, 2026-07-29 |
| [Figshare 24147135](https://doi.org/10.6084/m9.figshare.24147135.v2) | Fig. 4 I and Q arrays are 1-D, (50326,) and (70455,) | Superconducting qubit in a cavity, Lledó et al., Nat. Commun. 14, 6313 | Separate ground and excited files | 114 MB, CC BY 4.0, 2023-09-25 |
| [Figshare 30636059](https://doi.org/10.6084/m9.figshare.30636059) | `Parity data with the Ramsey protocol.npy` is (12582912,) int8, apparently thresholded outcomes | Offset-charge-tunable transmon | 4 µs cadence per description | 96 MB, CC BY 4.0, 2025-11-17 |
| [Zenodo 14033026](https://zenodo.org/records/14033026) | `Fig_4a/IQblobs_*.mat` by file name; shapes not read | All-optical readout, arXiv:2310.16817 | With `_prep` variants | 85 MB, CC-BY-4.0, 2024-11-06 |
| [Zenodo 14584494](https://zenodo.org/records/14584494) | `Fig_1bd_IQblobs_paramp.txt` has columns Ig, Qg, Ie, Qe | Ta fluxonium | Labelled g and e columns | 2.4 MB, no licence in metadata, 2025-01-01 |
| [4TU 29e13811](https://doi.org/10.4121/29e13811-83f5-45e5-8242-5f0933e25c4a.v1) | Sampled `data.npy` is (50000, 12) complex128 | Andreev spin qubits | 50,000 × 12 in sampled file | 19.5 GB, CC BY 4.0, 2023-08-11 |
| [4TU 20219522](https://doi.org/10.4121/20219522) and [4TU 19103525](https://doi.org/10.4121/19103525) | `.npy` is (2, 1000000) and (2, 878906) complex128; first row is timestamps at 4.5 µs and 2.28 µs | Transmon circuits per the titles; lab not verified | About 10⁶ ordered points per file, no labels. Each point looks like one repeated measurement; if treated as a continuous record instead, this would be T2 | 1.4 GB and 111 GB uncompressed, CC BY 4.0, 2022 |
| [4TU 886d8394](https://data.4tu.nl/datasets/886d8394-adfe-4560-807c-c0fa007110df) | Description: "integrated voltages for both I and Q quadratures"; files not opened | 17-qubit transmon, DiCarlo group | Calibration points included | CC-BY-4.0, 2026 |
| [Zenodo 21908500](https://zenodo.org/records/21908500) | `state_arr` (200000,) bool, so thresholded and not analog | Superconducting qubit, arXiv:2603.11889 | 11,009 files; 6 µs cadence | 2.6 GB, CC-BY-4.0, 2026-08-12 |
| [Hugging Face 300-Qubits-Register-Dataset](https://huggingface.co/datasets/anonymousshare/300-Qubits-Register-Dataset) | README: 27,000 PNG fluorescence frames with per-site annotations, no time axis | Trapped ions, per JSON field `ion_count`; lab not stated | Bright and dark labels per site | 740 MB zip, MIT, 2026 |
| [GitHub qc-coder/SingleShotReadout](https://github.com/qc-coder/SingleShotReadout) | `.dat` columns Real, Real-pi, Imag, Imag-pi plus preselection | Superconducting, type not verified | With and without π pulse | 22 MB, licence not verified, 2019 |
| [Mendeley vzs49gddsm](https://data.mendeley.com/datasets/vzs49gddsm/2) | Description of `QuadratureDistribution.mat`; files not opened | Transmon wQED, New J. Phys. 27, 073001 | Ground, excited, superposition | CC-BY-4.0, 2026 |
| [GitHub GaifutdinovDamir/Qubits-readout](https://github.com/GaifutdinovDamir/Qubits-readout) | `ones3.txt` holds one complex value per line; no README | Hardware origin not verified | Label by file name | Size 32 MB; licence not verified |
| [GitHub ahmad-shafiei/N_Qubits_BinaryClassification](https://github.com/ahmad-shafiei/N_Qubits_BinaryClassification) | `dataset_X_v2.txt` has 8 values per row, consistent with 4 qubits of I and Q; no README | Hardware origin not verified | Folder names say `experimental_correlated` and `synthetic` | 370 MB; licence not verified |

Also likely T3 but not opened: [4TU 1092cb12](https://data.4tu.nl/datasets/1092cb12-9198-4d43-8500-401c78a5dc15) (fluxonium SSRO, 14.6 GB of HDF5) and [4TU 20202071](https://doi.org/10.4121/20202071) (99 GB of `.npz` from the same acquisition pipeline as 20219522).

#### T4 (time axis, averaged)
- [Zenodo 6344913](https://zenodo.org/records/6344913), Delaney et al., Nature 2022: `envelopeData*.dat` holds 8 comma-separated columns, and histograms appear only as processed files. This is tentative, from the listing and first lines. 4.2 GB, CC-BY-4.0, 2022-03-10.
- [Zenodo 7773981](https://zenodo.org/records/7773981) `qubit-bolometer-readout-MC6-I06-126.nc`: time axis, no shot axis in the readable metadata.
- [GitHub qc-coder/SingleShotReadout](https://github.com/qc-coder/SingleShotReadout) `111113_time_plot.dat`: V(t) with a standard-deviation column.

#### T5 (simulated)
- [Zenodo 23974607](https://zenodo.org/records/23974607): QuTiP trajectory simulations in `.pkl`, 434 MB, CC-BY-4.0, 2026-11-05.
- [Zenodo 14252485](https://zenodo.org/records/14252485): "Numerical dataset for quantum trajectory simulations of a dissipative qubit under continuous measurement and feedback", 9.9 GB, CC-BY-4.0, 2024-12-01.
- [Zenodo 20050550](https://zenodo.org/records/20050550): numerical reproduction package for singlet-triplet readout, 633 MB, CC-BY-4.0, 2026-05-06.
- [GitHub samuelnoger/qubit-readout-ML](https://github.com/samuelnoger/qubit-readout-ML): QuTiP `mcsolve` "synthetic readout data".
- [GitHub buildgreat-bot/QuantumReadoutSimulator](https://github.com/buildgreat-bot/QuantumReadoutSimulator): signal-chain simulator.

#### T6 (not obtainable)
- [GitHub satvikmaurya/HERQULES](https://github.com/satvikmaurya/HERQULES): README says "Feel free to shoot me an email for dataset access."
- [Figshare 33997593](https://doi.org/10.6084/m9.figshare.33997593.v1): the only file, `data.txt`, says "Working on collecting data files."
- [4TU 19103525](https://doi.org/10.4121/19103525): the README says a further "500+ GB" of time traces is "available upon request". The published part is listed under T3.
- Code with no trace data: [advent-lab/LUNA](https://github.com/advent-lab/LUNA) (points to Zenodo 14427490), [RunawayFancy/DNN_dispersive_readout](https://github.com/RunawayFancy/DNN_dispersive_readout) (its `Data/` holds only PDFs), [sandras13/ml-qubit-discrimination](https://github.com/sandras13/ml-qubit-discrimination), [silicon-quantum-computing/qubit_readout](https://github.com/silicon-quantum-computing/qubit_readout), [nhanvtran/qubit-readout](https://github.com/nhanvtran/qubit-readout), [quantumiracle/On_board_FNN_qubit_discrimination](https://github.com/quantumiracle/On_board_FNN_qubit_discrimination), [xiasitan/ReadoutWithRL](https://github.com/xiasitan/ReadoutWithRL), [fischermoseley/readout_classifier](https://github.com/fischermoseley/readout_classifier), [JUNGJEIHUANG/Trapped-Ion-Qubits-readout](https://github.com/JUNGJEIHUANG/Trapped-Ion-Qubits-readout) (code and checkpoints).

#### Opened and ruled out (no readout data in any tier)
- Binary outcomes only: [Google Zenodo 13273331](https://zenodo.org/records/13273331) (`measurements.b8`), [4TU 5fe71572](https://data.4tu.nl/datasets/5fe71572-e7a9-4c07-88dd-3dfda3ff99a2) ("stored as binary outcomes").
- Figure values only: [NIST mds2-2956](https://doi.org/10.18434/mds2-2956), [NIST mds2-3216](https://doi.org/10.18434/mds2-3216), [Harvard 5XMCMU](https://doi.org/10.7910/DVN/5XMCMU), [Harvard TJ6OIF](https://doi.org/10.7910/DVN/TJ6OIF), [Harvard L9YIYC](https://doi.org/10.7910/DVN/L9YIYC).
- Figure and simulation files, no per-shot data seen: [NIST mds2-4177](https://doi.org/10.18434/mds2-4177).
- Event timestamps only: [Figshare 34803777](https://doi.org/10.6084/m9.figshare.34803777.v1).
- Amplifier data behind a login, no qubit traces: [IEEE DataPort TMTT2024PulsedLNA](https://ieee-dataport.org/documents/tmtt2024pulsedlna).

#### Lienhard et al. re-host check
No public re-host found. On Zenodo, `Lienhard AND readout` returned 0. On DataCite it returned 16, all arXiv preprints or unrelated records. ([DataCite](https://api.datacite.org/dois?query=Lienhard+AND+readout)) The GitHub searches `Lienhard qubit readout` and `KLiNQ qubit readout` returned 0. ([GitHub](https://api.github.com/search/repositories?q=Lienhard+qubit+readout)) The HERQULES README still directs readers to email for access. ([README](https://github.com/satvikmaurya/HERQULES))

### Inferences
- The draft's statement that "no other raw-trace dataset is publicly available" is false as written. Three T1 deposits contradict it, and the QICK set alone has 1,000,000 labelled shots and is already described as public by another paper.
- A narrower claim may still hold. All three T1 deposits are single-qubit, and I found no public per-shot time-resolved set with simultaneous multi-qubit readout on superconducting hardware. The five-qubit set may be unique in that respect, though this rests on the absence of a hit in this sweep, not on proof.
- The Harvard set is the closest public match to leakage work, because it has g, e and f preparation with time-resolved per-shot I/Q. That is my judgement from the file names and notebook. Its usable labels need care because of the variable delays.
- These sets are not interchangeable benchmarks. The bolometer traces use 64 ns spacing and thermal detection, the QICK traces use about 3.25 ns spacing and dispersive readout, and the Harvard set has only 80 samples per readout.

### Gaps
- Could not list files because automated access was refused: [ETH ethz-b-000551461](https://doi.org/10.3929/ethz-b-000551461) ("Single-Shot Spin Readout in Graphene Quantum Dots", DataCite lists 367.7 GB with HDF5 and zip, plausibly T2), [Illinois IDB-1447103](https://doi.org/10.13012/b2idb-1447103_v1) (fluxonium readout-induced leakage, described as "raw and processed data"), [UQ 10.14264/ab6f55f](https://researchdata.edu.au/data-supporting-neural-state-classification/3369705) and [UQ 10.48610/9d3c2eb](https://doi.org/10.48610/9d3c2eb) (transmon neural-network readout, "data for Sample A"), and [TUM 1760064](https://doi.org/10.14459/2024mp1760064) (erbium optical single-shot readout). No tier is assigned to these. The UQ and Illinois records are worth checking by hand in a browser, since either could be T1.
- Large records not opened: [Zenodo 23891291](https://zenodo.org/records/23891291), [Zenodo 15480420](https://zenodo.org/records/15480420), [Zenodo 19426904](https://zenodo.org/records/19426904), [Harvard RGTLP7](https://doi.org/10.7910/DVN/RGTLP7), and the raw files of [4TU 17876240](https://doi.org/10.4121/17876240), where my metadata read failed.
- Small "figure source" deposits were judged from file listings only and may include a few example traces, for example [Zenodo 13368027](https://zenodo.org/records/13368027), [Zenodo 16616153](https://zenodo.org/records/16616153), [Zenodo 10473513](https://zenodo.org/records/10473513), [Zenodo 7937818](https://zenodo.org/records/7937818).
- Trapped-ion raw deposits from ARTIQ were not opened: [Zenodo 17626177](https://zenodo.org/records/17626177), [Zenodo 23940044](https://zenodo.org/records/23940044).
- The sample rate and readout duration for the Harvard set are not verified. I did not read the PRL or its supplement.
- GitHub licences marked "not verified" could not be read because the API was rate-limited.

## Which repositories returned nothing relevant? List the queries you ran for each.

### Takeaway
Hugging Face returned one marginal hit (trapped-ion images), while OSTI, DOE Data Explorer, OSF and Kaggle returned nothing relevant, and IEEE DataPort, Mendeley and ETH were close to empty for this purpose. The useful repositories were Zenodo, Harvard Dataverse, 4TU, Figshare and GitHub, with one Dryad record.

### Cited Findings

Common 18-term list (call it L18): `qubit readout`, `readout traces`, `single-shot readout`, `IQ traces`, `time traces qubit`, `heterodyne qubit`, `dispersive readout`, `readout transients`, `quantum trajectories qubit`, `measurement record qubit`, `state discrimination qubit`, `qubit discriminator`, `transmon readout`, `superconducting qubit`, `spin qubit readout`, `trapped ion readout`, `neutral atom readout`, `qubit machine learning readout`.

Common prefix list (call it P6): `qubit`, `transmon`, `qubit AND readout`, `"single-shot" AND (qubit OR spin)`, `"quantum trajectories"`, `heterodyne AND qubit`.

| Repository | Queries run | Result |
|---|---|---|
| [Hugging Face datasets](https://huggingface.co/api/datasets?search=qubit) | L18 (0 hits each), then `qubit` 16, `qubits` 2, `transmon` 0, `readout` 17, `quantum-readout` 0, `superconducting` 1, `iq-traces` 0, `quantum measurement` 0 | One marginal hit (300-qubit images, T3). Others ruled out: `jwt625/sc-qubit-dataset` is paper metadata, `QSBench-Readout-Demo` is synthetic circuits. |
| [OSTI](https://www.osti.gov/api/v1/records?q=%22qubit+readout%22) | L18 with `product_type=Dataset` (0 each). Without the filter: `"qubit readout"`, `"single-shot readout"`, `"dispersive readout"`, `"readout traces"`, `"IQ traces"`, `"quantum trajectories" qubit`, `"state discrimination" qubit`, `transmon readout dataset` | Nothing relevant. Unfiltered results were journal articles, conference papers, reports, theses and patents. |
| [DOE Data Explorer](https://www.osti.gov/dataexplorer/api/v1/records?q=qubit) | `qubit`, `qubit readout`, `transmon`, `superconducting qubit`, `single-shot readout` | Nothing relevant. `qubit` gave 9, the visible item being a qubit-lattice algorithm paper mirrored from Dataverse; the others gave 0. |
| [OSF](https://api.osf.io/v2/nodes/?filter[title,description]=qubit) and [SHARE](https://share.osf.io/trove/index-card-search?cardSearchText=qubit&acceptMediatype=application/json) | `qubit`, `qubit readout`, `transmon`, `superconducting qubit`, `single-shot readout`, `dispersive readout`; also P6 via [DataCite prefix 10.17605](https://api.datacite.org/dois?prefix=10.17605&query=qubit) | Nothing relevant. Preprints and preregistered cloud-hardware projects. |
| [Kaggle](https://www.kaggle.com/api/v1/datasets/list?search=qubit) | `qubit`, `qubit readout`, `transmon`, `quantum readout`, `superconducting qubit`, `iq traces`, `single-shot readout`, `quantum measurement`, `quantum computer noise`, `dispersive readout`, `trapped ion` | Nothing relevant. Simulated circuit logs and qubit-mapping data. |
| [IEEE DataPort via DataCite prefix 10.21227](https://api.datacite.org/dois?prefix=10.21227&query=qubit) | P6: `qubit` 17, `qubit AND readout` 1, others 0. The native search URL returned 404. | Only the amplifier dataset, ruled out above. |
| [Mendeley Data via DataCite prefix 10.17632](https://api.datacite.org/dois?prefix=10.17632&query=qubit) | P6, plus the [native API](https://data.mendeley.com/api/research-data/search?search=qubit+readout) with 14 terms: L18 subset and `time traces qubit`, `readout transients` | One relevant record through DataCite. The native API returned 10 unrelated records per query. |
| [ETH Research Collection via DataCite prefix 10.3929](https://api.datacite.org/dois?prefix=10.3929&query=qubit) | P6. The native API returned 403. | One candidate collection, listing blocked (see Gaps above). |
| Other prefixes on DataCite: Caltech 10.22002, Princeton 10.34770, KIT RADAR 10.35097, PURR 10.4231, UMD DRUM 10.13016, Yale 10.60600, Materials Cloud 10.24435, Science Data Bank 10.57760, DTU 10.11583, Chalmers SND 10.5878, Kaggle 10.34740, Hugging Face 10.57967 | P6 for each ([example](https://api.datacite.org/dois?prefix=10.13016&query=qubit)) | Nothing relevant by title. UMD returned 164, nearly all theses and papers. |

Repositories that returned relevant results, with queries:

| Repository | Queries run |
|---|---|
| Zenodo | Wave 1, 41 queries: `"qubit readout"`, `"readout traces"`, `qubit readout traces`, `"single-shot readout"`, `"IQ traces"`, `"I/Q traces"`, `"time traces" qubit`, `heterodyne qubit readout`, `"dispersive readout"`, `"readout transients"`, `"readout transient" qubit`, `"quantum trajectories" qubit`, `"quantum trajectory" superconducting`, `"measurement record" qubit`, `"state discrimination" qubit`, `discriminator qubit readout`, `"discriminator dataset"`, `qubit readout neural network`, `qubit readout machine learning`, `superconducting qubit readout dataset`, `transmon readout`, `transmon dispersive`, `"qubit state discrimination"`, `readout classification qubit`, `"multiplexed readout"`, `"spin qubit" readout`, `"charge sensor" "time traces"`, `"rf reflectometry" "single shot"`, `"spin readout" traces`, `"trapped ion" readout photon`, `"neutral atom" readout fluorescence`, `"photon arrival" qubit`, `"NV center" "single-shot"`, `"quantum jumps" superconducting`, `"weak measurement" transmon`, `leakage qubit readout`, `"parametric amplifier" qubit readout`, `qubit ADC traces`, `qubit demodulated`, `QICK readout`, `FPGA qubit readout discriminator`. Wave 2, 67 AND queries, also run on DataCite without a type filter; they include the rows in the first table plus `"raw data" AND transmon`, `RFSoC AND qubit`, `qubit AND digitizer`, `qubit AND "ADC"`, `"parity switching"`, `"quasiparticle" AND "single-shot"`, `fluxonium AND readout`, `Elzerman`, `"spin-to-charge"`, `"state detection" AND ion AND qubit`, `ion AND PMT AND readout`, `"time-tagged" AND qubit`, `Lienhard AND readout`, `KLiNQ`, `HERQULES AND qubit`. |
| DataCite | Wave 1, 40 dataset-only queries, the same vocabulary as Zenodo wave 1 in AND form. Wave 2 as above. |
| Figshare, Dryad, Harvard Dataverse, 4TU | L18 on each native API. `readout traces` on Dryad failed with a connection error. |
| GitHub repository search | Wave 1: `qubit readout`, `qubit readout dataset`, `readout traces qubit`, `single-shot readout`, `qubit discriminator`, `qubit state discrimination`, `IQ traces qubit`, `dispersive readout`, `readout transients`, `quantum trajectories superconducting`, `superconducting qubit readout neural network`, `qubit readout machine learning`, `transmon readout`, `qubit classification FPGA`, `readout discriminator`, `spin qubit readout`, `trapped ion readout neural network`, `neutral atom readout`, `qubit heterodyne`, `leakage readout qubit`. Wave 2, 29 queries, for example `qubit readout data`, `elzerman readout`, `hls4ml qubit`, `Lienhard qubit readout`, `meas_level 0 raw qiskit readout`, `leakage 3-state discriminator transmon`; most returned 0 because GitHub ANDs the words. |

### Inferences
- General-purpose ML data hubs (Hugging Face, Kaggle) are not where this data lives. Physics data sits in Zenodo, institutional repositories and Harvard Dataverse, attached to specific papers.
- A zero on OSTI, OSF or Kaggle is a fair negative for these keywords. A zero on 4TU's or Mendeley's native search is not, because those engines missed records that DataCite found.

### Gaps
- IEEE DataPort, Kaggle and Mendeley were checked through APIs or DataCite only, not through their web interfaces.
- ETH's native search was blocked, so its coverage rests on DataCite metadata, which often lacks file detail.
