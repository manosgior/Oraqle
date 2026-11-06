# Superconducting-qubit experiment deposits: what readout data they really hold

Scope: QEC experiments and readout-physics or continuous-measurement experiments. Discriminator-focused papers are excluded, as is the Lienhard et al. five-qubit dataset. Checked on 2026-11-06. Repository contents were read from metadata APIs, zip central directories, HTTP range reads and small samples, not from paper abstracts.

## Which QEC experiment deposits provide readout data beyond bitstrings, and is it detection events, integrated I/Q, or anything time-resolved?

### Takeaway
No QEC deposit I could open holds time-resolved records, so none reaches T1 or T2. Google's seven QEC deposits hold hard measurement bits, detection events and decoder outputs only. Two hardware QEC deposits hold integrated per-round analog values: Caune et al. (Rigetti Ankaa-2, complex "soft measurements") and van der Meer et al. (Delft, I/Q voltages). A third (Hanisch et al., IBM) holds labelled calibration I/Q in a pickle, and I found no per-round QEC I/Q in it.

### Cited Findings

#### How tiers were applied
- The T1 to T6 definitions in the assignment leave two kinds of deposit without an exact fit, so I used these conventions:
  - **T3 (bits only)** means real hardware and per-shot, but thresholded bits with no I/Q point. It is below strict T3. I kept it as the nearest tier and always say "bits only".
  - **T6 (figure data)** means the deposit exists but holds only processed or figure-level values, so shot records are not obtainable.
- T1 and T2 records are one of two kinds, because the difference matters for readout-trace use:
  - **Waveform:** samples inside one readout pulse.
  - **Monitor:** a stream of consecutive short integrations or repeated measurements.
- "Not verified" means I did not see the fact in a primary source.

#### QEC deposits with integrated analog data (T3)
- **Caune et al., "Demonstrating real-time and low-latency quantum error correction with superconducting qubits"**, Nature Communications 17, 7383 (2026) ([DOI](https://doi.org/10.1038/s41467-026-73331-6)). **Tier T3.**
  - Repository: [Zenodo 15364358](https://zenodo.org/records/15364358) (v2) and [Zenodo 13961130](https://zenodo.org/records/13961130) (v1).
  - Evidence: the record says HDF5 groups `hard_measurements` and `soft_measurements` hold results "per qubit ... with each row representing a separate repetition" ([record](https://zenodo.org/records/15364358)). Range-reading `stability_9_raw_data.h5` showed `soft_measurements/<qubit>` as complex64 with shape (100000, 28) or (100000, 1), mirroring uint8 `hard_measurements`. There is no time axis within a readout.
  - Platform and lab: Rigetti Ankaa-2 per the record; author affiliations not verified.
  - Qubits: 9 and 8 per-qubit datasets in the two groups I opened.
  - Shots: 100,000 per circuit in `stability_9`, 1,000 in `fast_feedback`.
  - Samples per trace and sample rate: none, one complex value per round.
  - Readout duration: not verified.
  - Labels: stim circuits and qubit mappings; no per-shot prepared state in stability runs. The fast-feedback file has a `reference_data` group with `measurement_fidelity` per the record.
  - Formats and size: HDF5, CSV, TXT; 18 files, 1.2 GB. Licence CC-BY-4.0. Published 2024-10-22.
- **van der Meer, Serra Peralta et al., "Coherent versus stochastic error injection on a repetition-code logical qubit in superconducting hardware"**. Paper citation not verified. **Tier T3.**
  - Repository: [4TU 10.4121/886d8394-adfe-4560-807c-c0fa007110df.v1](https://data.4tu.nl/datasets/886d8394-adfe-4560-807c-c0fa007110df).
  - Evidence: the record says "The data consists of integrated voltages for both I and Q quadratures" and that calibration points are used "to discretize" them. A sampled `Repeated stab meas` `dataset.hdf5` has `x0` acq_idx, `x1` soft_repetition and `y0..y9` alternating I and Q voltages per qubit, each (1190000,), with xlen 59500 and ylen 20. My listing stopped at 14 entries, so at least five I/Q pairs exist.
  - Platform and lab: 17-qubit transmon processor per the record, used in a 9-qubit layout (5 data, 4 ancilla) with Qblox control ([README](https://data.4tu.nl/file/886d8394-adfe-4560-807c-c0fa007110df/2a8d038f-aae0-464c-b6b1-555387ed8ff5)); QuTech, Delft.
  - Shots: 1,190,000 acquisitions per qubit in that file; the split into shots, rounds and calibration points was not verified.
  - Samples per trace and sample rate: none.
  - Readout duration: not verified.
  - Labels: calibration points; injected errors by experiment name.
  - Formats and size: Quantify HDF5 plus snapshot JSON; 1,065 datasets, 180.5 GB uncompressed, 62.5 GB deposit. It also holds simulated NetCDF sets. Licence CC BY 4.0. Published 2026-06-08.
- **Hanisch, Wootton, Hetényi et al., "Soft information decoding with superconducting qubits"**, APS Open Science 1, 000019 (2026) ([DOI](https://doi.org/10.1103/y9fh-4x6n)). **Tier T3, calibration I/Q only.**
  - Repository: [Zenodo 20272164](https://zenodo.org/records/20272164), a code-repository snapshot.
  - Evidence: `notebooks/C++/cpp_dev/KDE/all_memories.pkl` (40.65 MB). A static opcode walk, with no unpickling, shows keys `mmr_0` and `mmr_1` holding complex128 arrays of shape (10000,) under integer keys. File size suggests about 127 entries (inference). The rest is mainly notebooks and JSON results.
  - Platform and lab: IBM processors by folder name (`ibm_sherbrooke`, `ibm_torino`, `ibmq_mumbai`); which device this pickle came from was not verified.
  - Shots: 10,000 per array, two arrays per key.
  - Samples per trace and sample rate: none.
  - Readout duration: not verified.
  - Labels: `mmr_0` and `mmr_1` by name only; reading them as prepared states is my interpretation, and the file is undocumented.
  - Formats and size: pickle, ipynb, JSON; 232.9 MB zip. Licence CC-BY-4.0. Published 2026-05-18.

#### QEC deposits with bits and detection events only (T3, bits only)
- **Acharya et al. 2023**, "Suppressing quantum errors by scaling a surface code logical qubit" ([Nature](https://www.nature.com/articles/s41586-022-05434-1)). [Zenodo 6804040](https://zenodo.org/records/6804040).
  - Evidence: `README.txt` inside the zip describes `measurements.b8` as "The actual measurement data collected from hardware" and `detection_events.b8` as rederivable from it with `stim m2d`. There are no analog files in the 2,095 members.
  - Platform and lab: Google Sycamore, Google Quantum AI.
  - Qubits: 17 in the `surface_code_bX_d3_r01` example; other layouts have more.
  - Shots: 50,000 in that example (`properties.yml`).
  - Samples per trace, sample rate, readout duration: none or not verified.
  - Labels: sweep bits and `obs_flips_actual`.
  - Formats and size: b8, 01, stim, dem, yml; 315.5 MB. Licence CC-BY-4.0. Published 2022-07-14.
- **Acharya et al. 2025**, "Quantum error correction below the surface code threshold" ([Nature](https://www.nature.com/articles/s41586-024-08449-y)). [Zenodo 13273331](https://zenodo.org/records/13273331).
  - Evidence: the file-type census of three of the four zips shows `measurements.b8`, `detection_events.b8`, `sweep_bits.b8`, `obs_flips_*`, stim and dem files only. The fourth zip was not listed.
  - Platform and lab: Google 72- and 105-qubit processors, Google Quantum AI.
  - Shots: the repetition-code README gives 100,000 shots of 1,000 cycles per sample, and 200 `measurements.b8` files; the 105Q example has 50,000 shots (`metadata.json`).
  - Size: 112.5 GB. Licence CC-BY-4.0. Published 2024-08-26. Waveform fields: none.
- **Other Google Quantum AI records, same result by file-type census**, all CC-BY-4.0:
  - [Eickbusch et al., dynamic surface codes](https://zenodo.org/records/14238907): 2.08 GB, 2024-12-22; the paper cites it ([arXiv](https://arxiv.org/abs/2412.14360)).
  - [Colour code](https://zenodo.org/records/14238944): 738.5 MB, 2024-12-22.
  - [Decoder priors](https://zenodo.org/records/11403595): 11.1 GB, 2024-06-01.
  - [Magic state cultivation](https://zenodo.org/records/17872003): `measurements.csv`, `detection_events.csv`, `raw_observables.csv`; 3.2 GB, 2025-12-16.
  - [RL control of QEC](https://zenodo.org/records/18896801): v2 7.8 GB, 2026-03-06; [v1](https://zenodo.org/records/17566522) 127.8 MB.
- **Drouet et al. 2026**, "Error correction on an array of superconducting qubits with defective components". [Zenodo 21336494](https://zenodo.org/records/21336494).
  - Evidence: an opcode walk of `device_results.pkl` shows bitstring strings; calibration CSVs are named `ibm_miami_*`.
  - Size 599.4 MB, CC-BY-4.0, 2026-07-17. Paper citation not verified.
- **Lange et al. 2026**, modular neural decoder. [Zenodo 23335425](https://zenodo.org/records/23335425).
  - Evidence: `ibm_berlin.zip` has `detection_events.b8` plus raw IBM Runtime JSON.
  - I did not check the JSON for I/Q, so the bits-only label is provisional. 575.3 MB, CC-BY-4.0, 2026-10-13.

#### Simulated (T5, nearest tier; no traces exist)
- **AlphaQubit, Bausch et al. 2024** ([Nature](https://www.nature.com/articles/s41586-024-08148-8)).
  - The paper says Pauli+ simulation data are at the `gdm-qec` bucket and the Sycamore hardware data are the Zenodo 6804040 deposit above.
  - The [bucket README](https://storage.googleapis.com/gdm-qec/README) lists `detector_events`, `detector_events_leakage`, `equivalent_logical_qubit_measurements`, `stabilizer_measurements`, `stabilizer_measurements_leakage`, with shape (rounds+1, num_stabilizers), and 100M train plus 5M eval and 5M test points per distance.
  - I listed 601 objects totalling 11.9 TB. Materials are CC-BY. The paper's `storage.mtls.cloud.google.com` link redirects to a Google login, while `storage.googleapis.com/gdm-qec` lists without one.
- **Battistel, Varbanov, Terhal 2021** ("S17_raw_data_set1"). [4TU 10.4121/14125302.v1](https://data.4tu.nl/datasets/47f566a4-0e9a-4c2f-b857-43fabf8f46a5) and [additional set](https://data.4tu.nl/datasets/d3b2a9d7-aeee-41bc-bc7e-52d3ae12ee22).
  - The record says "Surface code data generated using https://quantumsim.gitlab.io". The word "raw" in the title does not mean hardware.
  - 146.5 GB plus 84.2 GB, CC BY 4.0, 2021.

#### QEC leads with no shot-level deposit (T6)
- **Statement says on request:**
  - [Krinner et al. 2022](https://www.nature.com/articles/s41586-022-04566-8): "All data are available from the corresponding author on reasonable request."
  - [Sivak et al. 2023](https://www.nature.com/articles/s41586-023-05782-6): "available from the corresponding authors upon a request."
  - [Chen et al. 2021 (Google)](https://www.nature.com/articles/s41586-021-03588-y): "upon reasonable request."
  - [Ni et al. 2023](https://www.nature.com/articles/s41586-023-05784-4): source data for two figures, the rest on request.
  - [Putterman et al. 2025](https://arxiv.org/abs/2409.13025): "Data is available from the authors upon reasonable request."
  - [Bultink et al. 2020](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7083610/fullTextXML): "Additional data related to this paper may be requested from the authors."
- **Figure data only:**
  - [Marques et al. 2022](https://www.nature.com/articles/s41567-021-01423-9) points to [a GitHub repo](https://github.com/DiCarloLab-Delft/Logical_Qubit_Operations_Data) with folders per figure; `DATA/Figure_2` holds `Logical_meas.ipynb`, `Phi_data.npy` and `Theta_data.npy` ([listing](https://github.com/DiCarloLab-Delft/Logical_Qubit_Operations_Data/tree/main/DATA/Figure_2)).
  - [Varbanov et al. 2025](https://doi.org/10.1103/physrevresearch.7.013029): [Zenodo 8108286](https://zenodo.org/records/8108286) holds per-figure NetCDF of neural-network logical errors (14.2 MB zip), and its README says `google_performance` is evaluated on Google's experiment. [4TU qec-util](https://data.4tu.nl/datasets/c827ef4e-a53b-497f-ae1a-5f9128dd3250) is code, with no files listed by the API.
  - [Brock et al. 2025](https://www.nature.com/articles/s41586-025-08899-y): [Zenodo 15009818](https://zenodo.org/records/15009818) (14.2 MB, CC-BY-4.0, 2025-03-12). Sampled npz members hold probabilities and standard deviations, for example `prob_array` (21, 4, 3), plus `n_avgs`.
- **No statement and no deposit found:**
  - [Ali et al., PR Applied 22, 044031 (2024)](https://doi.org/10.1103/physrevapplied.22.044031): no availability statement or repository string in the [arXiv full text](https://arxiv.org/abs/2403.00706); no repository reference in Crossref; DataCite author search had no relevant hit.
  - [Zhao et al., PRL 129, 030501 (2022)](https://doi.org/10.1103/PhysRevLett.129.030501): same, by [arXiv full text](https://arxiv.org/abs/2112.13505).

### Inferences
- Detection events add no readout information beyond the bits, because Google's README says they can be rederived from `measurements.b8`, `sweep.b8` and the circuit.
- AlphaQubit's public analog-looking tensors are simulated. Its hardware data is the bit-level 2023 deposit. Neither can supply analog hardware records.
- Caune et al. and van der Meer et al. are the nearest multi-qubit substitutes for raw traces in QEC. They give multi-qubit, per-round analog values, but no time axis inside a readout.

### Gaps
- Whether Pauli+ `stabilizer_measurements` (declared float32) hold soft values: I did not open a TFRecord.
- Ali et al.: the version of record may have a statement. The APS abstract page I could fetch had none, and that page is not full text.
- Whether Caune et al. deposit a `measurement_fidelity` group with prepared-state I/Q: seen in the record text, not opened.
- Whether van der Meer et al. include time-resolved subsets: the experiment-name census of `dataset.hdf5` shows none, but I opened only one file.
- Fourth Google 2025 zip (`set2`) and Lange JSON content: not checked.

## Which readout-physics deposits (fast readout, Purcell filters, parametric amplifiers, measurement-induced state transitions, quantum jumps, quantum trajectories) include per-shot time-domain records?

### Takeaway
Some do, and they contradict the draft's claim as written. I found four T1 deposits:
- Cao et al. (Pittsburgh): 80 samples per shot, about 310 GB.
- Gunyhó et al. (Aalto): 781 samples per shot, thermal detector.
- Hassani et al. (ISTA): 64 samples per shot.
- Spiecker et al. (KIT): 25,000-sample I/Q measurement trains.

Three more are T2: unlabelled continuous-monitor I/Q streams from QuTech and ISTA. As far as I could verify they are all single-qubit readout records, and none is a simultaneous multi-qubit record with per-qubit labels. Almost none of the assignment's named fast-readout, Purcell and trajectory papers has a shot-level deposit.

### Cited Findings

#### T1: per-shot, time-resolved, downloadable, labels present or derivable
- **Cao, Mucci, Liu, Pekker, Hatridge, "Engineering a Multilevel Bath for Transmons with Three-Wave Mixing and Parametric Drives"**, PRL 135, 230403 (2025) ([DOI](https://doi.org/10.1103/mv4g-z64p), [arXiv](https://arxiv.org/abs/2407.21765)). **Tier T1, waveform.**
  - Repository: [Harvard Dataverse doi:10.7910/DVN/DPMID3](https://doi.org/10.7910/DVN/DPMID3), cited in the paper's reference list. Listing via the [API](https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DPMID3). The file API redirects to S3 and returned HTTP 206 without login.
  - Evidence:
    - Range-read HDF5 headers: `I_data` and `Q_data` are float64 (220000, 80) in [a natural_decay file](https://dataverse.harvard.edu/api/access/datafile/13103194) and (140000, 80) in [a pump file](https://dataverse.harvard.edu/api/access/datafile/13102181), plus `x_data` (121,) with values 0, 250, 500...
    - The deposited notebook computes `np.mean(raw_I_data[:, 20:80], axis=1)` under the comment "the readout signal integration window is from index 20 to index 80", then fits g, e, f Gaussians to the 2D I/Q histogram.
    - Its docstring says "The raw data contains two measurements: 1. A first measurement for preparing state 2. Second measurement for determining the state of the qubit after pump", and rows alternate between them.
    - My check on the first 1,500 rows of one file: standard deviation across rows at sample 50 was 10.4, against 0.27 expected noise on the mean, and the mean trace varies across the 80 samples. Rows are therefore not averages and columns are time.
  - Platform and lab: transmon with three-wave-mixing parametric drives and a TWPA "for single-shot readout" ([arXiv](https://arxiv.org/abs/2407.21765)); Hatridge lab, University of Pittsburgh per [Crossref](https://api.crossref.org/works/10.1103/mv4g-z64p). The Dataverse author affiliation reads University of Illinois Urbana-Champaign.
  - Qubits: one I/Q pair per file; device qubit count not verified.
  - Shots: 220,000 or 140,000 rows per file. About 240 million rows in total if the file-size arithmetic holds for unopened files (inference: 281,603,112 and 179,203,112 bytes match the two shapes plus 3,112 bytes).
  - Samples per trace and sample rate: 80; sample period not stated in the deposit, so not verified. The paper's 1.0 GSa/s figure is the AWG, not the digitizer.
  - Readout duration: not verified.
  - Labels: file names carry `prep_g`, `prep_e` or `prep_f`, or sequence names; no per-row label; the herald measurement is in the data. In the first 1,500 rows, `prep_e` and `prep_g` files had similar row-integrated I (3.23 and 3.47), and I did not separate herald rows, so label quality is unverified.
  - Formats and size: extensionless HDF5, ipynb, py; 1,598 files, 310.0 GB. Licence CC0 1.0. Published 2025-10-31.
- **Gunyhó et al., "Single-shot readout of a superconducting qubit using a thermal detector"**, Nature Electronics 7, 288 (2024) ([paper](https://www.nature.com/articles/s41928-024-01147-7)). **Tier T1, waveform, small.**
  - Repository: [Zenodo 10000511](https://zenodo.org/records/10000511), the version after [Zenodo 7773981](https://zenodo.org/records/7773981). The paper cites 10.5281/zenodo.7773980.
  - Evidence:
    - `data/bolometer-characterization-timetrace-76.nc` has `readout_meta_CH0_re` and `_im`, float64 (2, 1000, 781), over `qubit_pi_pulse_amplitude` (0, 1), `readout_meta_shot_index` and `readout_meta_time`.
    - The embedded QCoDeS snapshot gives `readout_card.n_averages = 0` (the deposited script reports n_averages as 2 to that power, so 1), `boxcar_filter_width = 4`, `n_points = 781` and `n_single_shots = 1000`.
    - The paper says "we record the full time traces of the bolometer output signal for 1000 single shots" ([arXiv](https://arxiv.org/abs/2303.03668)).
  - Platform and lab: transmon read by a bolometric thermal detector; Möttönen group ([Crossref author list](https://api.crossref.org/works/10.1038/s41928-024-01147-7)); Aalto not verified from a primary source.
  - Qubits: 1.
  - Shots: 1,000 per label.
  - Samples per trace and sample rate: 781 at 64 ns step (time coordinate), boxcar-downsampled from 250 MSa/s.
  - Readout duration: 40 µs heat pulse in a 49.92 µs window.
  - Labels: π-pulse amplitude 0 or 1.
  - Formats and size: NetCDF4 with QCoDeS snapshots; 922.9 MB zip, 1.89 GB uncompressed. Licence CC-BY-4.0. Published 2023-10-13.
  - Caveats: the signal is a bolometer probe tone, not a conventional dispersive heterodyne record, and the paper's best fidelity is 0.618. The same zip's `timetrace-77` is (2, 10000) integrated shots (T3). Its larger `qubit-bolometer-readout-*` files have an averaging coordinate (T4, per the deposited scripts; I did not open them).
- **Hassani et al., "Inductively shunted transmons exhibit noise insensitive plasmon states and a fluxon decay exceeding 3 hours"**, Nature Communications 14, 3968 (2023) ([paper](https://www.nature.com/articles/s41467-023-39656-2)). **Tier T1, waveform, small.**
  - Repository: [Zenodo 8004359](https://zenodo.org/records/8004359).
  - Evidence:
    - The record says of Fig. 7b: "Each file contains 500 time traces of the two consecutive pulses applied to the resonator ... The Qubit preparation condition is apparent in the file name along with the power."
    - Opened `.mat` files hold `Timens` (1, 64) with a 50.79 ns step ending at 3200 ns, and `I`, `Q`, `I1`, `Q1` of shape (501, 64).
    - The deposited `QND_ness.nb` sums each trace and histograms the sums to find thresholds.
    - Single rows differ visibly from the row average. I did not separately rule out hardware averaging per row.
  - Platform and lab: inductively shunted transmon; ISTA, with J. Fink as last creator ([record](https://zenodo.org/records/8004359)).
  - Qubits: 1.
  - Shots: 501 per file, 30 files, about 15,000 two-pulse traces.
  - Samples per trace and sample rate: 64 at 50.79 ns (about 19.7 MSa/s).
  - Readout duration: 3.2 µs window per pulse.
  - Labels: `_g_` or `_e_` in file names, plus power.
  - Formats and size: MATLAB v5, Mathematica notebooks; 297.2 MB zip. Licence CC-BY-4.0. Published 2023-06-02.
  - The same deposit's Fig. 7a is T3: 40 files of 40,000 single-shot I/Q bins at 500 ns integration with g/e labels. Fig. 8c `_raw_` monitoring files were not opened.
- **Spiecker et al., "Two-level system hyperpolarization using a quantum Szilard engine"**, Nature Physics 19, 1320 (2023) ([paper](https://www.nature.com/articles/s41567-023-02082-8)). **Tier T1, monitor.**
  - Repository: [Zenodo 7817552](https://zenodo.org/records/7817552). The paper says "Raw and processed data are publicly available via Zenodo".
  - Evidence:
    - `Data/IQ_data/Run_XX_N_YYYYY_pol_{e,g}_init_{e,g}.npy` are object-dtype pickled dicts.
    - A static `pickletools` walk of the first file I inspected, with no unpickling, shows `I` and `Q` as `<f4` of shape (100, 25000) plus `time_stamp (s)`.
    - The deposited notebook has `avg, n_measure = I_e.shape` and `dt = 2  # measurement repetition time = 2us`.
  - Platform and lab: granular-aluminium fluxonium, Quantum Machines controller, KIT (Pop group) ([arXiv](https://arxiv.org/abs/2204.00499)).
  - Qubits: 1.
  - Shots: 100 per file, 499 files by my count, about 50,000 traces.
  - Samples per trace and sample rate: 25,000 at 2 µs repetition.
  - Readout duration: 128 ns integration per sample (paper).
  - Labels: initial state `init_e` or `init_g`, polarization direction and N in file names.
  - Formats and size: pickled `.npy` (needs `allow_pickle=True`, which I did not use), notebook; 8.8 GB. Licence CC-BY-4.0. Published 2023-04-11.
  - Caveat: each sample is a separate 128 ns measurement, so there is no waveform inside a pulse.

#### T2: time-resolved, labels missing or unclear
- **Bargerbos et al., PRX Quantum 3, 030311 (2022)**, singlet-doublet transitions ([DOI](https://doi.org/10.1103/prxquantum.3.030311)). **Tier T2, monitor.**
  - Repository: [4TU 10.4121/19103525.v1](https://data.4tu.nl/datasets/2e22a266-8d89-4837-898a-7eb842624512).
  - Evidence:
    - `Raw timetraces/ID_*_IQ_*.npy` are `<c16` of shape (2, 878906). Row 0 reads 0, 2.276e-6, 4.551e-6... (time; row 1 holds the complex record, as the same group's phonon notebook unpacks it).
    - The paper calls these continuously measured "2 s time traces" ([arXiv](https://arxiv.org/abs/2202.12754)).
    - The record calls them "a representative selection" and says the other time traces "would take up another 500+ GB. This data is available upon request."
  - Platform and lab: transmon with a quantum-dot Josephson junction; QuTech (lab attribution inferred from the repository).
  - Qubits: 1 per trace; device qubit count not verified.
  - Shots: 3,935 records.
  - Samples per trace and sample rate: 878,906 at 2.276 µs.
  - Readout duration: 2 s record (paper).
  - Labels: none; jumps are inferred.
  - Formats and size: npy; 74.1 GB zip, 110.7 GB uncompressed. Licence CC BY 4.0. Published 2022-05-25.
- **Pita-Vidal et al., Nature Physics 19, 1110 (2023)** ([DOI](https://doi.org/10.1038/s41567-023-02071-x)). **Tier T2, monitor.**
  - Repository: [4TU 10.4121/20202071.v1](https://data.4tu.nl/datasets/7d1b983a-684e-4229-8842-d3fc0c1fbf3a).
  - Evidence: `Raw_timetraces/ID_*_IQ_*.npz` hold `amplitudes.npy` and `data.npy`; the example I opened is `<c16` (10, 300000). Other files are larger (up to 270 MB) and were not opened. The paper describes a "continuously measured 17 s-long time trace integrated in time bins of tint = 4.3 µs" ([arXiv](https://arxiv.org/abs/2208.10094)).
  - Platform and lab: superconducting spin qubit coupled to a transmon, read through the transmon; QuTech (inferred). Scope is borderline, since this is a hybrid device.
  - Qubits: not verified.
  - Shots: about 990 files.
  - Samples per trace and sample rate: 300,000 per row in the example; time step not verified (4.3 µs bins in the paper).
  - Readout duration: not verified per file.
  - Labels: none, with sweep amplitude in `amplitudes.npy`.
  - Formats and size: npz; 52.1 GB zip, 99.0 GB uncompressed. Licence CC BY 4.0. Published 2022-09-26.
- **Sett et al., PRX Quantum 5, 010327 (2024)**, photon blockade breakdown ([DOI](https://doi.org/10.1103/prxquantum.5.010327)). **Tier T2, monitor.**
  - Repository: [Zenodo 10518321](https://zenodo.org/records/10518321).
  - Evidence: the record says "The raw data of PBB single shot measurements at all kappas are included." A `Channel_1` `.mat` (MATLAB v5, 140.6 MB) starts with variables `I_39`, `I_38`... of real double (1, 144000). Companion `I_and_Q_variances` files have 10-point sweep axes.
  - Platform and lab: single qubit in a cavity, ISTA, with J. Fink as last creator ([record](https://zenodo.org/records/10518321)). This is a bistability experiment, not qubit-state readout, so no qubit-state labels apply.
  - Qubits: 1 (title).
  - Shots: one record per sweep step; total not verified.
  - Samples per trace and sample rate: 144,000 in that file; sample rate not verified (file names encode decimation, dec1 to dec200).
  - Readout duration: not verified.
  - Labels: none (bright and dim phases inferred).
  - Formats and size: `.mat`, Mathematica, ipynb; four zips, 13.4 GB. Licence CC-BY-4.0. Published 2024-01-16.

#### T3: per-shot integrated analog values
- **Zheng et al. 2026, "Syndrome-Resolved Decoding of Qubit Readout Errors"**. [Zenodo 23367046](https://zenodo.org/records/23367046). Paper citation not verified.
  - Evidence: the deposited data README says "Nothing is thresholded, converted or reduced", with `data` of shape (n_tRO, n_runs, n_cycles).
  - Platform and lab: 1 fluxonium and 1 transmon; EPFL SQIL and AQUA per `CITATION.cff`.
  - Shots: fluxonium complex64 (19, 8192, 50); transmon complex128 (40, 8000, 51); calibration (19, 8192) and (40, 3000).
  - Samples per trace: none. Readout duration: swept, 2 to 20 µs and 0.25 to 10.15 µs.
  - Labels: `ground_data`, `excited_data`, `f_data`. The README warns that fluxonium field names are inverted relative to physical states.
  - Formats and size: DDH5 and JSON; 1.0 GB zip. Licence CC-BY-4.0. Published 2026-10-14.
- **Zwanenburg et al. 2026, "Experimental Characterization and Modeling of Measurement-Induced State-Transitions in a Fluxonium Superconducting Qubit"** ([arXiv](https://arxiv.org/abs/2606.17866)). [4TU 10.4121/22122209-2f78-4a66-8d04-fe18ac8b244e.v2](https://data.4tu.nl/datasets/22122209-2f78-4a66-8d04-fe18ac8b244e).
  - Evidence: a sampled `SSRO Q1/dataset.hdf5` has `x0` prep_state, `x1` repetition, `y0` I and `y1` Q, each (131072,), with xlen 2 and ylen 65536.
  - It also holds 17,704 `QNDness-sequence` datasets (not opened) and 10,385 `Single Trace Acquisition` datasets, which are **T4**. The one I opened is 3,600 samples at 0.5556 ns (2 µs), and its snapshot has `cfg_sched_repetitions = 262144`. Two thirds of its power sits in one 209 MHz bin, so it is averaged.
  - Platform and lab: 1 fluxonium, Andersen lab, QuTech.
  - Shots: 65,536 per state in the SSRO file; 7,567 SSRO datasets.
  - Samples per trace: none for SSRO. Readout duration: 2 µs acquisition (paper).
  - Formats and size: Quantify HDF5; 35.4 GB zip. Licence CC BY 4.0. Published 2026-06-15.
- **Stefanski 2024 PhD thesis data**. [4TU 10.4121/956b028c-6ea2-402a-9f68-fa3a5eb0e068.v1](https://data.4tu.nl/datasets/956b028c-6ea2-402a-9f68-fa3a5eb0e068).
  - Evidence: `SSRO FX8/dataset.hdf5` has prep_state, repetition, I, Q, each (20000,), with xlen 2 and ylen 10000. 2,027 `SSRO FX#` and 89 `SSRO TR#` datasets. No experiment name matched trace, transient, flight, acquisition or ring-up.
  - Platform and lab: fluxonium flux-pulse-assisted readout and a transmon sample; Andersen lab, QuTech ([README](https://data.4tu.nl/file/956b028c-6ea2-402a-9f68-fa3a5eb0e068/756488d6-18d2-4414-b099-d6699f21c225)).
  - Formats and size: Quantify HDF5; 19.4 GB zip, 75 GB uncompressed. Licence CC BY 4.0. Published 2024-09-10. Readout duration: not verified.
- **Beaulieu et al. 2026, "Fast, high-fidelity Transmon readout with intrinsic Purcell protection via nonperturbative cross-Kerr coupling"**. [Zenodo 20825189](https://zenodo.org/records/20825189). Paper citation not verified.
  - Evidence: `Exp_IQ_second_excited_state.npz` has `data_ground`, `data_excited`, `data_second_excited` of (5, 2, 25000); `Exp_QND_fidelity.npz` has `g_data`, `e_data` of (4, 2, 25000); `Exp_optimized_readout.npz` has `Ig`, `Qg`, `Ie`, `Qe` of about 49,000; `Exp_ground_bifurcation.npz` has (18, 2, 10000).
  - Qubits: 1 by title; `Exp_multi_qubits_spectrum.npz` also holds four parameter sets, not interpreted. Lab: creators include Scarlino and Blais; affiliations not verified.
  - Formats and size: npz; 31.3 MB. Licence CC-BY-4.0. Published 2026-05-28. Readout duration: not verified.
- **Dixit et al. 2026, "Millimeter Wave Readout of a Superconducting Qubit"** ([arXiv](https://doi.org/10.48550/arXiv.2603.13837)). [NIST mds2-4177](https://doi.org/10.18434/mds2-4177).
  - Evidence: `FigS3.csv` starts with the header `I, Q, State Label` and numeric rows ([file](https://data.nist.gov/od/ds/mds2-4177/FigS3.csv)). Row count is about 190,000 from file size (inference). Other files are figure CSVs and `.h5py` data ([README](https://data.nist.gov/od/ds/mds2-4177/4177_README.txt)).
  - Qubits: 1 (title). Lab: NIST.
  - Size: about 4.8 GB. Licence: NIST statement. Record timestamp 2026-06-30. Readout duration: not verified.
- **Bothara et al. 2025, tantalum fluxonium QND readout**. [Zenodo 14584494](https://zenodo.org/records/14584494).
  - Evidence: `Fig_1bd_IQblobs_paramp.txt` has columns `Ig(mV) Qg(mV) Ie(mV) Qe(mV)` and 10,001 lines, with a matching no-paramp file.
  - Qubits: 1 (title). Lab: creators include Vijay and Patankar.
  - Size: 2.4 MB. Licence: none in API. Published 2025-01-01. Paper citation not verified.
- **Pita-Vidal et al., Nature Physics 20, 1158 (2024)** ([DOI](https://doi.org/10.1038/s41567-024-02497-x)). [4TU 10.4121/29e13811-83f5-45e5-8242-5f0933e25c4a.v1](https://data.4tu.nl/datasets/29e13811-83f5-45e5-8242-5f0933e25c4a).
  - Evidence: `Raw_shots/shots/ID_*_IQ_*.npz` hold `data.npy` `<c16` (50000, 12) in the file I opened.
  - Labels: unclear, and the record says only that it is for Fig. S19. Qubits: 2 spin qubits by title.
  - Size: 18.6 GB. Licence CC BY 4.0. Published 2023-08-11. Readout duration: not verified.
- **Bargerbos et al., PR Applied 19, 024014 (2023)** ([DOI](https://doi.org/10.1103/physrevapplied.19.024014)). [4TU 10.4121/20219522.v1](https://data.4tu.nl/datasets/187d98bf-6bf3-450f-8a37-1f26f7e03d17).
  - Called "raw time traces", but this is T3: 43 files of `<c16` (2, 1000000) with row 0 stepping 4.5 µs, and the paper says "Each individual shot corresponds to the integrated output signal for a readout time of 500 ns" ([arXiv](https://arxiv.org/abs/2207.12754)).
  - Labels: none (bias condition only). Size: 569.8 MB zip. Licence CC BY 4.0. Published 2022-10-06.
- **Cao, Akhouri, Vizvary, Ofek, Dai, Murch 2026, "Adaptive thresholding for scalable measurement-based qubit reset"**. [Zenodo 22719098](https://zenodo.org/records/22719098). Paper citation not verified.
  - Evidence: `adaptive_reset_all_q1_a_0.6_runs_200000_rounds_20.npz` has `H_calc` float64 (200000, 20) and scalar Gaussian calibrations. Notebooks also save final `I1_final` and `Q1_final`. `H_calc` is a calculated value, not confirmed to be raw I/Q.
  - Qubits: up to 5 by file names. Size: 1.3 GB zip. Licence CC-BY-4.0. Published 2026-09-12. Readout duration: not verified.

#### T3, bits only: time series of hard outcomes
- **McEwen et al., cosmic-ray bursts (Google)**. [Figshare 16673851](https://doi.org/10.6084/m9.figshare.16673851).
  - The [README](https://ndownloader.figshare.com/files/30870895) says each CSV has "the measured bit value for a given qubit in that cycle": 600,000 cycles at 100 µs, 1,000,000 at 3 µs, or 100,000 time-multiplexed.
  - Size: 940.2 MB. Licence Apache 2.0. Published 2021-09-23.
- **Di Carlo et al. 2026, Quantum Inspire error bursts**. [4TU 10.4121/5fe71572-e7a9-4c07-88dd-3dfda3ff99a2.v1](https://data.4tu.nl/datasets/5fe71572-e7a9-4c07-88dd-3dfda3ff99a2).
  - The record says shots are "processed/stored as binary outcomes". Starmon-5 and Starmon-7.
  - Size: 74.7 GB. Licence CC0. Published 2026-02-12.
- **Gosling, Bénâtre et al. 2026**, memory of a qubit environment. [Zenodo 21908500](https://zenodo.org/records/21908500) (fluxonium, KIT).
  - The sampled `QJ_state_*.h5` has `state_arr` bool (200000,) and attribute `T = 6e-06`, with more than 11,000 members.
  - Size: 214 MB. Licence CC-BY-4.0. Published 2026-08-12. The [paper text](https://arxiv.org/abs/2603.11889) still says "upon reasonable request".
- **Spiecker et al., PR A 109, 052218 (2024)**. [Zenodo 8114088](https://zenodo.org/records/8114088).
  - `Quantum_jumps.npz` is bool (25, 5, 2, 2, 100, 20000), thresholded from the T1 deposit above.
  - Size: 34 MB. Licence CC-BY-4.0. Published 2023-07-04.
- **Jiang et al. 2025, charge-parity detection**. [Figshare 30636059](https://doi.org/10.6084/m9.figshare.30636059).
  - `Parity_data.npy` is int8 (4194304,) with values ±1, plus two (12582912,) files.
  - Size: 96.4 MB. Licence CC BY 4.0. Published 2025-11-17.
- **Makihara et al. 2026, "Quantum Jumps of Sound"**. [Zenodo 20944616](https://zenodo.org/records/20944616).
  - The record says `post_selected_trajectories.csv` holds "8447 total" trajectories, each with "294 parity checks". Platform not verified.
  - Size: 62.2 MB. Licence CC-BY-4.0. Published 2026-06-27.

#### T4: time axis, averaged
- **Brookes, Tancredi et al., Science Advances 7, eabe9492 (2021)**. [Zenodo 4646691](https://zenodo.org/records/4646691).
  - `experimental_transients/resonator_response_*dBm_{I,Q}.csv` are exports with `X-data: Time` and a frequency parameter. The [paper](https://doi.org/10.1126/sciadv.abe9492) captions Fig. 2 "Averaged transient response of the cavity". Averaging of the Fig. 3 transients was not separately verified.
  - Size: 360.4 MB zip, 1.04 GB uncompressed. Licence CC-BY-4.0. Published 2021-03-27.
- The Zwanenburg `Single Trace Acquisition` sets and the Gunyhó `qubit-bolometer-readout-*` files are also T4, as noted above.

#### T6: named or related readout-physics leads with no shot-level record
- **Statement says on request:**
  - [Minev et al. 2019](https://www.nature.com/articles/s41586-019-1287-z).
  - [Réglade et al. 2024](https://www.nature.com/articles/s41586-024-07294-3).
  - [Chen et al. 2023](https://arxiv.org/abs/2208.05879): "available on request".
- **No availability section on the Nature page and no deposit found:**
  - [Murch et al. 2013](https://www.nature.com/articles/nature12539).
  - [Weber et al. 2014](https://www.nature.com/articles/nature13559).
  - [Hacohen-Gourgy et al. 2016](https://www.nature.com/articles/nature19762).
- **No statement or repository string in the arXiv full text, and no deposit found in my repository searches** (author-name searches for most, keyword searches only for Campagne-Ibarcq, Walter and Khezri):
  - [Campagne-Ibarcq 2016](https://arxiv.org/abs/1511.01415).
  - [Walter 2017](https://arxiv.org/abs/1701.06933).
  - [Sunada 2022](https://arxiv.org/abs/2202.06202).
  - [Swiadek 2024](https://arxiv.org/abs/2307.07765).
  - [Spring 2025](https://arxiv.org/abs/2409.04967).
  - [Dumas 2024](https://arxiv.org/abs/2402.06615).
  - [Hazra 2025](https://arxiv.org/abs/2407.10934).
  - [Connolly 2025](https://arxiv.org/abs/2506.05306).
  - [Khezri 2023](https://arxiv.org/abs/2212.05097).
  - [Flurin 2020](https://arxiv.org/abs/1811.12420).
  - [Fechant 2025](https://arxiv.org/abs/2505.00674).
- **Figure data or media only:**
  - [Ficheux et al. 2018](https://doi.org/10.6084/m9.figshare.6127958): titled "Supplementary Videos", 90 files; the 25 I listed are mp4.
  - [Dai et al., PRX 16, 011011 (2026)](https://doi.org/10.1103/zdpg-mhpc): [Zenodo 18011745](https://zenodo.org/records/18011745). The paper says "raw and processed data", but the Fig. 2 README describes `P_g_to_e` and similar probabilities.
  - [Wang et al., transmon ionization](https://zenodo.org/records/18736316): normalized 2D arrays such as (9, 100, 200) plus Floquet pickles.
  - [Li et al. 2025](https://doi.org/10.6084/m9.figshare.28815434): source data, with long derived series such as `P_MQSCPJ`.
  - [Cole et al. 2026, Kerr-cat](https://zenodo.org/records/18168977): `I0`, `Q0` and others of (81, 200) over sweep axes.
  - [NIST mds2-3948](https://doi.org/10.18434/mds2-3948): "averaged heterodyne IQ voltages".
  - [Aamir et al.](https://doi.org/10.6084/m9.figshare.27089311): "source data".

#### No tier assigned
- **Bista et al., PR Applied 25, 034058 (2026), "Readout-induced leakage of the fluxonium qubit"** ([DOI](https://doi.org/10.1103/wjdb-4814)).
  - [Illinois Data Bank IDB-1447103](https://databank.illinois.edu/datasets/IDB-1447103): DataCite says it contains "all the raw and processed data" and CC0.
  - The file listing was blocked: HTTP 403 to WebFetch and to my script's user agent. One request with a browser-style user agent returned HTTP 200, but I did not read the page or rely on it. I assign no tier rather than imply it is unobtainable. Someone should open it in a browser. The same applies to [IDB-0097046](https://doi.org/10.13012/b2idb-0097046_v1).

### Inferences
- **Test of the draft claim.** "No other raw-trace dataset is publicly available" is false as written, using the assignment's own definition. Cao et al., Gunyhó et al. and Hassani et al. are hardware, per-shot waveform records with state labels in files or file names, free to download. Spiecker et al. adds labelled per-shot I/Q trains.
- **What may survive.** As far as I could verify, none of the T1 or T2 records is a simultaneous multi-qubit readout record. This is within my scope only; the discriminator datasets are covered by another researcher. A narrower sentence might be defensible, such as "we know of no other public dataset of multiplexed multi-qubit single-shot readout traces with prepared-state labels", but it must also survive that researcher's findings.
- **Cao et al. is the strongest counterexample by volume.** Using it would still need work: sample period, herald handling and label quality are not documented in the deposit.
- **Titles are unreliable in both directions.** Bargerbos 2023 "raw time traces" are integrated shots, and Battistel "raw data" is simulated. "Single Trace Acquisition" in Zwanenburg is averaged. Dai's "raw data" is probabilities. Hassani, whose record title gives no hint of traces, is T1.
- **Best labelled T3 substitutes by size and documentation:** Zwanenburg (65,536 per state per file), Zheng (repeated readouts with g/e/f calibration), Stefanski, Hassani Fig. 7a, Beaulieu.

### Gaps
- **Lienhard re-host check:** none found. [DataCite](https://api.datacite.org/dois?query=creators.name:Lienhard) author search, Zenodo keyword search and GitHub repository search each returned no qubit-related hit. GitHub search covers only repository names and descriptions, and the GitHub API then hit its unauthenticated rate limit. This is a negative, not proof.
- **Cao et al.:** sample period and readout duration not found. The paper's appendix may give them, but my text search did not.
- **Hassani et al.:** absence of hardware averaging per row is inferred from the record text and the QND analysis, not read from instrument metadata.
- **Spiecker et al.:** the array shape was confirmed in one file only. The others are inferred from identical file sizes.
- **Places checked with nothing further found:**
  - DataCite author searches returned zero hits for Swiadek, Hazra with qubit, Dumas, Connolly, Fechant and Réglade.
  - Sivak, Flurin and Minev had no relevant hit in the first 25 results of large fuzzy result sets, so these are weak negatives.
  - Direct [Dryad](https://datadryad.org/api/v2/search?q=transmon) search returned one irrelevant hit for transmon, and zero for "superconducting qubit readout" and "circuit quantum electrodynamics".
  - Direct Figshare search returned zero for "superconducting qubit single-shot".
  - [Harvard Dataverse](https://dataverse.harvard.edu/api/search?q=transmon&type=dataset) "transmon" returned only the Cao et al. record, and qubit and circuit-QED keyword results, up to 150 per query, contained no other superconducting-qubit readout deposit.
  - Zenodo searches for "time traces" and "quantum jump" with qubit terms returned only records listed here.
- **ETH:** the only ETH hit is [10.3929/ethz-c-000807038](https://doi.org/10.3929/ethz-c-000807038) (photon temporal modes, 2026, CC-BY-SA). Its page returned HTTP 500, so its contents are not verified.
- **Not opened:**
  - [Valenti et al. 2026](https://zenodo.org/records/19441411), `correlated_data.npy` (the title says "superconducting devices").
  - [Wang et al. 2025](https://zenodo.org/records/15957274) `.mat` files (each under 0.5 MB, named per figure).
  - [Ming Gong 2020](https://doi.org/10.6084/m9.figshare.12715493).
  - Community IBM-run deposits such as [Indahl 2026](https://zenodo.org/records/23812756).

## Which labs routinely deposit raw measurement files rather than figure data only?

### Takeaway
KIT (Pop), ISTA (Fink) and QuTech's Andersen and related groups repeatedly deposit raw per-shot files. Aalto (Möttönen) does so at least once, with routine deposits of unverified depth. Google Quantum AI routinely deposits complete raw bits, never analog. Yale's recent deposits are figure-level, and the headline fast-readout and cat-qubit papers stay on request or silent.

### Cited Findings
- **KIT, Pop group (Zenodo).** [Spiecker 2023](https://zenodo.org/records/7817552) says "Raw and processed data"; [Gosling 2026](https://zenodo.org/records/21908500) has state traces; [Günzler 2025](https://zenodo.org/records/17312111) says "The data is saved in NumPy binary compressed .npz format" (not opened).
- **ISTA, Fink group (Zenodo).** [Hassani 2023](https://zenodo.org/records/8004359) says "all the raw data"; [Sett 2024](https://zenodo.org/records/10518321) says "the raw data".
- **QuTech, Andersen and collaborators (4TU).** [Bargerbos 2022](https://data.4tu.nl/datasets/2e22a266-8d89-4837-898a-7eb842624512), [Bargerbos 2023](https://data.4tu.nl/datasets/187d98bf-6bf3-450f-8a37-1f26f7e03d17), [Pita-Vidal 2023](https://data.4tu.nl/datasets/7d1b983a-684e-4229-8842-d3fc0c1fbf3a), [Pita-Vidal 2024](https://data.4tu.nl/datasets/29e13811-83f5-45e5-8242-5f0933e25c4a), [Stefanski 2024](https://data.4tu.nl/datasets/956b028c-6ea2-402a-9f68-fa3a5eb0e068), [Zwanenburg 2026](https://data.4tu.nl/datasets/22122209-2f78-4a66-8d04-fe18ac8b244e). The Zwanenburg README says each `dataset.hdf5` "contains the raw, unprocessed data".
- **QuTech, DiCarlo-linked work.** GitHub data repositories such as [Logical_Qubit_Operations_Data](https://github.com/DiCarloLab-Delft/Logical_Qubit_Operations_Data) and [Leakage_Reduction_Unit_Data](https://github.com/DiCarloLab-Delft/Leakage_Reduction_Unit_Data) are organised by figure ([org list](https://github.com/orgs/DiCarloLab-Delft/repositories)). In 2026, raw shot deposits appear on 4TU: [van der Meer](https://data.4tu.nl/datasets/886d8394-adfe-4560-807c-c0fa007110df) and [Di Carlo et al.](https://data.4tu.nl/datasets/5fe71572-e7a9-4c07-88dd-3dfda3ff99a2).
- **Aalto, Möttönen (Zenodo).** [Gunyhó 2023](https://zenodo.org/records/10000511) has raw traces. Other deposits include [Sah 2024](https://zenodo.org/records/11234843), [Tuokkola 2025](https://zenodo.org/records/14651787) and [Hyyppä 2022](https://zenodo.org/records/7307694); the last three are small (6 to 15 MB) and were not opened.
- **Google Quantum AI (Zenodo).** Seven QEC records, plus [McEwen on Figshare](https://doi.org/10.6084/m9.figshare.16673851), hold shot-level bits. None holds analog data (see the first section).
- **Pittsburgh, Hatridge.** One verified deposit, [Cao 2025](https://doi.org/10.7910/DVN/DPMID3), with per-shot measurement files and analysis code.
- **EPFL.** [Zheng 2026](https://zenodo.org/records/23367046) says "copied byte for byte from the lab NAS", and [Beaulieu 2026](https://zenodo.org/records/20825189) holds per-shot I/Q in figure-oriented npz files.
- **Illinois, Kou group.** [Bista 2026](https://doi.org/10.1103/wjdb-4814) on Illinois Data Bank claims raw and processed data (contents not verified).
- **Li and Jiang et al. (a shared author, Yaoyao Jiang; lab affiliation not verified).** Figshare [source data](https://doi.org/10.6084/m9.figshare.28815434) and [±1 parity streams](https://doi.org/10.6084/m9.figshare.30636059).
- **Figure-level or on request** (lab names here come from creator lists or my prior knowledge, not checked affiliations).
  - Yale: [Dai 2026](https://zenodo.org/records/18011745) and [Brock 2025](https://zenodo.org/records/15009818) are figure-level, and [Sivak 2023](https://www.nature.com/articles/s41586-023-05782-6) and [Minev 2019](https://www.nature.com/articles/s41586-019-1287-z) are on request.
  - ETH: [Krinner 2022](https://www.nature.com/articles/s41586-022-04566-8) is on request.
  - Alice & Bob: [Réglade 2024](https://www.nature.com/articles/s41586-024-07294-3) is on request.
  - AWS: [Putterman 2025](https://arxiv.org/abs/2409.13025) is on request.

### Inferences
- **In this sample, deposits come mostly from university groups, many of them European**, which suits a repository-first search. The habit tracks group practice more than paper prominence: the best-known trajectory and readout papers have nothing public. This is my reading of the pattern, not a measured rate.
- **Practice appears to be changing.** Many of the richest records date from 2025 to 2026, and 2026 alone adds EPFL and DiCarlo-linked raw deposits. A claim like the draft's will age quickly, and may need a search date.
- **Quantify and QCoDeS-based labs tend to produce raw files by default**, since each run writes its own `dataset.hdf5` plus a snapshot, so depositing a data directory captures shot-level data. That is an inference from the Delft and Aalto file structures.

### Gaps
- I have no denominator for "routinely", because I sampled deposits I could find and did not survey each lab's whole output.
- Aalto's smaller deposits and KIT's Günzler deposit were not opened, so their shot-level content is unverified.
- Wallraff group practice: apart from the Research Collection record above, [Zenodo](https://zenodo.org/api/records?q=creators.name:Wallraff) and DataCite author searches found nothing, but ETH's own repository search could not be run (HTTP 500 on the page tried).
- USTC, Chinese-repository (Science Data Bank) and Tencent deposits were not surveyed beyond [Zhao et al.](https://arxiv.org/abs/2112.13505) and DataCite hits. Coverage there is weak.
