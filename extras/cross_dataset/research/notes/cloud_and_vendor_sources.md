# Cloud and vendor sources of raw qubit readout traces (checked 2026-11-06)

Tier key used throughout. T1: real hardware, per-shot, time-resolved, downloadable now, labels present or derivable. T2: as T1 but labels unclear, format undocumented or very few shots. T3: real hardware, per-shot, integrated only. T4: real hardware, time axis, averaged only. T5: simulated. T6: not obtainable (on request, dead link, code only). "Not verified" means I did not see it in a primary source. No jobs were submitted, no accounts created, and no file over about 50 MB was downloaded. Shapes marked "read by me" come from `.npy` headers read with HTTP range requests or from small files opened with `allow_pickle=False`.

## Does each cloud provider offer, or has it offered, per-shot time-resolved capture?

### Takeaway
Rigetti QCS is the only provider whose documentation shows per-shot raw ADC capture, and its worked examples are on older Aspen devices; IQM's current SDK exposes a time-trace acquisition whose per-shot behaviour on the cloud I could not confirm. IBM now returns at most one integrated I/Q point per shot and withdrew pulse access in February 2025, Braket returns a bit, OQC's OpenPulse front end gives time-resolved data only as a shot average, and Quantum Inspire returns bits.

### Cited Findings

#### Route summary

| Route | API feature | Per-shot? Time axis? | Devices named | Status, with date | Cost or access |
|---|---|---|---|---|---|
| IBM level 0 | `meas_level=0`, `meas_return="single"` through `backend.run` | Per-shot with time axis in the SDK only; no device snapshot advertises it | None found | `backend.run` ended 15 Oct 2024; pulse control removed 3 Feb 2025 | Withdrawn |
| IBM kerneled | Sampler V2 `meas_type="kerneled"` | Per-shot, no time axis | Runtime backends | Live, `stable/0.51` docs read 2026-11-06 | Open plan free up to 10 min/month; from $96/min pay-as-you-go |
| Braket Pulse | `capture_v0` | Per-shot bit, no time axis | Rigetti Ankaa-3 and Cepheus-1-108Q in examples | Live, docs read 2026-11-06; OQC Lucy removed June 2024 | $0.30/task plus $0.000425/shot on Cepheus-1-108Q |
| Rigetti QCS | Quil-T `RAW-CAPTURE` | Per-shot, time axis, raw ADC | Aspen-8 and Aspen-M-3 in docs; Ankaa and Cepheus not verified | Documented in pyQuil 4.17 docs; current-device support not verified | QCS account; prices not verified |
| Azure Quantum (Rigetti) | Quil-T with `skipQuilc: true` | `RAW-CAPTURE` on Azure not verified | Cepheus-1-108Q | Page updated 2026-04-10 | Azure pricing not verified |
| OQC (QAT) | `capture_v1`, `capture_v2`, `capture_v3` | Integrated per-shot, bit, or averaged trace; per-shot `RAW` mode exists below the front end | Toshiko-era cloud not verified | QAT `main` read 2026-11-06 | Not verified |
| IQM Resonance | Pulla, `Measure.time_trace()`, `TimeTrace` | Time axis yes; per-shot not verified | Named on Resonance dashboard | iqm-pulse and iqm-pulla 15.0.0, 2026-10-06 | Free Starter tier, from $0.30/s; tier of pulse access not verified |
| Quantum Inspire | Raw data file of final bits | Per-shot bits, no time axis | Tuna-5/9, Starmon-7, Spin-2+ | Page shows © 2026, no date | Not verified |
| IQCC | QUA `adc_stream`, `save_all` | Per-shot ADC in QUA; cloud policy not verified | OPX1000 systems | Client 0.22.0 on PyPI | Priority for Israeli users; terms not verified |

#### IBM
- The Qiskit Pulse paper defines the levels: "The lowest level accessible to the user, level-zero or raw data, typically corresponds to a digitized time-series signal", and says users can retrieve data "by specifying the readout data-level, i.e., zero, one, or two". Its only device-specific statement concerns level one: users of IBM processors "may request the kerneled data". — [Alexander et al., arXiv:2004.06755](https://ar5iv.labs.arxiv.org/html/2004.06755)
- Qiskit 1.4 documents the level-0 result shapes: `single` gives `[shots, memory_slots, memory_slot_size]`, and `avg` gives `[memory_slots, memory_slot_size]`. `MeasLevel.RAW = 0`. — [qiskit/result/result.py, stable/1.4](https://github.com/Qiskit/qiskit/blob/stable/1.4/qiskit/result/result.py); [qiskit/qobj/utils.py](https://github.com/Qiskit/qiskit/blob/stable/1.4/qiskit/qobj/utils.py)
- I read `meas_levels` in all 68 IBM configuration snapshots shipped with qiskit-ibm-runtime. None lists level 0: 52 list `[1, 2]`, `ibmq_poughkeepsie` lists `[1]`, and 15 omit the field. That includes pulse-enabled devices such as `ibmq_armonk` and `ibmq_almaden`, and 50 snapshots have `open_pulse: true`. — [fake_provider/backends](https://github.com/Qiskit/qiskit-ibm-runtime/tree/main/qiskit_ibm_runtime/fake_provider/backends)
- HERQULES (ISCA 2023) states: "Access to ADC data traces on IBM Quantum Cloud is restricted." — [Maurya et al., arXiv:2212.03895, §3.1](https://ar5iv.labs.arxiv.org/html/2212.03895)
- IBM's changelog: on 15 October 2024 "Support for `backend.run` has ended"; on 3 February 2025 "Pulse-level control on all IBM Quantum processors has been removed." The page's newest section is November 2026. — [IBM changelog](https://quantum.cloud.ibm.com/docs/en/guides/latest-updates)
- `qiskit.pulse` was deprecated in Qiskit SDK 1.3.0 and scheduled for removal in 2.0.0; IBM points users to fractional gates and to Qiskit Dynamics. — [IBM pulse migration guide](https://quantum.cloud.ibm.com/docs/en/guides/pulse-migration)
- Current Sampler V2 options are `classified`, `kerneled` ("A complex NumPy array of shape `(*pub_shape, num_shots, creg_size)`, where each entry represents an IQ data point") and `avg_kerneled`. No raw or time-resolved option appears. Source link points to `stable/0.51`. — [SamplerExecutionOptionsV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-sampler-execution-options-v2)
- Prices: Open plan free up to 10 minutes per month, Pay-As-You-Go from $96/minute, Flex from $72/minute, Premium from $48/minute. — [IBM pricing](https://www.ibm.com/quantum/pricing)

#### Amazon Braket
- The SDK docstring for `capture_v0` reads "Adds an instruction to capture the bit output from measuring the specified frame." — [pulse_sequence.py](https://github.com/amazon-braket/amazon-braket-sdk-python/blob/main/src/braket/pulse/pulse_sequence.py)
- The developer guide lists "capture_v0: Returns the value on a capture frame to a bit register" as the only capture function, and its OpenPulse example writes `psb[0] = capture_v0(...)` into `bit[1] psb`. — [OpenQASM device support](https://docs.aws.amazon.com/braket/latest/developerguide/braket-openqasm-device-support.html); [Hello Pulse](https://docs.aws.amazon.com/braket/latest/developerguide/braket-hello-pulse.html)
- The vendor example notebook targets Cepheus-1-108Q and reads `measurement_counts` after `capture_v0`. — [1_Bringup_experiments.ipynb](https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/pulse_control/1_Bringup_experiments.ipynb)
- Device schemas carry a `pulse` field for Rigetti and OQC, and none for IQM, IonQ or AQT (I grepped the schema files). — [amazon-braket-schemas-python](https://github.com/amazon-braket/amazon-braket-schemas-python/tree/main/src/braket/device_schema)
- Current hardware providers are AQT, IonQ, IQM, QuEra and Rigetti; OQC is absent. The Rigetti rows are Ankaa-3 and Cepheus-1-108Q. — [Braket devices](https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html)
- Braket Pulse launched in October 2022 on Rigetti and OQC hardware at no extra cost (seen in a search extract, page not opened). — [AWS What's New](https://aws.amazon.com/about-aws/whats-new/2022/10/amazon-braket-supports-pulse-level-access-study-performance-todays-quantum-computers/)
- "Access to OQC's Lucy system was previously available via Amazon's London region in the UK, but access was removed in June 2024." (secondary source, dated 27 Aug 2024). — [SDxCentral](https://www.sdxcentral.com/news/rigettis-84-qubit-ankaa-2-quantum-computer-available-through-aws-braket-service/)
- The AWS/OQC Lucy tutorial from 23 Aug 2023 reports only excited-state probability, and the part I could read does not mention raw, I/Q or time-resolved results. — [AWS blog](https://aws.amazon.com/blogs/quantum-computing/how-to-use-pulse-level-control-on-oqcs-superconducting-quantum-computer)
- Pricing lists Cepheus-1-108Q at $0.30/task and $0.000425/shot. Ankaa-3 is not listed there although the device page lists it, and pulse is not mentioned. — [Braket pricing](https://aws.amazon.com/braket/pricing/)
- For probability-returning devices such as IonQ, "per-shot measurements are not available". — [Braket result types](https://docs.aws.amazon.com/braket/latest/developerguide/braket-result-types.html)

#### Rigetti (QCS, Quil-T)
- `CAPTURE` values are demodulated and integrated on hardware. "What `RAW-CAPTURE` does is give you, the user, access to the raw values of that passband signal." — [pyQuil notebook](https://github.com/rigetti/pyquil/blob/master/docs/source/quilt_raw_capture.ipynb)
- Limits: at most 8192 samples per shot (error text suggests under 4.096 µs), at most 2^24 samples per job, one `RAW-CAPTURE` per program, and no `CAPTURE` in the same program. Results are integers in [-2^15, 2^15], returned as a flat array to reshape. The Aspen-M-3 frame is 2 GS/s and the printed example is `shape (1000, 4160)`. — [pyQuil raw capture docs](https://pyquil-docs.rigetti.com/en/stable/quilt_raw_capture.html)
- An older page is titled "RAW-CAPTURE on Aspen-8", also at 2 GHz. — [pyQuil v3.5.2](https://pyquil-docs.rigetti.com/en/v3.5.2/quilt_raw_capture.html)
- A search extract of the QCS docs says available data "includes raw ADC samples captured by a Quil RAW-CAPTURE, one for each cycle of the ADC". A direct fetch returned only a page title, so this is not verified in a loaded page. — [QCS docs](https://docs.rigetti.com/qcs/guides/quil/customizing-program-readout)
- Another search extract mentioned an `ENABLE-RAW-CAPTURE` frame attribute on Ankaa and later devices; a second targeted search did not reproduce it. Not verified, conflicting. — [QCS language support](https://docs.rigetti.com/qcs/guides/quil/language-support)
- `get_raw_readout_data` is pyQuil v4's accessor, and the next-generation compiler backend is "required for Ankaa-family QPUs". — [Introducing pyQuil v4](https://pyquil-docs.rigetti.com/en/stable/introducing_v4.html)
- Azure: "All of Rigetti's publicly available QPUs are available through Azure Quantum", with Cepheus-1-108Q the only QPU target listed. Quil-T requires native Quil and `skipQuilc: true`. Page metadata shows `updated_at: 2026-04-10`. — [Azure Rigetti provider](https://learn.microsoft.com/en-us/azure/quantum/provider-rigetti)
- qBraid notes "Quil-T instructions are not supported by quilc or the QVM", and for a Cepheus calibration flag, "Rigetti restricts the flag that lifts this to certain users". — [qBraid Rigetti guide](https://docs.qbraid.com/v2/sdk/user-guide/providers/rigetti)
- Rigetti's get-quantum page shows no prices or access terms in the fetched text. — [Rigetti](https://www.rigetti.com/get-quantum)

#### OQC
- QAT defines three acquisition modes: `RAW` "Returns the time-series acquisition data for each shot", `SCOPE` "Returns the time-series acquisition data where each point is averaged over the number of shots", and `INTEGRATOR` returns "a single data point" per shot, with the note "different backends will only allow selected acquisition modes". — [instruction_basetypes.py](https://github.com/oqc-community/qat/blob/main/src/qat/ir/instruction_basetypes.py)
- QAT's QASM3 parser: `capture_v0` raises `NotImplementedError`, `capture_v1` returns an integrated I/Q value, `capture_v2` a discriminated bit, `capture_v3` uses `AcquireMode.SCOPE` ("averaging over all the shots"), and `capture_v4` is not implemented. — [qasm3.py](https://github.com/oqc-community/qat/blob/main/src/qat/frontend/parsers/qasm/qasm3.py)
- QAT's live backend code has a branch for `AcquireMode.RAW`. — [purr/backends/live.py](https://github.com/oqc-community/qat/blob/main/src/qat/purr/backends/live.py)
- qBraid's OQC guide gives `results_format` as `raw` or `binary`, without defining `raw`. — [qBraid OQC guide](https://docs.qbraid.com/sdk/user-guide/runtime_oqc)
- AWS said users of "Amazon Braket and OQC's private cloud" had pulse-level access, as of 23 Aug 2023 (search extract). — [AWS blog](https://aws.amazon.com/blogs/quantum-computing/how-to-use-pulse-level-control-on-oqcs-superconducting-quantum-computer)

#### IQM
- `iqm.pulse` documents `TimeTrace`: "Capture the raw IQ signal without integration." Release 15.0.0, updated 2026-10-06. — [iqm-pulse instructions](https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.playlist.instructions.html)
- The `iqm_pulse-15.0.0` wheel has `Measure.time_trace()`: "Returns a multiplexed simultaneous measurement with an additional time trace acquisition", keyed by probe line, default key `readout.time_trace`. — [iqm-pulse on PyPI](https://pypi.org/project/iqm-pulse/)
- The `iqm_pulla-15.0.0` wheel has `_add_time_trace_return_parameters`, with a `use_ssro` flag documented as "Iff True, individual shots are returned from the station", set when `averaging_bins > 1`. That function's own docstring leaves `use_ssro` as "TODO". — [iqm-pulla on PyPI](https://pypi.org/project/iqm-pulla/)
- Pulla can point at Resonance through `IQM_SERVER_URL=https://resonance.iqm.tech`, with a personal token. The quick start does not mention time traces. — [Pulla Quick Start](https://docs.iqm.tech/iqm-pulla/Quick%20Start.html)
- Resonance: Starter is free with "up to 30 credits per month", pay-as-you-go starts at $0.30/s, and "Pulse-level access" is a listed feature, but the tier checkmarks did not survive text extraction. — [IQM Resonance](https://www.iqm.tech/products/iqm-resonance)

#### Quantum Inspire
- "the final measurement results will be stored in the binary register and written to the raw data file". Backends are Tuna-5/9, Starmon-7 and Spin-2+. Pulse-level access is not mentioned. — [QI hardware backends](https://www.quantum-inspire.com/kbase/hardware-backends/)

#### Others
- IQCC: `iqcc-cloud-client` 0.22.0 is a "Client for remote (cloud) acccess to IQCC quantum and classical resources." — [PyPI JSON](https://pypi.org/pypi/iqcc-cloud-client/json)
- IQCC uses OPX1000 and QUA, and gives priority and discounted access to Israeli academia and industry while being open to others (search extract). — [Quantum Machines press release](https://quantum-machines.co/press_release/quantum-machines-opens-the-israeli-quantum-computing-center)
- QUA has `measure(..., adc_stream=...)`: "The stream variable into which the raw ADC data will be saved", with the example `adc_st.input1().save_all("raw_adc_stream")`. — [QUA API](https://docs.quantum-machines.co/latest/docs/API_references/qua/dsl_main/)
- Quafu, Tianyan and Origin: one search surfaced only circuit-level descriptions. — [Tianyan, arXiv:2512.10504](https://arxiv.org/html/2512.10504v2); [Origin qcloud result docs](https://qcloud.originqc.com.cn/document/pyqpanda3-docs/en/api/qcloud/result)

### Inferences
- IBM's level 0 exists as an SDK data type, but I found no evidence of general availability on any device. The accurate statement is that per-shot kerneled data is the deepest documented level today, and that the `backend.run` path through which `meas_level` was set has been closed since October 2024.
- The Rigetti route is the cleanest user-capture option on paper. The per-job cap of 2^24 samples allows about 4,032 shots at the 4,160-sample length in the docs, so a dataset would need many jobs.
- Braket's Rigetti route cannot produce traces even though the same hardware can through QCS, because Braket exposes only `capture_v0`.
- For OQC, a user going through OpenPulse can get averaged traces (T4-type capture). Per-shot `RAW` would need the Python-level builder and a backend that allows it.

### Gaps
- Whether `RAW-CAPTURE` works today on Ankaa-3 or Cepheus-1-108Q through QCS or Azure could not be determined: `docs.rigetti.com` is JavaScript-rendered and returned only a title. Checking would need a QCS account.
- Whether IQM cloud time traces are per-shot or averaged, and which Resonance tier allows them, is not verified.
- OQC's own cloud documentation was not found, so cost, device list and `RAW` availability are not verified.
- I found no IBM statement that names level 0 and says it is unavailable. The IBM conclusion rests on snapshots, HERQULES and the removal dates.
- Pasqal, Quantinuum, QuEra and AQT primary docs were not read. One search returned nothing on them, so I can only report no evidence, not absence.

## Has anyone published data collected through these routes?

### Takeaway
I found no published time-resolved per-shot traces collected through any commercial cloud route. Three public records do contain per-shot integrated I/Q from IBM hardware (T3), and two of them call it "raw", which is not the same as time-resolved.

### Cited Findings

#### Records collected on cloud hardware
- **FCM discriminator data (T3).**
  - Repository: [Quasar-UniNA/FCM-based-QubitDiscriminator](https://github.com/Quasar-UniNA/FCM-based-QubitDiscriminator). Paper: not cited in the repo, which says only that it was submitted to ACM Transactions on Quantum Computing. Single commit 2026-02-04, no README, licence not shown.
  - Platform: `ibm_brisbane`, `ibm_kyoto`, `ibm_osaka`, `ibm_sherbrooke`, arranged as `{2,5,10}qubits/{ghz,w_state}/run0..9`.
  - Evidence (read by me): `run0/0s.npy` is `<c8` (8192, 127), `run0/data.npy` is `<c8` (8192, 10), so one complex value per shot per qubit with no time axis.
  - Shots 8192; qubits 127 in calibration files, up to 10 in data; samples per trace and sample rate not applicable; labels from `0s`/`1s` filenames (preparation inferred from names, not documented); `.npy`, 1,572 files, about 543 MB per the GitHub API. Collection dates not verified.
- **"A Readout Is Not an Idle" (T3).**
  - Paper: K. Indahl, Heuremen LLC; peer review not verified. Record: [10.5281/zenodo.23812756](https://zenodo.org/records/23812756) (concept 10.5281/zenodo.23169755), published 2026-10-30, CC BY 4.0. Mirror: [Hugging Face](https://huggingface.co/datasets/indahl/readout-run-data).
  - Evidence: in the code bundle, `run_iq.py` sets `smp.options.execution.meas_type = 'kerneled'` with `SHOTS = 2000` and says "every readout returned as a raw IQ point"; `iqjob_*.json` lists calibration arms with prepared state.
  - Platform: `experiments/2026-10-21-iq-run/iqraw_ibm_{fez,kingston,marrakesh}.npz`, about 90 MB each, in a 2.95 GB raw zip of 111 `.npz`. Inner array shapes not verified, because the files are over my download limit and compressed in the outer zip.
- **Dynamic compass code, Pittsburgh data (T3, inferred).**
  - Paper: [Lee et al., arXiv:2604.14296](https://arxiv.org/html/2604.14296). Record: [10.6084/m9.figshare.32539992](https://doi.org/10.6084/m9.figshare.32539992), published 2026-06-02, CC BY 4.0.
  - Evidence: the record says `pittsburgh_data.zip` "contains the raw IQ readout data, ACES results, and GMM estimators". The paper says "We capture the full information available in the IQ values from qubit readout" on `ibm_pittsburgh` on 24 February 2026, with calibration circuits of 4000 shots per state.
  - Files (zip directory read by me): 7.70 GB, 13.2 GB uncompressed, 24 `.pickle` files including `code_{1..5}.pickle` of 660 MB each. Not unpickled, so shape is not verified. The word "kerneled" does not appear on the arXiv page.

#### Papers that collected IBM data without a release I could find
- HERQULES used the Lienhard chip for its main results, and a figure caption mentions 131,000 shots on IBM Manila; the README says "Feel free to shoot me an email for dataset access." Tier T6. — [arXiv:2212.03895](https://ar5iv.labs.arxiv.org/html/2212.03895); [satvikmaurya/HERQULES](https://github.com/satvikmaurya/HERQULES)
- Quiroga et al. used `ibmq_bogota` signal data of 1024 to 4096 points and give no data or code link. Tier T6. — [arXiv:2112.00313](https://ar5iv.labs.arxiv.org/html/2112.00313)
- Azad and Zhang used 2048 shots per state on IBM five-qubit devices and give no availability statement. Tier T6. — [arXiv:2210.08574](https://ar5iv.labs.arxiv.org/html/2210.08574)

#### Places checked that found no time-resolved cloud data
- Zenodo: `meas_level`, `+"meas_level"`, `+kerneled`, `+readout +"IBM" +"kerneled"` all returned 0 hits; `"RAW-CAPTURE"` returned 11 hits, none relevant; `+qubit +readout +traces` returned only circuit-level IBM runs. — [Zenodo search](https://zenodo.org/search)
- DataCite: `qubit readout traces` returned 12 records, none described as cloud-collected traces (two spin-qubit records not opened); `transmon single-shot traces dataset` returned 0. — [DataCite API](https://api.datacite.org/dois)
- Hugging Face `readout` and `qubit` searches found no time-resolved qubit traces. Hits I opened were unrelated or non-trace: `readoutbench-r1` is clinical-trial forecasting, `QSBench-Readout-Demo` is synthetic circuits with flip noise, `sc-qubit-dataset` is literature metadata. — [Hugging Face datasets](https://huggingface.co/datasets)
- Rigetti's QCS paper datasets are CSV summaries (`readout-error-mitigation.csv`, `active-reset-calibration.csv` and similar), with no traces by file name. — [rigetti/qcs-paper](https://github.com/rigetti/qcs-paper)
- Sourcegraph public search for literal `meas_level=0` found only `test/python/result/test_result.py` in Qiskit, a synthetic unit test; `RAW-CAPTURE` matched only pyQuil among relevant repos; `capture_v0` in notebooks returned 0. — [Qiskit test_result.py](https://github.com/Qiskit/qiskit/blob/main/test/python/result/test_result.py)

### Inferences
- "Raw IQ" in the 2026 IBM records means per-shot kerneled points, not traces. Any report should keep the two apart.
- IBM kerneled data needs only a standard Sampler option, which fits with T3 dumps existing while T1 cloud dumps do not. I did not check whether any plan restricts it.

### Gaps
- GitHub code search was unavailable: no `gh` CLI, no token, the code-search API needs authentication, `grep.app` returned 429, and Sourcegraph queries often timed out. A notebook-level `meas_level=0` or `RAW-CAPTURE` dump in a small repo could have been missed.
- Kaggle was not searched directly.

## Do control-stack projects ship real per-shot sample traces?

### Takeaway
No control-stack repository I inspected ships per-shot time-resolved qubit traces, but one QICK-based project publishes them on Zenodo as a T1 record of 1,000,000 shots. Fixtures elsewhere are integrated single shots (T3) or averaged traces (T4).

### Cited Findings

#### T1 and T2 from a control-stack project
- **QICK and hls4ml readout data (T1).**
  - Paper: Di Guglielmo, Du, et al., [arXiv:2501.14663](https://arxiv.org/abs/2501.14663) (24 Jan 2025), [IEEE Trans. Quantum Eng. 6, doi:10.1109/TQE.2025.3604712](https://api.crossref.org/works/10.1109/TQE.2025.3604712).
  - Data: [10.5281/zenodo.14427490](https://zenodo.org/records/14427490), creator Botao Du (Purdue), published 2024-12-12, CC BY 4.0, open access.
  - Evidence (read by me): `0528_X_train_0_770.npy` is `<f8` (900000, 1540); `0528_X_test_0_770.npy` is `<f8` (100000, 1540); `y_train` is (900000,); `y_test` is (100000,) with exactly 50,000 of label 0 and 50,000 of label 1. A 1 MB slice of `X_test` is integer-valued within about ±3,500. [generate_window.py](https://github.com/fastmachinelearning/ml-quantum-readout/blob/dev/data/qick_data/generate_window.py) asserts width equals `770*2` and slices by `*2`, so samples are I,Q pairs.
  - Platform: frequency-tunable multi-transmon device with one qubit read and others far-detuned; ZCU216 RFSoC with QICK firmware; ADCs at 6.88 GS/s, demodulated on the FPGA; author affiliations include Purdue and Fermilab. — [arXiv HTML](https://arxiv.org/html/2501.14663)
  - Shots: 500,000 per state, 1,000,000 total. Samples: 770 I/Q pairs. Readout: "the readout pulse length of 2.5 μs corresponds to 770 ADC clock cycles", about 3.25 ns per sample (the paper also says 3.22 ns in one place). The repo's [data/README.md](https://github.com/fastmachinelearning/ml-quantum-readout/blob/dev/data/README.md) describes an older version as 2000 ns at 2.6 ns with width 1440, which does not match this file.
  - Labels: prepared ground by cooling, excited by a π pulse with error ≤0.4%. Format and size: `.npy`, 12.33 GB. Code repo: [ml-quantum-readout](https://github.com/fastmachinelearning/ml-quantum-readout), no licence field per the GitHub API.
  - Reuse: LUNA calls it "the publicly available superconducting-qubit readout dataset (time-series I/Q traces, labeled |0⟩/|1⟩)" ([arXiv:2512.07808](https://arxiv.org/html/2512.07808)); a third-party archive says "The readout data are public" and uses it as D1 and D2 ([Zenodo 23777889](https://zenodo.org/records/23777889)).
- **Same repo, `data/fpga_testing/malab_{e,g}_state_A_fpga.csv` (T2).** The folder README calls these "Mock readout data", "recorded on the board". Read by me: 1540 columns, 5,000 rows each, 38 MB each, state in the filename. Whether a qubit was involved is not verified. — [folder README](https://github.com/fastmachinelearning/ml-quantum-readout/blob/dev/data/fpga_testing/README.md)
- **Same repo, older data links (T6).** "Quantum Machine" data (`00002_IQ_plot_raw.h5`, train (9000, 2000), test (1000, 2000), 1 ns per sample) and QICK version 20230529 are on Purdue SharePoint, which redirected to a Microsoft login when I tested. A Dropbox "Partitioned" link returned HTTP 200 but no file names, so its contents are not verified. — [data/README.md](https://github.com/fastmachinelearning/ml-quantum-readout/blob/dev/data/README.md)

#### T3 and T4 fixtures
- **PycQED (T3, some T4).** [DiCarloLab-Delft/PycQED_py3](https://github.com/DiCarloLab-Delft/PycQED_py3), MIT licence. `pycqed/tests/test_data` holds 520 files, 2.83 GB, dated 2016 to 2020 by path. Read by me: `20170716/134634_SSRO_QR` has `Experimental Data/Data` (24564, 2) with value `lin_trans w0` over sweep `Shots`; `20190916/184929_Mux_SSRO_['X','D4','D2','Z2']__device` has (131072, 5) with four UHFQC weight channels; `20170227/115118_acquisition_delay_scan_qubit` has (100, 3) of |S21| and angle versus delay, which is T4-like. File names contain no transient or time-trace experiments among the ones I listed. Labels not verified from the files.
- **Quantify (T3, T4).** [quantify-core](https://github.com/quantify-os/quantify-core), BSD 3-Clause, README says it "will henceforth be archived"; the same fixtures appear in [qblox-scheduler](https://gitlab.com/qblox/packages/software/qblox-scheduler). Read by me: `20230509/…ReadoutCalibration/dataset.hdf5` has `States` (1000,), `I`, `Q`; `20250522/…Readout Calibration` has 800 points; `20230927/…time_of_light_calibration` has 500 points of magnitude and phase. Real-hardware origin not verified, since no `snapshot.json` exists for those runs.
- **py-qua-tools (T3).** [tests/IQblobs.npy](https://github.com/qua-platform/py-qua-tools/blob/main/tests/IQblobs.npy) is (4, 10000) float64, loaded in [test_analysis.py](https://github.com/qua-platform/py-qua-tools/blob/main/tests/test_analysis.py) as `Ig, Ie, Qg, Qe`. Hardware origin not verified. `tests/calib_res.pkl` not opened.
- **qua-libs (T4).** [intro-to-QUAlibrate `#1_time_of_flight_191124/arrays.npz`](https://github.com/qua-platform/qua-libs/tree/main/Tutorials/intro-to-QUAlibrate), created 2025-06-30, node settings `simulate: false`, `num_shots: 10`. Read by me: `raw_data.adcI1` and `adcQ1` (5000,) averages, plus `adc_single_runI1` and `Q1` (5000,) one last-shot trace. No prepared-state label, and whether a qubit or loopback was used is not verified. The standard [raw ADC script](https://github.com/qua-platform/qua-libs/blob/main/Quantum-Control-Applications/Superconducting/Single-Fixed-Transmon/02_raw_adc_traces.py) saves `.average()` and "only last run".
- **iq_readout (T3).** [docs/…/data_two_state_calibration.npy](https://github.com/MarcSerraPeralta/iq_readout) is (2, 50000, 2) float64, MIT licence. Hardware origin not verified.

#### Capability and negatives by project
- **QICK.** `acquire_decimated` returns "decimated values, averaged over rounds", with `(n_reps*n_reads, length, 2)` for multi-rep programs, so per-rep traces are supported in the API. — [qick_asm.py](https://github.com/openquantumhardware/qick/blob/main/qick_lib/qick/qick_asm.py). `qick_demos/06_qubit_demos.ipynb` says "The measurements in this notebook were used for the QICK paper" and holds 10 PNG plots, no arrays. — [notebook](https://github.com/openquantumhardware/qick/blob/main/qick_demos/06_qubit_demos.ipynb). Other data-like files are firmware testbench files, a hardware zip and a near-empty `sds_trace_data.dat`.
- **QubiC.** I listed 16 GitLab projects by file extension and found nothing named or shaped like per-shot traces. Of the 61 `.npz` in `QubicCali2021a` I opened one pair: `X4Y1/Q0_gmix.npz` holds Gaussian-mixture parameters and `Q0_rcal.npz` a 2×2 matrix (read by me). `software/test/toolchain/test_outputs/*.npz` and `qubic_emulator/test/test_suite/npz_files.zip` are test outputs by their paths; `qubicai/backend/ClassificationManager.pkl` not opened. — [LBL-QubiC group](https://gitlab.com/LBL-QubiC)
- **QubiC paper (T6).** Jung et al. use QubiC 2.0 on eight fixed-frequency transmons, about 25,000 shots per state (75,000 per qubit), 1500 to 2000 timesteps at 2 GS/s, 750 to 1000 ns, labelled by prep pulse. The arXiv HTML has no availability statement. — [arXiv:2601.19057](https://arxiv.org/html/2601.19057)
- **Quantum Machines.** `qualibration-libs` has no data-like files and `qualibrate` is a one-file stub. — [qua-platform](https://github.com/qua-platform)
- **Qblox.** `qblox_instruments` (141 files) and `qblox-applications` (56 files) have no data files. `Trace` "measures a signal s(t)", its default `bin_mode` is `BinMode.AVERAGE`, and possible bin modes "may depend on the control hardware". — [acquisition_library.py](https://gitlab.com/qblox/packages/software/qblox-scheduler/-/blob/main/src/qblox_scheduler/operations/acquisition_library.py). The scope buffer is 16384 points (search extract). — [Quantify tutorial](https://quantify-quantify-scheduler.readthedocs-hosted.com/tutorials/ScheduleGettable.html)
- **Zurich Instruments.** `zhinst/laboneq` has 836 files and no data-like files. "`RAW`: Returns raw data after the ADC, up to 4096 samples. Only a single raw acquire event within an averaging loop per experiment is allowed." — [acquisition_type.py](https://github.com/zhinst/laboneq/blob/main/src/python/laboneq/core/types/enums/acquisition_type.py). Docs list single-shot averaging separately but do not say RAW may combine with it, and the how-to uses cyclic (search extract). — [LabOne Q averaging](https://docs.zhinst.com/labone_q_user_manual/core/functionality_and_concepts/03_sections_pulses/concepts/04_averaging_sweeping.html); [raw data how-to](https://docs.zhinst.com/labone_q_user_manual/applications_library/how-to-guides/sources/01_superconducting_qubits/02_pulse_sequences/01_tuneup/00_readout_raw_data.html)
- **Presto (Intermodulation Products).** [presto-measure](https://github.com/intermod-pro/presto-measure) (76 files) and [presto-demo](https://github.com/intermod-pro/presto-demo) (24 files) have no data-like files. `bak/readout_singleshot.py` runs with `num_averages = 10_000`, reads `get_store_data()` and uses `pls.match` for single shots. API docs were not located, so whether `store` can return per-shot traces is not verified. Anritsu: one search found no quantum control link.
- **Qibo.** `qibolab` declares `AcquisitionType.RAW` ("Return the waveform, demodulated to baseband") and `AveragingMode.SINGLESHOT`, giving shape `(nshots, samples, 2)`, and has no data-like files. — [execution_parameters.py](https://github.com/qiboteam/qibolab/blob/main/src/qibolab/_core/execution_parameters.py). `qibocal` fixtures are spectroscopy, flux, Rabi, Ramsey, DRAG and ZZ only. — [qibocal tests](https://github.com/qiboteam/qibocal/tree/main/tests). `qibolab_platforms_qrc` has none.
- **Labber.** The org page lists only [Drivers](https://github.com/Labber-software/Drivers), whose only data-like files are two RB pickle tables. A third-party repo holds Labber logs, but `Q1Q2Q3SSROTTgefhReadout_25.hdf5` has `Data/Data` (84, 8, 4) and its CSVs are histograms. — [muxueqianxun/SuperconductingQubitReadout](https://github.com/muxueqianxun/SuperconductingQubitReadout)
- **qiskit-experiments.** `test/` non-code fixtures are quantum-volume JSON only. `MockIQBackend` defaults to `meas_level=KERNELED`, `meas_return="single"` and samples Gaussian clusters. — [mock_iq_backend.py](https://github.com/qiskit-community/qiskit-experiments/blob/main/qiskit_experiments/test/mock_iq_backend.py)
- **Other negatives.** No data-like trace files by extension in [qililab](https://github.com/qilimanjaro-tech/qililab), [qubex](https://github.com/amachino/qubex), [iqm-client](https://github.com/iqm-finland/iqm-client), [QAT](https://github.com/oqc-community/qat), [pyQuil](https://github.com/rigetti/pyquil), [Braket examples](https://github.com/amazon-braket/amazon-braket-examples), [LUNA](https://github.com/advent-lab/LUNA) or the [trapped-ion FNN repo](https://github.com/quantumiracle/On_board_FNN_qubit_discrimination) for arXiv:1810.07997. A UW-Madison course repo says its QOLab/Qualibrate data "are not public" (T6). — [Phys763.HW3](https://github.com/Chumita003/Phys763.HW3-ReadoutAnalysis). [GaifutdinovDamir/Qubits-readout](https://github.com/GaifutdinovDamir/Qubits-readout) has a 3 MB CSV of what looks like I, Q, label triples, with no README and unknown origin.

#### Other public T1 and T3 records met on the way (outside vendor repos)
- **Thermal detector readout (T1, small).** Gunyhó et al., [arXiv:2303.03668](https://arxiv.org/abs/2303.03668), Nature Electronics 2024, doi:10.1038/s41928-024-01147-7 (doi as given by a [third-party README](https://zenodo.org/records/23777889)). Data: [10.5281/zenodo.7773981](https://zenodo.org/records/7773981) (2023-03-27) and later [10000511](https://zenodo.org/records/10000511) (2023-10-13), CC BY 4.0, about 873 to 923 MB zip. Read by me in `bolometer-characterization-timetrace-76.nc`: `readout_meta_CH0_re` and `_im` (2, 1000, 781), time axis 781 points at 64 ns over 49.92 µs, `qubit_pi_pulse_amplitude` [0, 1], so labels are derivable and there are 1,000 shots per setting. Qubit count and lab not verified; other `.nc` files not opened. This is bolometric rather than standard dispersive amplifier readout.
- **RED (T3).** Zheng et al., [10.5281/zenodo.23367046](https://zenodo.org/records/23367046), 2026-10-14, CC BY 4.0, 1.02 GB. Read by me in `transmon_03664_I_g_6p596us.h5`: `iq` (8000, 51) complex64, calibration (3000,) for g, e and f; repeated single-shot readouts with no within-readout time axis. Fluxonium and transmon.
- **Adaptive thresholding (T3).** Cao, Akhouri, Vizvary, Ofek, Dai, Murch, [10.5281/zenodo.22719098](https://zenodo.org/records/22719098), 2026-09-12, CC BY 4.0, 1.31 GB. Read by me in one 19 MB file: `I0_1` to `I0_5`, `I1_final` and similar, each (100000,) float64.
- **300-ion fluorescence frames (T3-like, images).** [anonymousshare/300-Qubits-Register-Dataset](https://huggingface.co/datasets/anonymousshare/300-Qubits-Register-Dataset): 27,000 PNG frames of 456×88 pixels with 300 annotated sites and manual labels, 740 MB, one image per shot with no time axis. Paper, lab and hardware origin not verified. Related code: [Site-AIT](https://github.com/JUNGJEIHUANG/Trapped-Ion-Qubits-readout).

#### Tier roll-up of the records above
- **T1:** [Zenodo 14427490](https://zenodo.org/records/14427490) (QICK transmon); [Zenodo 7773981](https://zenodo.org/records/7773981), run 76 (bolometer).
- **T2:** [`malab_{e,g}_state_A_fpga.csv`](https://github.com/fastmachinelearning/ml-quantum-readout/blob/dev/data/fpga_testing/README.md).
- **T3:** [FCM IBM data](https://github.com/Quasar-UniNA/FCM-based-QubitDiscriminator); [Indahl IBM IQ runs](https://zenodo.org/records/23812756); [Pittsburgh IQ data](https://doi.org/10.6084/m9.figshare.32539992) (inferred, pickles unopened); [PycQED SSRO files](https://github.com/DiCarloLab-Delft/PycQED_py3); [Quantify ReadoutCalibration](https://github.com/quantify-os/quantify-core); [py-qua-tools IQblobs](https://github.com/qua-platform/py-qua-tools); [iq_readout](https://github.com/MarcSerraPeralta/iq_readout); [RED](https://zenodo.org/records/23367046); [adaptive thresholding](https://zenodo.org/records/22719098); [300-ion frames](https://huggingface.co/datasets/anonymousshare/300-Qubits-Register-Dataset). Hardware origin is not verified for Quantify, py-qua-tools, iq_readout and the 300-ion frames.
- **T4:** [qua-libs time-of-flight](https://github.com/qua-platform/qua-libs/tree/main/Tutorials/intro-to-QUAlibrate); [Quantify time_of_light](https://github.com/quantify-os/quantify-core); [PycQED acquisition delay scan](https://github.com/DiCarloLab-Delft/PycQED_py3).
- **T5:** [Quarton simulation](https://zenodo.org/records/12212723) (see the simulator section).
- **T6:** [Purdue SharePoint and Dropbox versions](https://github.com/fastmachinelearning/ml-quantum-readout/blob/dev/data/README.md); [QubiC 2.0 traces](https://arxiv.org/html/2601.19057); [HERQULES IBM Manila data](https://github.com/satvikmaurya/HERQULES); [Quiroga](https://ar5iv.labs.arxiv.org/html/2112.00313); [Azad and Zhang](https://ar5iv.labs.arxiv.org/html/2210.08574); [Phys763 course data](https://github.com/Chumita003/Phys763.HW3-ReadoutAnalysis); [QDataSet](https://github.com/eperrier/QDataSet) (dead link, and simulated, not traces).

### Inferences
- The QICK record is the closest public analogue to the Lienhard data format, but it is single-qubit. It cannot substitute for a frequency-multiplexed five-qubit benchmark, though it can support single-qubit discriminator work.
- Most stacks can capture per-shot or averaged traces, but the projects treat traces as a transient calibration artefact and do not archive them.

### Gaps
- Whether the T3/T4 fixtures in Quantify, py-qua-tools and iq_readout come from real devices is not verified.
- Whether per-shot RAW is permitted on Zurich Instruments or Qblox hardware is not verified from a primary statement.
- Leads surfaced but not opened, likely covered by the academic-dataset researchers: Zenodo [8004359](https://zenodo.org/records/8004359), [21671989](https://zenodo.org/records/21671989), [13368027](https://zenodo.org/records/13368027), [6344913](https://zenodo.org/records/6344913), [14033026](https://zenodo.org/records/14033026), [20825189](https://zenodo.org/records/20825189), [23913632](https://zenodo.org/records/23913632); figshare [34803777](https://doi.org/10.6084/m9.figshare.34803777.v1); Dryad [1c59zw4b2](https://doi.org/10.5061/dryad.1c59zw4b2).

## Which simulated trace datasets and readout simulators exist?

### Takeaway
I found no downloadable simulated dataset of realistic dispersive readout traces, and QDataSet is not one. QuTiP and dynamiqs can generate per-trajectory measurement records but ship no readout model, so a fallback means writing the model or using small unreviewed code repositories.

### Cited Findings

#### Datasets
- **QDataSet (T6, and not readout traces).** Perrier, Youssry and Ferrie, [arXiv:2108.06661](https://arxiv.org/abs/2108.06661). 52 datasets on one- and two-qubit simulations, 10,000 samples each, zipped pickles, about 14 TB; measurement outputs are "Pauli measurement distributions", averaged over noise realisations; no measurement record, dispersive readout or I/Q time series is mentioned. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2108.06661). The repo points only to a [Cloudstor link](https://cloudstor.aarnet.edu.au/plus/s/rxYKXBS7Tq0kB8o), which failed to connect when I tested. — [README](https://github.com/eperrier/QDataSet). AARNet shut Cloudstor down on 15 December 2023 (search extract). — [AARNet](https://support.aarnet.edu.au/hc/en-us/articles/5697089309711). Zenodo holds only the 205 kB source zip. — [10.5281/zenodo.5202814](https://zenodo.org/records/5202814)
- **Quarton coupler simulation (T5).** Ye et al., [10.5281/zenodo.12212723](https://zenodo.org/records/12212723), 2024-06-21, CC BY 4.0: `Readout_Simulation.ipynb` plus four `integrated_signal_{0,1}_{I,Q}.npy`, of which I read one as (8, 12800) float64. Axis meaning not verified.
- **SME trajectory data (not traces).** Yamamoto and Tokura, [10.5281/zenodo.14252485](https://zenodo.org/records/14252485), 9.9 GB, simulate a stochastic master equation but store correlation functions and Fano factors.
- **Hugging Face and Zenodo searches** for simulated readout traces found no other dataset. — [Hugging Face datasets](https://huggingface.co/datasets)

#### General solvers
- **qiskit-dynamics 0.6.0 (page dated 2025/10/07).** `DynamicsBackend`: "Supported values are `1` and `2`" for `meas_level`, with `iq_centers` and `iq_width` as the IQ distribution around centres. No level 0 or time-resolved output is documented, and the page does not describe a time-resolved measurement model. — [DynamicsBackend](https://qiskit-community.github.io/qiskit-dynamics/stubs/qiskit_dynamics.backend.DynamicsBackend.html)
- **QuTiP.** `ssesolve` and `smesolve` support homodyne and `heterodyne=True`; `{"store_measurement": True}` gives `result.measurement`, and individual records are available via `measurements[0]`. No example models dispersive qubit readout. — [QuTiP stochastic guide](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-stochastic.html). Tutorials include `015_smesolve-heterodyne`, `017_smesolve-jc-photocurrent` and `Lecture-10-cQED-dispersive-regime`, but none is a readout chain by title. — [qutip-tutorials](https://github.com/qutip/qutip-tutorials)
- **dynamiqs.** `dsmesolve` solves the diffusive SME and returns `result.measurements` with shape `(..., ntrajs, nLm, nsave-1)`, time-averaged per save interval, one record per trajectory. — [dynamiqs dsmesolve](https://www.dynamiqs.org/stable/python_api/integrators/dsmesolve.html)
- **Quanlse (Baidu).** `readoutSim3Q` simulates readout-cavity calibration from a dispersive Jaynes-Cummings Hamiltonian, and the tutorial plots spectra of signal versus drive frequency. No per-shot traces in the tutorial. — [tutorial](https://github.com/baidu/Quanlse/blob/master/Tutorial/EN/tutorial-readout-cavity-calibration.ipynb)

#### Vendor emulators
- **Qibolab emulator.** Solves a master equation via QuTiP or dynamiqs and samples shots from state probabilities (`shots(probabilities, nshots)`); `results.py` branches on `AcquisitionType.INTEGRATION` and has no `RAW` case, so it does not emit waveforms. — [emulator docs](https://github.com/qiboteam/qibolab/blob/main/doc/source/main-documentation/emulator.rst); [results.py](https://github.com/qiboteam/qibolab/blob/main/src/qibolab/_core/instruments/emulator/results.py)
- **QubiC emulator.** Clock-accurate pulse emulation with "Options for simulating the effect on the QPU for educational purposes, such as the readout resonator response". — [qubic_emulator](https://gitlab.com/LBL-QubiC/qubic_emulator)
- **QICK `axis_kidsim_v3`.** "KID / Qubit Resonator Emulator" firmware block. — [kidsim.md](https://github.com/openquantumhardware/qick/blob/main/docs/source/firmware/readouts/kidsim.md)
- **qua-qsim.** README warns it "has been deprecated because it relies on Qiskit Pulse". — [qua-qsim](https://github.com/qua-platform/qua-qsim)

#### Small third-party code repositories (no shipped data, unreviewed)
- [samuelnoger/qubit-readout-ML](https://github.com/samuelnoger/qubit-readout-ML): QuTiP `mcsolve` T1 jump trajectories, cavity response, white Gaussian noise, leakage and cross-dispersive shifts for one or two qubits; generates raw time-series records for a 1D CNN. The closest to the need.
- [buildgreat-bot/QuantumReadoutSimulator](https://github.com/buildgreat-bot/QuantumReadoutSimulator): C simulator of six stages from resonator exit to ADC quantisation.
- [elifcuce/ECE456_2026_QubitReadoutSimulation](https://github.com/elifcuce/ECE456_2026_QubitReadoutSimulation): course notebook with quantum trajectories and amplification chain.
- [alexanderosherov/transmon_qubit_readout_simulation](https://github.com/alexanderosherov/transmon_qubit_readout_simulation): TUM/WMI thesis on QND readout optimisation, with Floquet analysis data.
- [fischermoseley/readout_classifier](https://github.com/fischermoseley/readout_classifier): classifier repo pointing to a simulation notebook, 0.4 MB.

### Inferences
- None of the named tools produces realistic dispersive traces out of the box. QuTiP or dynamiqs SME with a dispersive Jaynes-Cummings model is the defensible route for a paper, with a noise chain added by hand.
- Simulated traces cannot replace measured ones for claims about leakage or crosstalk unless the model includes those effects, which `samuelnoger/qubit-readout-ML` does only as added terms.

### Gaps
- "Quantum-Tailored Machine-Learning Characterization of a Superconducting Qubit" trains a recurrent network on synthetic time series (search extract); whether its code or data is public was not checked. — [ResearchGate listing](https://www.researchgate.net/publication/357194405_Quantum-Tailored_Machine-Learning_Characterization_of_a_Superconducting_Qubit)
- I did not run any simulator or assess fidelity against measured traces.

## Do the two draft claims hold?

### Takeaway
"Commercial cloud providers do not release raw readout traces publicly" holds only in the narrow sense that I found no provider-released dataset, and it hides user-capture routes. "No other raw-trace dataset is publicly available" is false: at least two real-hardware per-shot time-resolved records are downloadable now.

### Cited Findings
- Exact wording in the arXiv version: "Since commercial cloud providers do not release raw readout traces publicly" (§3.3) and "(no other raw-trace dataset is publicly available)" (Table 1). — [arXiv:2608.01939v1](https://arxiv.org/html/2608.01939v1)
- Provider released a dataset: none found. The provider repositories I listed (qiskit-ibm-runtime, qiskit-experiments, pyQuil, qcs-paper, QAT, iqm-client, Braket examples) hold no trace dump, and the dataset searches above found none. I did not enumerate those organisations in full. — [rigetti/qcs-paper](https://github.com/rigetti/qcs-paper)
- Provider lets users capture their own: Rigetti documents per-shot raw ADC. — [pyQuil raw capture docs](https://pyquil-docs.rigetti.com/en/stable/quilt_raw_capture.html). IQM exposes `TimeTrace`, per-shot not verified. — [iqm-pulse](https://docs.iqm.tech/iqm-pulse/api/iqm.pulse.playlist.instructions.html)
- Route withdrawn: IBM `backend.run` (15 Oct 2024) and pulse control (3 Feb 2025). — [IBM changelog](https://quantum.cloud.ibm.com/docs/en/guides/latest-updates). OQC Lucy on Braket (June 2024, secondary source). — [SDxCentral](https://www.sdxcentral.com/news/rigettis-84-qubit-ankaa-2-quantum-computer-available-through-aws-braket-service/)
- Counter-examples to the second claim: [Zenodo 14427490](https://zenodo.org/records/14427490) (QICK, 1,000,000 shots, 770 I/Q samples, CC BY 4.0) and [Zenodo 7773981](https://zenodo.org/records/7773981) (bolometer, (2, 1000, 781) in run 76, CC BY 4.0). A third party used both and states "The readout data are public". — [Avcibas, Zenodo 23777889](https://zenodo.org/records/23777889)
- The repo says of the five-qubit data: "The dataset is **not public and is not redistributed here**." — [Oraqle README](https://github.com/manosgior/Oraqle)

### Inferences
- The phrase "no other raw-trace dataset" implies that the Lienhard dataset is itself public, which contradicts the repo README. The sentence needs fixing for that reason as well.
- Suggested wording for claim 1: "We are not aware of a commercial provider that publishes a dataset of time-resolved single-shot traces. IBM now returns at most integrated I/Q points and removed pulse-level access in February 2025; Braket's `capture_v0` returns a bit; Rigetti's Quil-T `RAW-CAPTURE` documents per-shot ADC capture with per-shot and per-job sample limits."
- Suggested wording for claim 2: "Public per-shot time-resolved records exist, for example a single-transmon QICK dataset [Du 2024] and a bolometric readout dataset [Gunyhó 2023], but we found no public multi-qubit, frequency-multiplexed trace dataset of comparable size." The last clause is scoped to these notes and should be checked against the academic-dataset notes before use.

### Gaps
- I did not search every academic repository; other notes may add more counter-examples.

## Public re-hosts of the Lienhard dataset

### Takeaway
None found. Everything I saw treats the data as obtained from the authors on request.

### Cited Findings
- Oraqle's dataset note: "no Zenodo/figshare/GitHub release exists", and "HERQULES and KLiNQ obtained it from the authors as well". — [dataset/README.md](https://github.com/manosgior/Oraqle/blob/main/dataset/README.md)
- HERQULES says it uses "the dataset of a custom five-qubit chip provided by the authors"; its repo has four files and says "Feel free to shoot me an email for dataset access." — [arXiv:2212.03895](https://ar5iv.labs.arxiv.org/html/2212.03895); [satvikmaurya/HERQULES](https://github.com/satvikmaurya/HERQULES)
- Two 2026 papers using the data give no availability statement. — [arXiv:2610.02129](https://arxiv.org/html/2610.02129); [arXiv:2607.11668](https://arxiv.org/html/2607.11668)
- Searches: Zenodo `+qubit +readout +"Lienhard"` and `+"five-qubit" +readout +multiplexed` returned 0; DataCite `Lienhard readout` returned 0; GitHub `herqules readout` returned only the 28 kB code repo; `KLiNQ qubit` returned 0. — [Zenodo search](https://zenodo.org/search); [DataCite API](https://api.datacite.org/dois)

### Inferences
- The tier for this dataset as things stand is T6 (on request), which is why the draft's phrasing needs care.

### Gaps
- Without GitHub code search, a re-host in a private-looking or poorly described repository cannot be excluded. Searching for the file names `DRaw_C_Tr_v0-001` and `DRaw_C_Te_v0-002` with an authenticated GitHub account would be a cheap further check.
