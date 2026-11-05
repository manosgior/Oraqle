# results/external: numbers §7 takes from elsewhere

This folder covers every number in §7 (`system_bottleneck.tex`) and `system/system_cost.py` that was not produced by this repository. For each number it records the source, the citation or commit, and how it was checked.

The tex line numbers refer to oraqle-ics `chapters/system_bottleneck.tex` (working tree of 2026-11-05, with the [audit fix] edits). The code line numbers refer to `oraqle-ics/sims/system_cost.py` @ `d7cd4e9`, which is byte-identical to `system/system_cost.py` apart from path plumbing.

## Files

| File | What |
|---|---|
| `qshell/figure8_ler.json`, `figure8_ler_README.md` | qShell Helios d=9 SI1000 sweep: native RTL occupancy histograms per p. Verbatim, qShell @ `f53859d` |
| `qshell/figure8_core_ler.json`, `figure8_core_ler_assumptions.md` | The same shots, Helios kernel counter only. Verbatim, qShell @ `f53859d` |
| `qshell/capacity_inputs.json` | V80 resource model: LUT budget, shell, per-application footprints. Verbatim, qShell @ `f53859d` |
| `qshell_summary.json` | Output of `extract_qshell.py`: the qShell values `system_cost.py` uses, recomputed and checked |
| `cxl_quantum/requirements_tab_req-decoders.tex` | CXL-Quantum Table `tab:req-decoders` (lines 1-25 of `sections/requirements.tex` @ `2ac2adf`) |
| `cxl_quantum/workloads.py` | CXL-Quantum `plots/workloads.py` @ `2ac2adf`. It is the origin of `system/workloads.py`; reference only, not imported |
| `cxl_quantum_decoders.json` | Decoder latencies transcribed from CXL-Quantum, with their upstream and the workloads diff |
| `helios_paper.json` | Helios paper Table I LUTs and the Fig. 7c weighted/unweighted ratio, checked against the paper |
| `literature_and_presets.json` | Acharya 2025 Λ, Ross-Selinger and Litinski workload model, Micro Blossom streaming, CUDA-Q GPU, `ibm_boston` preset |
| `hand_set_constants.json` | Design constants that are *not* external data (`FRAME_NS`, `BUDGET`, CNN latency, `GATES_NS`) |
| `uniform_fraction_d9.json` | Output of `system/derive_uniform_fraction.py`, the recovered producer of `UNIFORM_FRACTION` |
| `COMPARISON_qshell_vs_sec7.md` | qShell vs. §7 decoder numbers, the meaning of the qShell 1680 ns, and an SI1000-vs-uniform sensitivity check |
| `extract_qshell.py` | Recomputes `QSHELL_P99_NS`, `LUT_D9` and the V80 budget/shell from the verbatim qShell files. Standard library only; exit 1 on mismatch |
| `si1000_sensitivity.py` | Sensitivity check, not used by the paper. Needs stim |

## Run

```bash
python results/external/extract_qshell.py                      # any python3; checks the qShell constants
python system/derive_uniform_fraction.py                       # needs stim 1.16.0; checks UNIFORM_FRACTION
docker run --rm -v $PWD:/repo:ro -e ORAQLE_RESULTS=/repo/results -w /repo oraqle-latsim \
    python system/derive_uniform_fraction.py --no-write        # the same check in the oraqle-latsim image
docker run --rm -v $PWD:/repo:ro -w /repo oraqle-latsim python results/external/si1000_sensitivity.py
```

## External numbers

The last column says how each number was checked:

- **primary**: compared with the original source, available locally.
- **upstream data**: recomputed from the raw data files of the source.
- **transcription only**: only the copy chain (paper ← CXL-Quantum table ← cited source) could be checked.

| Value | Where used | Source | How checked |
|---|---|---|---|
| Helios d=9 p99 **308 ns** at p=1e-3 (median 184 ns) | `HELIOS_P99_NS` (`system_cost.py:66`); sys:14; tab:sys_reaction | CXL-Quantum tab:req-decoders "decode stage" ← qShell `figure8_ler.json` (Helios RTL, SI1000, 50,000 shots) | **upstream data**: equals the qShell kernel-counter p99/median (296/172 ns) + 12 ns |
| Helios p99 vs p: **1680/1720/1804/1980/2132/2360/2592 ns** at p = 1e-4…1e-2 | `QSHELL_P`, `QSHELL_P99_NS` (`system_cost.py:67-68`) | qShell `figure8_ler.json` | **upstream data**: nearest-rank p99 of each histogram, exact (`extract_qshell.py`) |
| Helios whole shot **1.51 µs, 168 ns/round** at d=9 | sys:14 until the 2026-11-05 [audit fix] removed "168 ns per round" from the paper; docstring only | CXL-Quantum tab:req-decoders ← qShell XSim 1516/1552 ns (`appendix.tex:265-266`) | **upstream data** (qShell text); not used numerically. See the wording issue in `COMPARISON_qshell_vs_sec7.md` |
| Micro Blossom d=9 **772 ns** (86 ns/round) | `DECODERS["Micro Blossom"]` (`system_cost.py:72`); sys:14; tab:sys_reaction | CXL-Quantum tab:req-decoders, "Ours: cycle-accurate RTL" (added in CXL commit 3307a11) | **transcription only**. The bench that produced it was not found anywhere; qShell's Micro Blossom d9 latency (240/480 ns) is marked illustrative |
| GPU **~2.1 µs** at d=13 | `DECODERS["GPU"]` (`system_cost.py:75`); sys:14; tab:sys_reaction | Nvidia2026Cudaq (CUDA-Q QEC 0.6 blog) via CXL-Quantum | **transcription only** |
| Micro Blossom streaming 0.37 µs | `system_cost.py` docstring only | Wu2025Micro via CXL-Quantum | **transcription only**; not used |
| Helios weighted **×38/15 = 2.53** | `HELIOS_WEIGHTED` (`system_cost.py:71`); sys:14, :68, :188 | Liyanage2023Scalable, Fig. 7c / Sec. V text (15 → 38 ns per round, d=13, w_max=16) | **primary**: `CXL-Quantum/helios.txt:1496-1500` (arXiv 2301.08419v2) |
| Helios Table I LUTs **970/6425/52111/165718/448314/898715**, d=3…21 → K_LUT = 3.39 | `HELIOS_TABLE1`, `K_LUT` (`system_cost.py:78-80`); sys:14 "d^3.4", sys:127 | Liyanage2023Scalable, Table I (VCU129) | **primary**: `CXL-Quantum/helios.txt:1296-1338` |
| `LUT_D9`: Helios **116,575**, Micro Blossom **155,399** | `system_cost.py:77`; sys:14, sys:127 (incl. "25 % smaller") | qShell `capacity_inputs.json` (routed hierarchy, V80) | **upstream data**: exact |
| V80 budget **2,574,208** LUT, shell **63,989** LUT | `V80_LUT` (`system_cost.py:81`) | qShell `capacity_inputs.json` (budget, common_static) | **upstream data**: exact |
| `UNIFORM_FRACTION` **0.0166/0.0481/0.0774/0.1428/0.2448** | `system_cost.py:69-70` → Helios p_equiv; sys:85 "p≈0.7-1.2 %" | Stim rotated memory-Z d=9 with uniform noise (inline run, transcript 94019386 line 1039, 2026-10-26) | **reproduced exactly** by `system/derive_uniform_fraction.py` (stim 1.16.0) |
| Λ = **2.14** on hardware | sys:9 | Acharya2025Quantum (Nature 638, 2025) | **transcription only**: no local copy |
| Rotations to **1e-6**, **3 log2(1/ε) = 60** T gates each | `system/workloads.py:18-19`; sys:16 | Ross2016Optimal | **transcription only**; inherited unchanged from CXL-Quantum |
| Workload model (feed-forward steps, magic-state model) | `system/workloads.py`; sys:16 | Litinski2019Game; CXL-Quantum `plots/workloads.py` | Model citation. **Code diff**: the functions are identical to CXL-Quantum's; Adder-1024 and Ising-100 were added |
| `ibm_boston`: CZ **1.191e-3**, SX **1.637e-4**, T1/T2 **284.95/322.68 µs**, readout median 3.54e-3, SX/CZ 32/68 ns | `system/sweep_ler.py` `PRESETS`; ECCentric drivers; qec_impact:21 footnote | IBM Quantum dashboard snapshot of ibm_boston (eccentric `run_ler_noise_models.py:69-72`) | **not checkable**: the snapshot was not archived. The values agree across all code copies |
| `FRAME_NS` = 100 ns | `system_cost.py:63`; tab:sys_reaction caption | Hand-set allowance, the same 0.1 µs as CXL-Quantum's "decision allowance" | Design choice (`hand_set_constants.json`) |
