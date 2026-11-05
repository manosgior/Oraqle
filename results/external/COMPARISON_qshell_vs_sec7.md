# qShell vs. §7: decoder numbers

Question: qShell has decoder numbers. Do they match what §7 (oraqle-ics `chapters/system_bottleneck.tex`, `sims/system_cost.py`) uses?

Sources:

- **qShell**: Overleaf git `~/Documents/GitHub/qshell` @ `f53859d` (2026-09-09). The figure JSONs are copied verbatim into `qshell/`.
- **CXL-Quantum**: Overleaf git `~/Documents/GitHub/CXL-Quantum` @ `2ac2adf` (2026-10-28). Its decoder table is copied into `cxl_quantum/`.
- **§7 code**: `oraqle-ics/sims/system_cost.py` @ `d7cd4e9`. The repo copy is `system/system_cost.py`.
- **§7 text**: `oraqle-ics/chapters/system_bottleneck.tex`, referenced as `sys:N`, working tree of 2026-11-05.

How the numbers were checked:

- `python results/external/extract_qshell.py` recomputes every qShell-derived constant from the histograms. It exits 0: all values match.
- `python results/external/si1000_sensitivity.py` runs the noise-model sensitivity check described in section 3. It needs stim.

## 1. Table

| Quantity | §7 value (file:line) | qShell value (file:line) | CXL-Quantum value | Match? | Note |
|---|---|---|---|---|---|
| Helios d=9 p99 at p=1e-3 | 308 ns (`system_cost.py:66` `HELIOS_P99_NS`; sys:13; tab:sys_reaction sys:73) | Kernel counter p99 **296 ns** (`figure8_core_ler.json` top[p=0.001, Static].offered_native_service). Native occupancy p99 **1804 ns** (`figure8_ler.json` evidence["0.001"]) | 308 ns, "decode stage" (`requirements.tex:13`) | yes, constant +12 ns | CXL's "last syndrome to first result" is the qShell kernel counter plus 3 cycles. `fig_budget.py:14-16` in CXL names `figure8_ler.json` as its source. |
| Helios d=9 median at p=1e-3 | 184 ns (`system_cost.py:13`, docstring only) | Kernel 172 ns; native 1680 ns | 184 ns | yes, +12 ns | Not used in any computation. |
| Helios d=9 slowest shot at p=1e-3 | not used | Kernel 628 ns; native 2136 ns | 0.64 µs, "3.5× median" (`requirements.tex:50`) | yes, +12 ns | |
| Helios p99 vs p (`QSHELL_P`, `QSHELL_P99_NS`) | 1680/1720/1804/1980/2132/2360/2592 ns at p = 1e-4…1e-2 (`system_cost.py:67-68`) | Nearest-rank p99 of `figure8_ler.json` evidence[p].native_service_histogram_ns: identical | n/a | **exact** | §7 uses only the growth over p=1e-3; see section 2. |
| Helios p99 used per readout length | 968 / 977 / 1006 / 1142 ns at 1000/800/600/400 ns (`system_cost.json` helios; sys:80 "p99 reaches ~1 µs") | Kernel p99 at the same p, plus 12 ns | n/a | consistent | "~1 µs" holds: the qShell kernel p99 at p=1e-2 is 1084 ns. |
| Helios d=9 whole-shot latency | "168 ns per round" (sys:13; `system_cost.py:14` docstring) | XSim 1516 / 1552 ns for zero / hook input, native interface included (`sections/appendix.tex:265-266`). "1508 → 1552 ns" with adapter (`sections/evaluation.tex:162`). Figure-8 native minimum 1568 ns | 1.51 µs, 168 ns/round (`requirements.tex:12`) | yes (1.51 µs / 9 = 168 ns) | **Wording issue in sys:13**: the text pairs "308 ns at p99" with "168 ns per round". But 168 ns is the *whole-shot* path including the ~1.5 µs native serialization; the 308 ns decode stage is 34 ns per round. The 168 ns is not used numerically. |
| Helios kernel-internal / QEC-path latency | not used | "Kernel-internal XX ns median, XX ns p99" (TODO, `appendix.tex:244-245`). FPGA-internal QEC path 120 / 150 ns at **d=21**, without / with wrapper (`appendix.tex:246-248`). "Tens of ns median, ~100 ns tail" (`appendix.tex:465`) | n/a | n/a | These are different measurement points (d=21 streaming path, unfinished text) and are not comparable to the d=9 batch numbers. |
| Micro Blossom d=9 latency | **772 ns** (`system_cost.py:72`; sys:13; tab:sys_reaction 0.95 µs = 74 + 772 + 100) | **240 / 480 ns** for zero / nonzero syndrome, MicroBlossom d9 V80 (`figures/plots.py:428-435`, last entry). The caption marks these as **illustrative**, not measured (`evaluation.tex:154-156`) | 772 ns, 86 ns/round, "Ours: cycle-accurate RTL" (`requirements.tex:14`; added in CXL commit 3307a11, 2026-10-07) | **MISMATCH**, but qShell's value is illustrative | No bench output, log or script producing 772 ns exists in CXL-Quantum, qShell, oraqle-ics or the cluster. 3307a11 calls it "decoding a whole d=9 shot in batch". |
| Micro Blossom streaming | 0.37 µs (docstring only) | none | 0.37 µs, reported (Wu2025Micro) | n/a | Not used. |
| GPU decode | 2.1 µs at d=13 (`system_cost.py:75`; sys:13) | none | ~2.1 µs, reported (CUDA-Q QEC 0.6) | yes (CXL) | No qShell number. |
| Helios weighted (w_max=16) | ×38/15 = 2.53 (`system_cost.py:71`; sys:13, sys:64, sys:179) | none: qShell's RTL uses "edge weights uniformly two" (`appendix.tex:323-327`) | none | n/a | Checked against the primary source: Helios paper, 15 → 38 ns per round at d=13 (`helios_paper.json`). |
| Helios d=9 LUTs (`LUT_D9`) | 116,575 (`system_cost.py:77`) | `capacity_inputs.json` applications["Helios d9"][0] = 116,575 (= native 115,339 + qShell per-decoder extra 1,236) | n/a | **exact** | qShell also gives a *standalone* estimate of 179,328 LUTs (`appendix.tex:285`) and 8.79 % of the V80 including shared infrastructure (`evaluation.tex:101`). §7 uses the decoder-instance footprint. |
| Micro Blossom d=9 LUTs | 155,399 (`system_cost.py:77`) | `capacity_inputs.json` 155,399 | n/a | **exact** | Standalone 218,804 (`appendix.tex:285-286`); 10.30 % of the V80 (`evaluation.tex:102`). |
| "Helios 25 % smaller" (sys:121) | 1 − 116,575/155,399 = 25.0 % | Same inputs | n/a | yes | With qShell's standalone estimates it would be 18 %; with its percentage table, 15 %. |
| V80 LUT budget | 2,574,208 (`system_cost.py:81`) | `capacity_inputs.json` budget[0] | n/a | **exact** | |
| V80 shell | 63,989 (`system_cost.py:81`) | `capacity_inputs.json` common_static[0] = 63,989 | n/a | **exact** | Inside qShell itself, the "resident shell 1.70 % of V80 LUTs" (`evaluation.tex:96`) is ≈ 43.8 k, not 64.0 k (different accounting). §7 follows `capacity_inputs.json`. |
| Footprint growth K_LUT | d^3.39 from Helios Table I, d = 9…21, VCU129 (`system_cost.py:78-80`; sys:13 "d^3.4") | Only d3/d9 exist: Helios 7,758 → 116,575, which implies d^2.47 | n/a | **differs by design** | The `system_cost.py` docstring explains that d=3 is dominated by wrapper logic ("it gave the earlier k = 2.47"). sys:13 says the d=9 footprints come from "V80 synthesis". qShell classifies them as "routed hierarchy/subtractions" (only Helios d3 is synthesis). This is minor wording. |
| Largest d per V80 | Helios d=21, Micro Blossom d=19 (`v80_dmax`; sys:121) | "Largest Helios fit: d21" on V80 (`appendix.tex:118`) | n/a | yes (Helios) | qShell has no Micro Blossom equivalent. |
| Noise model of the Helios curve | `UNIFORM_FRACTION`: Stim uniform depolarizing p (`system_cost.py:69-70`) | SI1000(p), qLDPC 0.3.3 (`appendix.tex:310-311`) | "circuit-level noise at p=1e-3" | **model mismatch** | See section 3. |

## 2. What the qShell 1680 ns contains, and whether "growth only" is consistent

`figure8_ler.json` reports the *native RTL occupancy* of one complete nine-round d=9 shot ("source-reconstructed native RTL bench occupancy from each shot actual cycle counter plus exact fixed protocol/rearm"). It is the sum of two parts:

- a fixed **377 cycles × 4 ns = 1508 ns**: the native interface ingests the 52-byte request holding all 400 Z detectors (one idle cycle between request bytes), serializes the 263-byte correction, and rearms conservatively;
- Helios's **kernel counter**: 60 ns minimum, at zero syndrome.

So the 1680 ns p99 at p=1e-4 is 1508 + 172. The fixed part is not decoding.

`figure8_core_ler.json` holds the kernel counter alone for the same frozen shots. `extract_qshell.py` verifies the relation between the two files:

- native − kernel = **1508 ns exactly**, at all seven p and for the median, p99 and max alike.
- Hence the p99 growth over p=1e-3 is identical whether it is taken from the native occupancy or from the kernel counter. The values are 0 / 176 / 328 / 556 / 788 ns at p = 1e-3, 2e-3, 3e-3, 5e-3, 1e-2.

`system_cost.py` charges Helios `308 + (QSHELL_P99(p) − QSHELL_P99(1e-3))`. That equals the **qShell kernel p99 at p, plus 12 ns, for every p**. The 12 ns is CXL-Quantum's "last syndrome to first result" boundary, 3 cycles past the kernel counter. Using only the growth is therefore exactly consistent: it amounts to subtracting the constant serialization and rearm offset. It is not an approximation.

One open comparability point, a judgment call for the authors: §7 compares Micro Blossom's 772 ns with Helios's *decode stage*. CXL-Quantum's commit 3307a11 calls the 772 ns "decoding a whole d=9 shot in batch", and gives Helios's whole shot as 1.51 µs. It is not documented whether the 772 ns includes Micro Blossom's own request/response serialization. If it does, §7 charges Micro Blossom for its interface but not Helios, which favours Helios. If Helios were charged its whole-shot path, ~1.5 µs plus growth, its unweighted reaction would be ~2.6-2.8 µs (2-4 waiting rounds).

## 3. Uniform-noise mapping vs. qShell's SI1000 curve (sensitivity, not a correction)

`system_cost.helios_latency()` works in two steps:

1. It converts the detection fraction of our d=9 lattice-sim runs into the p of *uniform* depolarizing noise with the same fraction (`UNIFORM_FRACTION`, reproduced exactly by `system/derive_uniform_fraction.py`).
2. It reads qShell's p99 at that p. But qShell measured under **SI1000(p)**, which has measurement flips of 5p and reset flips of 2p, so its detection fraction at the same p is ~1.6-1.7× higher.

With an approximate SI1000 version of the same Stim circuit (`si1000_sensitivity.py`), the comparison is:

| Readout | Uniform-equivalent p (§7) | Helios p99 | R unw. / w | R w16 / w | SI1000-equivalent p | Helios p99 | R unw. / w | R w16 / w |
|---|---|---|---|---|---|---|---|---|
| 1000 ns | 0.68 % | 968 ns | 1.14 µs / 1 | 2.63 µs / 2 | 0.39 % | 757 ns | 0.93 µs / 1 | 2.09 µs / 2 |
| 800 ns | 0.70 % | 977 ns | 1.15 µs / 2 | 2.65 µs / 3 | 0.40 % | 769 ns | 0.94 µs / **1** | 2.12 µs / **2** |
| 600 ns | 0.76 % | 1006 ns | 1.18 µs / 2 | 2.72 µs / 3 | 0.44 % | 807 ns | 0.98 µs / 2 | 2.22 µs / 3 |
| 400 ns | 1.15 % (extrapolated) | 1142 ns | 1.32 µs / 2 | 3.07 µs / 5 | 0.66 % | 958 ns | 1.13 µs / 2 | 2.60 µs / **4** |

The §7 columns reproduce tab:sys_reaction (sys:72-75) exactly. The SI1000 mapping makes Helios ~180-210 ns faster. It would change three things:

- "unweighted Helios crosses at 800 ns" (sys:82) would become 600 ns;
- weighted Helios would wait at most 4 rounds, not 5 (sys:82, sys:84, sys:179);
- the 400 ns point would no longer be beyond qShell's measured range (table footnote, sys:64).

So the §7 mapping is pessimistic for Helios. Its conclusion is unaffected, because unweighted Helios is limited by Λ, not latency. The SI1000 circuit here is an approximation (CX instead of CZ+H, Stim's tick layout), so these numbers indicate a direction, not a replacement.
