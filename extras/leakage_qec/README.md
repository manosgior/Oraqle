# Readout-induced leakage in the surface code, and what readout outputs can do about it

Paper §7.8 quotes a few of these numbers, checked by `make verify`; the rest is not in the paper. Asks what a discriminator should output beyond a hard bit: a confidence for each measurement, a flag that the
ancilla has leaked, or both. lattice-sim has no leakage, so this folder adds it.

All numbers come from `analyze_leak.py` and are in `leak_summary.json`. Lambda is fitted as in `system/analyze.py`. Brackets are 95%
bootstrap intervals over leakage histories and shot noise. Spacetime (ST) is relative to hard 1 us readout without leakage.

## Results

**Confidence output, no leakage.** Per-measurement weights from the measured |logit| table:

| Readout | 400 ns | 600 ns | 1000 ns |
|---|---|---|---|
| hard | 1.53, ST 5.81 | 2.48, ST 0.70 | 2.49, ST 1.00 |
| 1 bit, least confident 10% | 2.10, ST 1.00 | 3.07, ST 0.39 | 2.69, ST 0.76 |
| 2 bits | 1.92, ST 1.51 | 3.35, ST 0.33 | 2.75, ST 0.76 |
| 3 bits | 2.34, ST 0.63 | 3.29, ST 0.33 | 2.82, ST 0.65 |
| 4 bits | 2.54, ST 0.55 | 3.41, ST 0.28 | 2.84, ST 0.65 |

Intervals are about +-0.15, so 2 to 4 bits cannot be told apart at 600 ns. Compared with the paper's full erasures, weights raise a
10% flag at 400 ns from 1.87 to 2.10 and a 5% flag at 600 ns from 2.90 to 2.98.

**Leakage without removal.** Lambda by leakage per measurement at 600 ns, random Pauli on the partner of a blocked gate (2.52 without
leakage on the same distances):

| Decoder | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|---|---|
| no leakage knowledge | 2.27 | 2.20 | 1.70 | 1.20 | 1.07 | above |
| flag | 2.32 | 2.28 | 1.87 | 1.49 | 1.05 | above |
| true history (oracle) | 2.48 | 2.57 | 2.22 | 1.83 | 1.22 | above |

The measured rates are 1.4-6.3% per readout (`leak_params.json`). At those rates the code is above threshold, or compatible with it,
for every decoder, including the oracle (at best 1.07 [0.81, 1.42]). A decoder that knows only the average rate gains nothing. In
paired comparisons at 1e-3 to 3e-3 the flag raises Lambda by 4-24% and the oracle by 12-53%.

**Removal.** Reduction unit with assumed efficiency 0.99, error 0.2%, 200 ns, at the measured rates, hard readout:

| Policy | Lambda, 600 ns | Lambda, 1000 ns |
|---|---|---|
| always on, every ancilla, every round | 2.00-2.37 | 1.93-2.21 |
| triggered by the flag, 1% FPR | 1.03-1.45 | 0.98-1.31 |
| triggered by the flag, 50% FPR, p_ld 0.75 | 1.84-2.05 | 1.93-1.94 |

Missed flags are the limit: at 1% FPR the detector finds 69-83% of L at 600 ns, and what it misses stays above the tolerance above.
The flag never clearly beats always-on at these rates. It ties at best, at about 50% FPR, where it fires on half the ancillas anyway.
It is competitive at low leakage: at 1e-3, flag-triggered removal reaches 2.21-2.42 against 2.28-2.41 for always-on, with about 1% of
the pulses. It starts to fall behind at 3e-3 and is clearly behind at 1e-2 (1.60-2.12 against 2.18-2.32).

Always-on depends on the unit: Lambda is 2.00-2.32 at efficiency 0.99, 1.47-1.76 at 0.95, 1.20-1.42 at 0.9, and at or above threshold
at 0.5. The slot costs more than the leakage it removes: at 600 ns the slot alone, with no leakage, takes ST from 0.70 to 1.10, and
leakage then takes it to 1.24-1.91.

**Together.** Always-on removal with soft readout, measured leakage, p_ld 0.75:

| Readout | 400 ns | 600 ns | 800 ns | 1000 ns |
|---|---|---|---|---|
| hard, pooled | ST 10.96 | ST 1.24 | ST 1.82 | ST 2.58 |
| 3 bits, pooled | ST 1.14 | ST 0.64 | ST 1.00 | ST 1.67 |
| 3 bits, q3_like | ST 1.14 | ST 1.10 | ST 1.45 | ST 2.58 |

## Model

`leaksim.py` draws a leakage history per pattern and compiles a truth circuit (sampled) and a belief circuit (decoder weights), both
from the lattice-sim `circuit_4` circuit. A leaked ancilla loses its gates, randomizes the partner with strength `p_ld`, and repeats its
previous outcome, so its detectors stay silent. Rates, survival and detector ROC are measured (`readout_outputs.py`, `make_params.py`).
Scenarios: `pooled` (Q1, Q3, Q4 = 0, Q5) and `q3_like` (the worst real leakage). Q4's L is a split |1> response, not leakage.

**Measured:** leakage creation, survival, flag TPR/FPR, the confidence table.

**Assumed, and swept:** `p_ld`, reduction-unit efficiency, error and duration.

**Limits:**

- The rates of one five-qubit device are combined with the `ibm_boston` gate noise. The tolerance table does not depend on the device's
  rates and is the most portable result.
- No transport of leakage to data qubits. Adding it would hurt no-removal and missed flags more than always-on, so it should widen the
  gaps above, but this was not simulated.
- Half of readouts are assumed to return |1>. A removed ancilla is assumed to return to the correct state.
- Weights change per measurement. Here the matcher is rebuilt per pattern; the cost of doing that in a real-time decoder is not measured.
- The flag is a linear filter on the 10 input channels, not synthesized. The CNN's pooled features do much worse because pooling
  destroys timing.
- Lambda falls with the distances fitted (no leakage, hard: 2.52 on d 5-11, 2.34 on d 9-13 at 600 ns), so required distances near 50 are
  extrapolations and probably optimistic. This applies to every row here and to the same fit in `system/analyze.py`.
- ST moves in steps because distance is odd, so small ST differences are not meaningful.

## Files

| File | Purpose |
|---|---|
| `train_cnn.py` | retrains the CNN per readout length (needs the dataset) |
| `models/` | the CNN weights these results used; pass `--checkpoints` to `readout_outputs.py` to reuse them instead of retraining |
| `readout_outputs.py` | leakage rates, flag detectors, confidence table; aggregates only, no per-shot data (needs the dataset) |
| `make_params.py` | builds `leak_params.json` |
| `leaksim.py`, `test_leaksim.py` | simulator and its checks (no-leakage DEM identical, published LER reproduced) |
| `sweep_leak.py` | experiments: `soft tolerance scenarios lru combined eta`; about 2 h in total on 100 cores |
| `analyze_leak.py` | tables and `leak_summary.json`, from `sweeps/` |

```bash
python analyze_leak.py                     # the tables, from the shipped sweeps; needs only numpy
# in the oraqle-latsim image, from the repository root:
python extras/leakage_qec/test_leaksim.py
python extras/leakage_qec/sweep_leak.py --exp soft --params extras/leakage_qec/leak_params.json --out OUT/ler_soft.jsonl
```
