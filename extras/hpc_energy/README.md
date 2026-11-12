# Power and energy-to-solution of chemistry and materials programs (draft)

Not in the paper. A first draft of two quantities an HPC facility asks about a readout operating point: the
power it draws per logical qubit (classical plane and QPU), and the time and energy a real application costs.
It reuses the §7 cost model (`system/system_cost.py`) and the committed fits (`results/system/summary.json`)
unchanged, so it needs no dataset, no docker and no new simulation.

    python extras/hpc_energy/energy_apps.py              # m8 CNN build, scaled QPU scenario
    python extras/hpc_energy/energy_apps.py --cnn m16    # the Table 2 CNN (3,061 params)
    python extras/hpc_energy/energy_apps.py --qpu today  # QPU power per qubit as measured on Sycamore

Needs numpy only. Output: `results/energy_apps{,_m16,_today}.{txt,json}`.

## Files

| File | Contents |
|---|---|
| `apps.py` | four published instances in the workload form of `system/workloads.py`, each with source and how it was checked |
| `constants.json` | power constants with sources |
| `energy_apps.py` | power per logical qubit, and qubits, runtime, decoder cards, power and energy per program |

## Model

Per logical qubit at distance `d`:

- **Decoder, FPGA:** Micro Blossom's V80 share (`v80_per_patch`, LUTs scaled as d^3.39 from the d=9 build,
  perfectly partitioned) times the card's 190 W.
- **Decoder, CPU:** the PyMatching cores that keep pace with the round (`cpu_cores`) times 3.5 W per core
  (EPYC 7713P package power over 64 cores, the machine the timings ran on).
- **Discriminator:** `d^2 - 1` ancillas, five per readout line, times the LUTs of one synthesized instance
  (`fpga/hls4ml/reports`), times the power per LUT.
- **QPU:** `2 d^2 - 1` physical qubits times a flat power per physical qubit that covers the refrigerator, its
  chilled water and the room-temperature control electronics. Two scenarios:
  - `scaled` (default): 6.25 W per qubit, a 26 kW refrigerator holding 4,158 qubits (Parker and Vermeer, RAND
    WR-A2427-3, 2023). The authors exclude the classical processing of error correction, which is what the
    decoder term adds, and call the figure extremely rough.
  - `today`: 491 W per qubit, the 26 kW measured for the 53-qubit Sycamore (Arute et al. 2019, supplement).

A program's energy is `N_L x (FPGA decoder + discriminator + QPU power) x runtime`, with Micro Blossom's
reaction wait and a 1% failure budget, as in §7.4.

Applications (`apps.py`): every non-Clifford gate is one sequential feed-forward step and a rotation is 60
sequential T gates, so runtimes are upper bounds. `N_L` is the algorithm's logical qubits, without routing space
or factories, as in `system/workloads.py`.

## First results (m8 CNN build, scaled QPU)

Per logical qubit at 1e-9 per operation:

| Readout | d | Classical | QPU | Total | Classical share |
|---|---|---|---|---|---|
| CNN 1000 ns | 47 | 7.3 kW | 27.6 kW | 34.9 kW | 21% |
| CNN 600 ns | 45 | 6.5 kW | 25.3 kW | 31.8 kW | 20% |
| CNN 400 ns | 105 | 68.6 kW | 137.8 kW | 206.5 kW | 33% |
| HERQULES 600 ns | 53 | 5.8 kW | 35.1 kW | 40.9 kW | 14% |

Programs:

| Program | Readout | d | Physical qubits | Runtime | Classical | QPU | Total energy vs CNN 1000 ns |
|---|---|---|---|---|---|---|---|
| FeMoco (Reiher) | CNN 1000 ns | 79 | 26.7 M | 6.7 d | 64 MW | 167 MW | 1.00 |
| | CNN 600 ns | 77 | 25.4 M | 4.6 d | 60 MW | 159 MW | 0.65 |
| | CNN 400 ns | 177 | 134 M | 8.2 d | 729 MW | 839 MW | 8.3 |
| | HERQULES 600 ns | 91 | 35.5 M | 5.4 d | 70 MW | 222 MW | 1.02 |
| Dynamics 10x10 | CNN 1000 ns | 53 | 0.56 M | 133 s | 1.0 MW | 3.5 MW | 1.00 |
| | CNN 600 ns | 51 | 0.52 M | 92 s | 0.9 MW | 3.3 MW | 0.63 |

- The classical plane is 20-28% of the total at 600-1000 ns and 33-47% at 400 ns.
- A 600 ns CNN readout costs 0.63-0.65 of the energy of 1000 ns in both QPU scenarios, almost all of it from
  the shorter runtime.
- HERQULES at 600 ns draws less classical power than the CNN at 1000 ns (5.8 against 7.3 kW per logical
  qubit), but it needs a larger distance, so with the QPU included its total energy is 0.91-1.02 of the CNN's.
- With `--qpu today` the QPU is over 99% of the power (FeMoco: 13 GW), so today's per-qubit power cannot be
  the one a machine of this size runs at.

The Ising-100 anchor reproduces §7.4 (12.3 s at 1000 ns, 8.5 s at 600 ns).

## What to trust

- **Ratios between operating points** follow from the same fits as §7 and should hold as well as §7's.
- **Absolute classical power is an extrapolation.** The decoder footprint is one d=9 build scaled to d=77-191
  with a law measured up to d=21; FeMoco's roughly 200,000 V80 cards and tens of MW say more about that law
  than about a real machine.
- **QPU power is one flat number per qubit.** The two scenarios differ by 78x, and the classical share moves
  with them from about 20% to under 1%. Neither separates cryogenics from control electronics, and both already
  contain their machines' readout electronics, so the discriminator is counted twice to a small degree.
- **Classical power is proportional to occupied LUTs at vendor ceilings (TDP).** No power report exists for
  any of the builds. The discriminator uses the V80's power per LUT although it runs on a ZCU216.
- **The discriminator share depends on the build.** With the m8 CNN it is about as large as the decoder's
  (4.1 against 3.2 kW per logical qubit at 1000 ns); with the m16 CNN it is three times the decoder's. Only the
  network is synthesized, not the demodulation front-end.
- **CPU power** beyond d=57 extrapolates the timing fit (starred rows).
- **Beverland's counts** were taken from the Qualtran reproduction of Appendix F, not from the paper's table.
- Links and the facility cooling of the classical plane are not counted.
