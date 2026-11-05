# Figure 8: whole-shot-loss logical error rate

All points contain 50,000 independent d9 nine-round memory-Z logical-qubit shots.
Top: 64 qubits, seven PER values. Bottom: PER 0.005. Physical rounds are 1 us;
complete requests arrive every 9 us per qubit independent of PER.
Static has one instance; QShell uses the documented resource-fit projection.
Admission is arrival-order, idle-eligible-instance only, with no queue.
Dropped shots receive no correction and remain in the LER denominator.
Accepted work drains after quantum readout. Ideal retains Helios's correction
algorithm but has zero effective processing time and no capacity drops.

## Timing and scope

source-reconstructed native RTL bench occupancy from each shot actual cycle counter plus exact fixed protocol/rearm; not full per-shot timestamps

Per-input RTL native occupancy, not wall-clock runtime or an average proxy,
controls admission. Native/occupancy times include conservative rearm; processed
times end at the final response handshake. Both policies use the same native
boundary. Shell/adapter/transport and interconnect contention are excluded.
Resource footprints predate the adapted finite-memory RTL: ten instances are a
resource-fit projection, not demonstrated placement or physical timing closure.
The timing provenance in figure8_ler.json specifies exact sender, clock and rearm.

## Statistics files

figure8_ler_statistics.csv contains every plotted point, LER, drop rates,
processed/drop logical-failure contributions, late-completion and unfinished counts.
native_* columns summarize all offered native occupancies (including hypothetical
work on dropped shots); processed_* summarize actual accepted response latencies;
occupancy_* summarize accepted effective busy intervals. Quantiles are nearest rank.
JSON additionally retains distributions as histograms and per-point sample counts.
Per-shot traces are retained in the producer evidence root p*/patches-*.jsonl.gz
with hashes in JSON; they include every accepted/dropped shot and applied parity.
No IID confidence interval is asserted for timing-selected outcomes. Zero remains
zero in data; a plot marker at 1/N only denotes zero observed failures.

## Shared operating point (64 qubits, PER 0.005)

| Policy | Logical failures / 50000 | LER | Drops | Drop rate |
|---|---:|---:|---:|---:|
| Static | 23654 | 0.47308 | 49218 | 0.98436 |
| QShell | 21408 | 0.42816 | 42180 | 0.8436 |
| Ideal capacity | 8140 | 0.1628 | 0 | 0 |
