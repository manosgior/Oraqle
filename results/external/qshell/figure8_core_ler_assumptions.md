# Figure8 core-capacity upper bound

CORE-COMPUTE upper bound: idealized wide/buffered I/O; native byte protocol and wrapper not implemented/measured. Not actual interface or hardware throughput.

Actual frozen per-shot Helios kernel counter multiplied by 4ns. Transport, initialization and readout negligible/overlapped; same assumption for Static and QShell. Ideal capacity uses the same actual predictions at zero service time.

Unlimited per-stream FIFO, fixed stream modulo cores binding, per-core round robin. One active job per core, no preemption. Full independent nine-round windows arrive every 9000ns per stream. Waiting inputs expire at arrival+9000ns; instantaneous input handoff at exactly the deadline is allowed. Already active kernels may finish later.

Frozen d9, nine-round SI1000 finite-memory Helios corpus; no counter, noise, correction or RTL changes. Complete independent windows are stateless; 256 streams, ten cores and unlimited storage are projections, not deployed hardware.

Actual frozen 350,000 SI1000 shots/counters/corrections are hash-bound in JSON. Every operating point scores all 50,000 outcomes: dropped shots use raw parity (identity correction), not assumed failure. Independent stateless full windows use rotating stream/sample mapping and persistent per-core RR cursors. Time zero is the first full-window arrival. Lower panel is p=.005 at 1..256 streams; upper panel is 64 streams at seven physical error rates. No forced drops, knees, inequalities, runtime constants, or capacity confidence intervals. Native byte protocol and wrapper not implemented/measured for this I/O assumption. Reference footprint fit gives ten QShell cores; it is not a ten-core placed design.
