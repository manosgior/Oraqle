# extras/leakage: beyond the paper

The paper reports only the error-mode attribution (Sec. 5.3, Table `tab:error_modes`, App. `app:error-modes`).
The full leakage study behind it is in this repository too:

* `leakage/docs/LEAKAGE_REPORT.md`: is there leakage in the Lienhard five-qubit data (yes on Q3, 11 % of
  prepared |1> at 1 us, not identified as |2>; Q1/Q5 tied to Q3), how it was measured four independent ways
  (tail statistics, 2-D mixtures, temporal HMM, confounder screens), three adversarial verifications, which
  existing numbers were wrong or circular, and the (open) proposal of a per-qubit L flag for MCMit-CNN.
* Code: `leakage/` families m1tail, m2mix, m3time, m4conf, s1/s2/s3, r, vphysics, vreproduction, vstatistics,
  f (README section 4, S8). Results: `results/leakage/` (`m*`, `v*`, `s*`, `r_*`, `f_cnn_leak_xtab.json`, PNGs).
* Open items listed in the report's section 6 (leakage-aware QEC simulation, CNN L-flag head, Q4 sub-clouds).

## Is MCMit-CNN at the accuracy ceiling?

[`hmm_ceiling/`](hmm_ceiling/README.md) uses a decay- and leakage-aware HMM as a discriminator and combines it with the
CNN. The combination lowers the error by at most 0.07 pp (Q1), so little accuracy is left to gain.
