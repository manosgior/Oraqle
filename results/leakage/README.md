# results/leakage

Small result files of the readout-leakage and error-mode analysis (173 files incl. this README and
`paper_values.json`, 37.6 MB; the largest is
`em_e1hmm_results.json`, 3.7 MB). Byte copies of `oraqle-ics/data/leakage/` as of 2026-11-05, except where
noted below; produced by the scripts in `leakage/` (map: `leakage/README.md`, md5: `leakage/PROVENANCE.md`).

**Files the paper uses directly**
* `em_e8fix_final_numbers.json`: every number of Sec. 5.3, Table `tab:error_modes`, Takeaway #5 and App.
  `app:error-modes`, each with CI, status, source file and key; `paper_text_map` maps paper sentences to keys.
* `em_e4recon_final.json`: the final attribution (counts, CIs) behind it; `em_e5vstat_*.json`,
  `em_e5vrepro_*.json`: the two adversarial verifications; `m1tail_v2_percfg_test.json`: per-configuration
  leakage populations (Q3 conditioning of Q1/Q5 leakage).
* `paper_values.json`: the printed paper values with chapter:line references, checked by `verify/leakage.py`.
  Not from oraqle-ics; written for this repository.

**Everything else** feeds those files or the leakage report (`leakage/docs/LEAKAGE_REPORT.md`): feature-set
diagnostics (`x_*`), HMM (`m3time_*`, `em_e1hmm_*`), segment detector (`em_e2seg_*`), HERQULES summary
(`em_e3herq_summary_v3.json`), tail/mixture/confounder/verifier studies (`m1tail_*`, `m2mix_*`, `m4conf_*`,
`vphysics_*`, `vreproduction_*`, `vstatistics_*`, `s1local_*`, `s2cluster_*`, `r_*`), and their PNG figures.
The `.md` digests `em_e2seg_summary.md`, `em_e4recon_final.md`, `em_e6edit_final_numbers.md` and
`r_leakage_summary.md` were written by hand next to the JSON (no script writes them); `em_e1hmm_summary.md` is
written by `em_e1hmm_summarize.py`.

**Not here**
* `em_e3herq_preds_{1us,600ns,400ns}.npz` (128 / 107 / 129 MB, over the 50 MB limit): per-shot HERQULES
  predictions on the full test set. Regenerate (55 s, no dataset needed once the e3herq runs exist):
  `python leakage/em_e3herq_assemble_v3.py $ORAQLE_WORK/leakage/em/e3herq/out $ORAQLE_WORK/leakage/em/e3herq/assembled`;
  the inputs `em_e3herq_{A,B,C}_*ns.npz` come from `em_e3herq_regen_{demux,scrambled,pertrace_ckpt}.py` (raw data,
  ~1.5 h in total). Re-generated this way on graham on 2026-11-05: all three files byte-identical (md5
  eef537b8..., c418cd15..., 1c717adb...).
* `LEAKAGE_REPORT.md` moved to `leakage/docs/`.
* Cluster intermediates (features, HMM fits, per-shot arrays; ~21 GB): regenerable, see `leakage/README.md`.

Note: `m3time_re_examples_raw.npz` (100 demodulated example traces, used only by the non-paper plot
`leakage/m3time_plot_re.py`) is intentionally not shipped, because it is a slice of the raw dataset.
Regenerate it with `leakage/m3time_examples_re.py` once ORAQLE_DATA is set.
