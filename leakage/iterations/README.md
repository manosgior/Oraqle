# Superseded iterations and probes

Kept for provenance, **byte-identical to the originals** (no path plumbing): they still refer to the original
docker mounts (`/feat`, `/reports`, `/work/out`, ...) or laptop paths, and some import sibling modules that now
live one directory up (`leakage/`). Run them, if needed, the way the canonical scripts are run, after adding
`leakage/` to `PYTHONPATH` and mounting the inputs at the original container paths. None of them feeds a paper
number or a committed result file.

| file | origin | superseded by / role |
|---|---|---|
| `em_e1hmm_attrib.py`, `em_e1hmm_attrib_v2.py` | local = graham `em/e1hmm/scripts` | `em_e1hmm_attrib_v3.py`, then `_v6.py` (v1/v2 had the HMM label/relabelling issues fixed in v3) |
| `em_e1hmm_attrib_v5.py` | idem | `em_e1hmm_attrib_v6.py` (v6 imports v3 and v4, not v5) |
| `em_e2seg_attr.py` | local only | `em_e2seg_attr_v2.py` / `_v3.py` |
| `em_e2seg_explore.py` | graham `em/e2seg/scripts` only | exploration probe before e2seg v1 |
| `em_e3herq_assemble.py`, `em_e3herq_assemble_v2.py` | local only | `em_e3herq_assemble_v3.py` |
| `em_e5vstat_coredebug.py` | local = graham | one-off debug of the late-core modes (imports `em_e5vstat_check`) |
| `m1tail_analysis.py`, `m1tail_plot.py` | local (+ graham `m1tail/scripts` for the analysis) | `m1tail_v2*.py` (v2 overwrote the PNGs) |
| `m2mix_core.py`, `m2mix_fit.py` | local = graham | `m2mix_core2.py`, `m2mix_fit2.py` |
| `m2mix_smoke2.py` | graham `m2mix/scripts` only | smoke test of the v2 mixture fit |
| `m3time_examples.py`, `m3time_plot.py` | local (+ graham for examples) | `m3time_examples_re.py`, `m3time_plot_re.py` (canonical state labels) |

md5 of each file: `../PROVENANCE.md`.
