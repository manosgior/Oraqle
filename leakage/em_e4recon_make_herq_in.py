"""em_e4recon_make_herq_in.py (tag e4recon) -- RECONSTRUCTED producer of em_e4recon_herq_in.npz.

No script for this step survives: on graham the file ~/oraqle-ics-sims/leakage/em/e4recon/in/
em_e4recon_herq_in.npz (81,378,545 B, 2026-11-05 09:18 UTC) was written by an inline command before
em_e4recon_glrt.py / em_e4recon_analyze.py ran. It is a key-renamed subset of the per-window HERQULES
prediction files written by em_e3herq_assemble_v3.py:

    herq_in key        em_e3herq_preds_<w>.npz key   meaning
    pt_pred_<w>        pred_bits_pertrace            per-trace HERQULES checkpoint, hard bits
    pt_p1_<w>          p1_pertrace                   per-trace HERQULES checkpoint, P(q=1)
    scr_pred_<w>       pred_bits                     demux-pipeline ("scrambled") HERQULES, hard bits
    scr_p1_<w>         p1                            demux-pipeline HERQULES, P(q=1)
    y_<w>              y_state                       prepared 5-bit state (raw TEST order)
    for <w> in (1us, 400ns)

Verified 2026-11-05: all ten arrays produced by this mapping from data/leakage/em_e3herq_preds_{1us,400ns}.npz
are byte-identical (md5 of the array buffers) to the arrays in the graham file.

Usage: python em_e4recon_make_herq_in.py [<dir with em_e3herq_preds_*.npz> [<output npz>]]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lkpaths as LP  # noqa: E402

IN = sys.argv[1] if len(sys.argv) > 1 else str(LP.stage("em/e3herq") / "assembled")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(LP.outdir(LP.stage("em/e4recon") / "in"),
                                                          "em_e4recon_herq_in.npz")
KEYS = {"pt_pred": "pred_bits_pertrace", "pt_p1": "p1_pertrace", "scr_pred": "pred_bits", "scr_p1": "p1",
        "y": "y_state"}

out = {}
for w in ("1us", "400ns"):
    z = np.load(os.path.join(IN, f"em_e3herq_preds_{w}.npz"))
    for new, old in KEYS.items():
        out[f"{new}_{w}"] = z[old]
np.savez_compressed(OUT, **out)
print("wrote", OUT, {k: (v.shape, str(v.dtype)) for k, v in out.items()})
