"""What would this readout do to QEC? The (T, readout error) pairs measured on another dataset, through the
paper's own simulation: system/sweep_ler.py, same circuit, same noise preset (ibm_boston), same fit
(system/analyze.py). Only the readout changes. The gates and coherence stay those of the paper's device,
so this asks what happens if a device with these gates had this readout, not what the source device does.

    python -I sweep_qec.py --dataset qick --models cnn herqules --out /w/cross_dataset/ler_qick.jsonl --workers 90
    python -I sweep_qec.py --dataset qick --models cnn herqules --out ... --analyze     # fit only

Run it in the oraqle-latsim image. Needs $ORAQLE_WORK/cross_dataset/summary.json from analyze.py.
"""
import argparse
import json
import os
import pathlib
import sys
from multiprocessing import Pool

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "system"), str(HERE)]

XD = pathlib.Path(os.environ.get("ORAQLE_WORK", str(REPO / "work"))) / "cross_dataset"


def points(dataset, models, distances, preset="ibm_boston"):
    tab = json.loads((XD / "summary.json").read_text())["datasets"][dataset]["table"]
    pts = []
    for m in models:
        for T, v in tab[m].items():
            pts += [(d, int(T), round(1 - v["F"], 6), preset, f"{dataset}/{m}") for d in distances]
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=90)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()

    if not a.analyze:
        from sweep_ler import DISTANCES, jobs_for, run

        jobs = jobs_for(points(a.dataset, a.models, DISTANCES), quick=False)
        print(len(jobs), "jobs", flush=True)
        with Pool(a.workers) as pool, open(a.out, "a") as f:
            for i, r in enumerate(pool.imap_unordered(run, jobs)):
                f.write(json.dumps(r) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(i, flush=True)

    from analyze import load, summarize  # system/analyze.py

    rows = summarize(load(a.out))
    keep = ("label", "T", "p", "Lambda", "d@1e-09", "qubits@1e-09", "op_us@1e-09", "spacetime@1e-09", "t_r_us", "readout_share")
    out = [{k: r.get(k) for k in keep} for r in rows]
    pathlib.Path(a.out).with_suffix(".summary.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(r["label"], r["T"], "p=%.4f" % r["p"], "Lambda", None if r["Lambda"] is None else round(r["Lambda"], 3),
              "d@1e-9", r["d@1e-09"], "spacetime", None if r["spacetime@1e-09"] is None else round(r["spacetime@1e-09"]))


if __name__ == "__main__":
    main()
