"""
assemble_three_level_table.py
=============================
Collect the per-model 3-level JSONs (train_three_level_*.py outputs) and join
them with the published 2-level F_5Q numbers to produce the survey's headline
table: the 2-level SOTA ranking vs the 3-level ranking, on three metrics
(plain F_5Q accuracy, balanced/leakage-aware F_5Q, and mean |2> recall).

The thesis: at 2-level HERQULES/CNN win; at 3-level the PLAIN-accuracy ranking
barely moves (|2> is rare), but the leakage-aware metric flips it - HERQULES's
binary matched filters are blind to |2> (recall ~ 0), while trace-based models
recover it.

    python -m runners.assemble_three_level_table \
        --json-dir /app/optimization_reports \
        --out /app/optimization_reports/three_level_survey_table.csv
"""

from __future__ import annotations

import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from loguru import logger

# Published 2-level mean-5 fidelity (from evaluate_truncated.mean5_acc_map_500).
TWO_LEVEL_F5Q = {
    "FNN": 0.909,
    "HERQULES_Net_rmf": 0.914,
    "CNN": 0.908,
    "Transformer": 0.906,
    "KLiNQ_Student": 0.908,
    "Arxiv240618807FNN": 0.893,
}

DISPLAY = {
    "FNN": "FNN",
    "HERQULES_Net_rmf": "HERQULES",
    "CNN": "CNN",
    "Transformer": "Transformer",
    "KLiNQ_Student": "KLiNQ",
    "Arxiv240618807FNN": "Arxiv (QubiCML)",
}


def run(json_dir, out_path):
    rows = []
    for path in sorted(glob.glob(os.path.join(json_dir, "*_3level.json"))):
        with open(path) as f:
            d = json.load(f)
        model = d.get("model")
        if model not in TWO_LEVEL_F5Q:
            logger.warning(f"{os.path.basename(path)}: unknown model '{model}', skipping.")
            continue
        leak = [x for x in d.get("per_qubit_leakage_recall", []) if x == x]  # drop NaN
        rows.append({
            "model": DISPLAY.get(model, model),
            "_key": model,
            "params": d.get("n_params"),
            "F5Q_2level": TWO_LEVEL_F5Q[model],
            "F5Q_3level_acc": d.get("f5q_acc"),
            "F5Q_3level_bal": d.get("f5q_balanced"),
            "mean_leak_recall": float(np.mean(leak)) if leak else float("nan"),
        })

    if not rows:
        logger.error(f"No *_3level.json found in {json_dir}. Run the train_three_level_* scripts first.")
        return

    df = pd.DataFrame(rows)
    # Rankings (1 = best). 2-level + 3-level plain accuracy + 3-level balanced.
    df["rank_2level"] = df["F5Q_2level"].rank(ascending=False, method="min").astype(int)
    df["rank_3level_acc"] = df["F5Q_3level_acc"].rank(ascending=False, method="min").astype(int)
    df["rank_3level_bal"] = df["F5Q_3level_bal"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("rank_3level_bal").reset_index(drop=True)

    cols = ["model", "params", "F5Q_2level", "F5Q_3level_acc", "F5Q_3level_bal",
            "mean_leak_recall", "rank_2level", "rank_3level_acc", "rank_3level_bal"]
    df = df[cols]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path}")

    # Pretty markdown to stdout.
    def fmt(r):
        return (f"| {r['model']:<16} | {r['params']:>8} | {r['F5Q_2level']:.3f} | "
                f"{r['F5Q_3level_acc']:.3f} | {r['F5Q_3level_bal']:.3f} | "
                f"{r['mean_leak_recall']:.3f} | {r['rank_2level']} | "
                f"{r['rank_3level_acc']} | {r['rank_3level_bal']} |")

    print("\n| Model            | Params   | 2L F5Q | 3L acc | 3L bal | |2> rec | r2L | r3La | r3Lb |")
    print("|------------------|----------|--------|--------|--------|---------|-----|------|------|")
    for _, r in df.iterrows():
        print(fmt(r))

    # Flip summary.
    best_2l = df.loc[df["rank_2level"].idxmin(), "model"]
    best_3l_bal = df.loc[df["rank_3level_bal"].idxmin(), "model"]
    logger.info(f"2-level winner: {best_2l}  |  3-level (balanced) winner: {best_3l_bal}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    p.add_argument("--json-dir", default=str(REPORTS_OUT),
                   help="Directory with the *_3level.json files.")
    p.add_argument("--out", default=str(ensure_dir(REPORTS_OUT) + "/three_level_survey_table.csv"))
    args = p.parse_args()
    run(args.json_dir, args.out)
