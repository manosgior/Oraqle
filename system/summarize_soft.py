"""Producer of summary_soft.json (§7.6 soft readout): analyze.summarize over the soft-readout sweep.

summary_soft.json had no producer script: it was written by an inline call in Claude session 94019386
(transcript ~/.claude/projects/-home-manosgior-Documents-GitHub-oraqle-ics/94019386-7e9a-4302-8276-39a280e43d5e.jsonl,
line 1417, 2026-10-26T12:36Z), run from oraqle-ics/sims as

    rows = load('../data/sims/ler_soft.jsonl'); S = summarize(rows)
    ... (prints the table below) ...
    json.dump(S, open('../data/sims/summary_soft.json', 'w'), indent=1, default=float)

This script is that call with the paths as arguments; the printed table is the one the inline call printed
(Lambda, d and qubits at 1e-9 per op, spacetime relative to hard 1000 ns, and the pairwise eps ratios).
It reproduces the published summary_soft.json exactly (numpy only).

Usage: python system/summarize_soft.py [--data DIR]   (reads DIR/ler_soft.jsonl, writes DIR/summary_soft.json;
       DIR defaults to $ORAQLE_WORK/system)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, summarize  # noqa: E402  (analyze.py also puts the repo root on sys.path)
from oraqle_env import WORK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(WORK / "system"))
    a = ap.parse_args()
    rows = load(os.path.join(a.data, "ler_soft.jsonl"))
    S = summarize(rows)
    base = next(s for s in S if s["label"] == "soft_f0" and s["T"] == 1000)
    print(f"{'T':>5}{'f':>6}{'e_u':>8}{'Lambda':>8}{'d@1e-9':>8}{'qubits':>8}{'ST(rel CNN1000 hard)':>22}  pairwise")
    for T in [200, 400, 600, 800, 1000]:
        for s in sorted([s for s in S if s["T"] == T], key=lambda s: float(s["label"][6:])):
            f = float(s["label"][6:])
            e = s["eps"]
            ds = sorted(int(d) for d in e)
            pw = [e[ds[i]] / e[ds[i + 1]] for i in range(len(ds) - 1)
                  if s["fails"][ds[i]] >= 30 and s["fails"][ds[i + 1]] >= 30 and e[ds[i + 1]] > 0]
            st = s.get("spacetime@1e-09")
            rel = st / base["spacetime@1e-09"] if st else float("nan")
            print(f"{T:>5}{f:>6.2f}{s['p']:>8.4f}{s.get('Lambda', float('nan')):>8.2f}{str(s.get('d@1e-09', '-')):>8}"
                  f"{str(s.get('qubits@1e-09', '-')):>8}{rel:>22.2f}  {' '.join(f'{x:.2f}' for x in pw)}")
    json.dump(S, open(os.path.join(a.data, "summary_soft.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
