"""Verify §6 (qec_impact), §7 (system_bottleneck) and the QEC appendix figures from the committed results.

Default (numpy + matplotlib only, minutes; e.g. Discriminators/.venv/bin/python or any python with numpy):
  system   1. re-run the §7 analysis chain from the raw sweep files in results/system into a fresh directory
              ($ORAQLE_OUT/verify/system): analyze.py -> summary.json, summarize_soft.py -> summary_soft.json,
              merge_decoders.sh (analyze_decoders.py over the 3 decoder files) -> decoders_summary.{txt,json},
              system_cost.py -> system_cost.json; compare each with the committed file (numbers: rel 1e-9;
              decoders_summary.txt: byte-identical)
           2. every §7 prose/table number (results/system/paper_values.json, chapter:line refs) from the
              re-derived files (system/paper_numbers.py)
           3. external inputs: qShell numbers re-extracted from the verbatim qShell files (results/external)
           4. the §7 soft-readout CNN imports discriminators/{networks/CNN.py, runners/_colleague_prep.py}: they must
              be byte-identical (md5) to the copies the paper run used (oraqle-ics/sims/qsd)
  qec      5. every §6 / QEC-appendix number (results/qec/paper_values.json) from the ECCentric CSVs
              (qec/eccentric/paper_numbers.py)
Options:
  --figures  regenerate the 5 §7 and 7 §6/appendix figures into $ORAQLE_OUT/paper_figures/{system,qec} and compare
             them with the paper PDFs (page size, text, raster, drawing streams; needs poppler-utils). The appendix
             figures fig:ler_current / fig:ler_futuristic are plots/readout_ler_cnn_{current,futuristic}.pdf
             (MCMit-CNN readout, since the 2026-11-05 [audit fix]); the superseded readout_ler_herqules_*.pdf are
             extras (make -C paper_figures/qec extras compare-extras), not checked here
  --stim     re-derive UNIFORM_FRACTION (system/derive_uniform_fraction.py; needs stim 1.16.0, i.e. the
             oraqle-latsim image:  docker run --rm -v $PWD:/repo:ro -e ORAQLE_OUT=/tmp/out -w /repo oraqle-latsim \\
             python verify/qec_system.py --stim)
  --only system|qec     run one half
Known content issues of the paper (listed in the paper_values.json files) are printed but do not fail the run.
Exit status: 0 if everything matches, 1 otherwise.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from oraqle_env import OUT, RESULTS  # noqa: E402

PY = sys.executable
RAW = ["ler_disc.jsonl", "ler_disc_hi.jsonl", "ler_grid.jsonl", "ler_soft.jsonl", "ler_decoders.jsonl",
       "ler_decoders_soft13.jsonl", "ler_decoders_soft11.jsonl", "soft_params.json"]
DERIVED = ["summary.json", "summary_soft.json", "decoders_summary.json", "system_cost.json"]
QSD_MD5 = {"discriminators/networks/CNN.py": "8b269681cdf5a5cba416d1f71a98a1a6",
           "discriminators/runners/_colleague_prep.py": "c0e0a6ffcdf73f9ebd2a293b8b6959b9"}
RESULTS_LOG = []


def report(name, ok, detail=""):
    RESULTS_LOG.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
    return ok


def run(cmd, cwd=REPO, quiet=True, env=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=quiet, text=True, env=env)
    if quiet and p.returncode:
        sys.stdout.write(p.stdout[-3000:] + p.stderr[-3000:])
    return p


def diff_json(a, b, path="", out=None, rel=1e-9, abs_=1e-12):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: key missing")
            else:
                diff_json(a[k], b[k], f"{path}/{k}", out, rel, abs_)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_json(x, y, f"{path}[{i}]", out, rel, abs_)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isnan(a) and math.isnan(b)) and not math.isclose(a, b, rel_tol=rel, abs_tol=abs_):
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def verify_system(work):
    src = RESULTS / "system"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for f in RAW:
        shutil.copy2(src / f, work / f)
    for f in src.glob("decode_timing_d*.jsonl"):
        shutil.copy2(f, work / f.name)
    shutil.copytree(src / "decode_large", work / "decode_large")
    os.symlink(src / "soft", work / "soft")                       # 44 MB of CNN outputs, read only
    print(f"== §7 analysis chain in {work}", flush=True)
    steps = [("analyze.py -> summary.json", [PY, "system/analyze.py", "--data", str(work)]),
             ("summarize_soft.py -> summary_soft.json", [PY, "system/summarize_soft.py", "--data", str(work)]),
             ("merge_decoders.sh -> decoders_summary.{txt,json}", ["sh", "system/merge_decoders.sh", str(work)]),
             ("system_cost.py -> system_cost.json", [PY, "system/system_cost.py", "--data", str(work)])]
    env = dict(os.environ, PYTHON=PY)
    for name, cmd in steps:
        if not report(f"run {name}", run(cmd, env=env).returncode == 0):
            return
    for f in DERIVED:
        d = diff_json(json.load(open(work / f)), json.load(open(src / f)))
        report(f"re-derived {f} == results/system/{f}", not d, f"{len(d)} differences, e.g. {d[:3]}" if d else
               "identical (numbers within rel 1e-9)")
    same = (work / "decoders_summary.txt").read_bytes() == (src / "decoders_summary.txt").read_bytes()
    report("re-derived decoders_summary.txt byte-identical", same)

    print("\n== §7 paper numbers (results/system/paper_values.json) from the re-derived files", flush=True)
    p = run([PY, "system/paper_numbers.py", "--data", str(work), "--check", str(src / "paper_values.json")], quiet=False)
    report("§7 paper values", p.returncode == 0)

    print("\n== external inputs (results/external)", flush=True)
    p = run([PY, "results/external/extract_qshell.py", "--no-write"], quiet=False)
    report("qShell transcriptions (QSHELL_P99_NS, LUT_D9, V80 budget/shell)", p.returncode == 0)

    for rel, md5 in QSD_MD5.items():
        f = REPO / rel
        if f.exists():
            got = hashlib.md5(f.read_bytes()).hexdigest()
            report(f"{rel} md5 == paper copy (sims/qsd)", got == md5, got)
        else:
            report(f"{rel} present (soft_readout_cnn.py imports it)", False, "missing")


def verify_qec():
    print("\n== §6 / QEC appendix paper numbers (results/qec/paper_values.json)", flush=True)
    p = run([PY, "qec/eccentric/paper_numbers.py", "--check"], quiet=False)
    report("§6 + appendix paper values", p.returncode == 0)


def verify_figures(which):
    if "system" in which:
        print("\n== §7 figures", flush=True)
        out = OUT / "paper_figures" / "system"
        report("regenerate §7 figures", run([PY, "system/plot_readout_bottleneck.py", "--out", str(out)]).returncode == 0)
        report("tab:sys_reaction", run([PY, "paper_figures/system/tab_sys_reaction.py", "--out", str(out)]).returncode == 0)
        p = run([PY, "paper_figures/system/compare_figures.py", "--out", str(out)], quiet=False)
        report("§7 figures == paper PDFs", p.returncode == 0)
    if "qec" in which:
        print("\n== §6 / appendix figures", flush=True)
        report("regenerate §6/appendix figures", run([PY, "paper_figures/qec/make_figure.py", "all"]).returncode == 0)
        ref = RESULTS / "qec" / "reference_figures"
        want = {"readout_ler_cnn_current.pdf", "readout_ler_cnn_futuristic.pdf"}
        report("reference figures = the 7 paper PDFs (incl. readout_ler_cnn_*)",
               len(list(ref.glob("*.pdf"))) == 7 and want <= {f.name for f in ref.glob("*.pdf")})
        p = run([PY, "paper_figures/qec/compare_figures.py"], quiet=False)
        report("§6/appendix figures == paper PDFs", p.returncode == 0)


def verify_stim():
    print("\n== UNIFORM_FRACTION (stim)", flush=True)
    p = run([PY, "system/derive_uniform_fraction.py", "--no-write"], quiet=False)
    report("UNIFORM_FRACTION re-derived (stim)", p.returncode == 0)


def main():
    ap = argparse.ArgumentParser(description="Verify §6/§7/QEC-appendix numbers and figures from results/.")
    ap.add_argument("--only", choices=["system", "qec"])
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--stim", action="store_true")
    ap.add_argument("--work", default=str(OUT / "verify" / "system"))
    a = ap.parse_args()
    which = [a.only] if a.only else ["system", "qec"]
    if "system" in which:
        verify_system(pathlib.Path(a.work))
    if "qec" in which:
        verify_qec()
    if a.figures:
        verify_figures(which)
    if a.stim:
        verify_stim()
    bad = [n for n, ok in RESULTS_LOG if not ok]
    print(f"\n{len(RESULTS_LOG) - len(bad)}/{len(RESULTS_LOG)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
