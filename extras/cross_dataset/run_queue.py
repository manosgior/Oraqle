"""Run the job matrix with a fixed number of parallel workers. No dependencies.

    python -I run_queue.py small --workers 12 --threads 6
    python -I run_queue.py transformer --workers 4 --threads 12

Each group runs in order, with its jobs in parallel, so the KLiNQ teachers are trained once before the
student runs that share them. A job that fails is logged and the rest continue; a finished job is
skipped by run_one.py, so a queue can simply be started again.
"""
import argparse
import itertools
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import XD  # noqa: E402

QICK_T = [200, 400, 600, 800, 1000, 1250, 1500, 1800, 2000, 2400]
CAO_T = [200, 400, 600, 800, 1000, 1200, 1360]
TABLE2_T = [200, 400, 600, 800, 1000]
SMALL = ["linear", "fnn", "qubic", "herqules", "cnn"]
HARD_QUBITS = {"lienhard_q2", "lienhard_q3"}  # the paper gives these two the larger KLiNQ student (FNN-B)


def small_jobs(ds, grid):
    """Everything except MCMit-T. Group 0 trains the KLiNQ teachers."""
    bins = ["--bins", "100"] if ds in HARD_QUBITS else []
    long_t = [t for t in grid if t > 1000]
    first = [(ds, "klinq", 1000, 1, bins)] + ([(ds, "klinq", long_t[-1], 1, bins)] if long_t else [])
    rest = [(ds, m, t, 1, []) for m, t in itertools.product(SMALL, grid)]
    rest += [(ds, "klinq", t, 1, bins) for t in grid if t not in (1000, long_t[-1] if long_t else None)]
    return first, rest


def seed_jobs(ds, grid):
    """Extra seeds, to measure the noise from training alone."""
    return [(ds, m, t, s, []) for m in ("cnn", "herqules", "klinq") for t in grid for s in (2, 3)]


def groups(name):
    if name == "small":
        g0, g1 = [], []
        for ds, grid in [("qick", QICK_T)] + [(f"lienhard_q{q}", TABLE2_T) for q in range(1, 6)]:
            a, b = small_jobs(ds, grid)
            g0 += a
            g1 += b
        g1 += seed_jobs("qick", [1000, 1800]) + seed_jobs("lienhard_q1", [1000])
        g1 += [("qick", "klinq", t, 1, ["--bins", "100", "--tag", "fnnb"]) for t in (400, 1000, 1800)]
        return [g0, g1]
    if name == "cao":
        a, b = small_jobs("cao", CAO_T)
        return [a, b + seed_jobs("cao", [1000])]
    if name == "transformer":  # the paper's configurations; each takes hours on CPU
        order = [("qick", 1000), ("qick", 400), ("qick", 1800), ("qick", 600), ("qick", 800), ("qick", 2400),
                 ("qick", 200), ("lienhard_q1", 1000), ("lienhard_q5", 1000), ("lienhard_q1", 400), ("lienhard_q5", 400)]
        return [[(ds, "transformer", t, 1, []) for ds, t in order]]
    if name == "transformer_cao":
        return [[("cao", "transformer", t, 1, []) for t in (1000, 400, 1200, 600, 800, 200)]]
    raise SystemExit(f"unknown queue {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("queue")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    logs = XD / "logs" / a.queue
    logs.mkdir(parents=True, exist_ok=True)

    def run(job):
        ds, m, t, s, extra = job
        tag = extra[extra.index("--tag") + 1] if "--tag" in extra else ""
        name = f"{ds}_{m}{('_' + tag) if tag else ''}_T{t}_s{s}"
        cmd = [sys.executable, "-I", str(HERE / "run_one.py"), "--dataset", ds, "--model", m, "--T", str(t),
               "--seed", str(s), "--threads", str(a.threads)] + extra
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
        t0 = time.time()
        with open(logs / f"{name}.log", "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"{time.strftime('%H:%M:%S')} rc={rc} {time.time() - t0:7.0f}s {name}", flush=True)

    for i, g in enumerate(groups(a.queue)):
        print(f"{time.strftime('%H:%M:%S')} group {i}: {len(g)} jobs", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(run, g))
    print(f"{time.strftime('%H:%M:%S')} queue {a.queue} finished", flush=True)


if __name__ == "__main__":
    main()
