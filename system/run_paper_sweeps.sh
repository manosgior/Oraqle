#!/bin/sh
# Re-run the §7 (system_bottleneck.tex) simulations exactly as the paper ran them, in the new repository layout.
#
# The commands are the ones executed on steve/graham on 2026-10-26..29 (Claude session 94019386, transcript
# ~/.claude/projects/-home-manosgior-Documents-GitHub-oraqle-ics/94019386-7e9a-4302-8276-39a280e43d5e.jsonl, lines given
# per step), with two path changes only: the old mount `-v ~/oraqle-ics-sims:/work -w /work/sims` becomes
# `-v $REPO:/repo:ro -w /repo` plus `-v $OUT:/out` for the outputs, and the CNN dataset mount `/data/five_qubit_data`
# becomes `-v $ORAQLE_DATA:/data:ro -e ORAQLE_DATA=/data`. Worker counts and --cpus are the paper's (steve: 256 cores).
# Every job is seeded from its point, so with the oraqle-latsim images the results reproduce the committed
# results/system/*.jsonl bit for bit (system/replay_points.py checks a sample); decode times are machine-dependent.
#
# Usage: system/run_paper_sweeps.sh STEP [STEP ...]     OUT=<dir> (default $ORAQLE_WORK/system = work/system)
#   disc grid disc_hi        sweep_ler.py      (oraqle-latsim)       ~5 / ~21 / ~15 min wall at 90/110/90 cores
#   timing timing_large      decode_timing.py  (oraqle-latsim, one pinned core per job; graham cores 40..)  13 / 25 min
#   soft_cnn                 soft_readout_cnn.py (oraqle-torch-gpu, needs $ORAQLE_DATA and a GPU)  ~30 min per length
#   soft_qec                 sweep_soft.py     (oraqle-latsim, 150 cores; needs OUT/soft from soft_cnn, or
#                                               SOFT=results/system/soft to reuse the committed CNN outputs)
#   decoders dec13 dec11     sweep_decoders.py (oraqle-latsim:dec)   3.6 h at 180 cores (stopped at 126/136 jobs), 22 min, 2.5 min
#   crosscheck               crosscheck_stim.py (oraqle-latsim:dec, 24 cores; not cited in the paper)
#   analysis                 analyze.py, summarize_soft.py, merge_decoders.sh, system_cost.py on OUT (numpy only)
# Images: docker/README.md. Container names are oraqle-<step> as in the paper runs; remove them with docker rm.
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-${ORAQLE_WORK:-$REPO/work}/system}
SOFT=${SOFT:-/out/soft}
mkdir -p "$OUT/logs"
U="$(id -u):$(id -g)"
RUN="docker run --user $U -e HOME=/tmp -v $REPO:/repo:ro -v $OUT:/out -w /repo"
case "$SOFT" in /out/*) SOFTMNT="" ;; *) SOFTMNT="-v $(cd "$SOFT" && pwd):/soft:ro"; SOFT=/soft ;; esac

for step in "$@"; do
  case "$step" in
  disc)      # transcript line 463 (2026-10-26T10:32Z): ~200 failures per point (superseded by disc_hi in §7)
    $RUN -d --name oraqle-disc --cpus 90 oraqle-latsim sh -c \
      "python system/sweep_ler.py --set disc --out /out/ler_disc.jsonl --workers 90 > /out/logs/disc.log 2>&1" ;;
  grid)      # line 463: T 200..1000 ns x p 1..20% x d 5..15, target 200 failures
    $RUN -d --name oraqle-grid --cpus 110 oraqle-latsim sh -c \
      "python system/sweep_ler.py --set grid --out /out/ler_grid.jsonl --workers 110 > /out/logs/grid.log 2>&1" ;;
  disc_hi)   # line 562 (10:39Z): every discriminator's (T, p), target 1000 failures, 5x shot cap
    $RUN -d --name oraqle-disc-hi --cpus 90 oraqle-latsim sh -c \
      "python system/sweep_ler.py --set disc --out /out/ler_disc_hi.jsonl --workers 90 --target-fails 1000 --cap-scale 5 > /out/logs/disc_hi.log 2>&1" ;;
  timing)    # line 490 (graham, 10:35Z): one pinned core per distance; d = 19, 21 failed there (decoder-latency
             # lookup; decode_timing.py now bypasses it above d = 17) and were covered by timing_large
    core=40
    for d in 5 7 9 11 13 15 17; do
      $RUN -d --name oraqle-dec-d$d --cpuset-cpus $core oraqle-latsim sh -c \
        "python system/decode_timing.py --d $d --out /out/decode_timing_d$d.jsonl > /out/logs/dec_d$d.log 2>&1"
      core=$((core + 2))
    done ;;
  timing_large)  # lines 791 and 1085 (graham, 11:42Z and 11:59Z): d 21..57, 2000 shots (d <= 33) or 400
    mkdir -p "$OUT/decode_large"; core=40
    for d in 21 25 29 33 41 49 57; do
      for T in 400 600 800 1000; do
        if [ $d -le 33 ]; then n=2000; else n=400; fi
        $RUN -d --name oraqle-decL-d$d-T$T --cpuset-cpus $core oraqle-latsim sh -c \
          "python system/decode_timing.py --d $d --T $T --shots $n --out /out/decode_large/d${d}_T$T.jsonl > /out/logs/decL_d${d}_T$T.log 2>&1"
        core=$((core + 1))
      done
    done ;;
  soft_cnn)  # line 935 (steve H200, 11:49Z): CNN retrained per readout length (CNN_LEN raw samples, 2 ns each)
    : "${ORAQLE_DATA:?set ORAQLE_DATA to the directory with DRaw_C_Tr_v0-001 and DRaw_C_Te_v0-002}"
    mkdir -p "$OUT/soft"
    for L in 100 200 300 400 500; do
      $RUN -d --name oraqle-soft-$L -e CNN_LEN=$L -e THREADS=8 --device=nvidia.com/gpu=all --cpus 8 \
        -v "$ORAQLE_DATA":/data:ro -e ORAQLE_DATA=/data oraqle-torch-gpu sh -c \
        "python system/soft_readout_cnn.py --out /out/soft > /out/logs/soft_$L.log 2>&1"
    done ;;
  soft_qec)  # lines 974 and 1387 (12:26Z; helpers/chain_soft.sh waited for soft_cnn): flagged fractions as erasures;
             # also writes OUT/soft_params.json
    $RUN -d --name oraqle-softqec --cpus 150 $SOFTMNT oraqle-latsim sh -c \
      "python system/sweep_soft.py --soft $SOFT --out /out/ler_soft.jsonl --workers 150 > /out/logs/softqec.log 2>&1" ;;
  decoders)  # line 1901 (2026-10-29T13:49Z): every decoder on every shot. The paper run was stopped by hand at 126 of
             # 136 jobs (17:56Z, line 2146); the missing d=11/13 soft jobs came from dec13/dec11 without union-find.
             # Letting it finish changes only union-find shot counts; MWPM results are identical (seeded per point).
    $RUN -d --name oraqle-decoders --cpus 180 oraqle-latsim:dec sh -c \
      "python system/sweep_decoders.py --out /out/ler_decoders.jsonl --soft /out/soft_params.json --workers 180 > /out/logs/decoders.log 2>&1" ;;
  dec13)     # line 2058 (16:31Z): d = 13 soft points, union-find off
    $RUN -d --name oraqle-decoders-13 --cpus 16 oraqle-latsim:dec sh -c \
      "python system/sweep_decoders.py --out /out/ler_decoders_soft13.jsonl --soft /out/soft_params.json --d 13 --soft-only --uf-frac 0 --workers 8 > /out/logs/decoders_soft13.log 2>&1" ;;
  dec11)     # line 2146 (17:56Z): the d = 11 soft jobs missing from the stopped main run, union-find off
    $RUN --rm --name oraqle-decoders-11 oraqle-latsim:dec sh -c \
      "python system/sweep_decoders.py --out /out/ler_decoders_soft11.jsonl --soft /out/soft_params.json --d 11 --soft-only --uf-frac 0 --workers 8 --done-from /out/ler_decoders.jsonl > /out/logs/decoders_soft11.log 2>&1" ;;
  crosscheck)  # line 2872 (20:41Z): simulator cross-check (I11), not cited in the paper
    $RUN -d --name oraqle-xcheck --cpus 24 oraqle-latsim:dec sh -c \
      "python system/crosscheck_stim.py --out /out/ler_crosscheck.jsonl --workers 24 > /out/logs/crosscheck.log 2>&1" ;;
  analysis)  # numpy only; same chain as verify/qec_system.py, written into OUT
    PY=${PYTHON:-python3}
    $PY "$REPO/system/analyze.py" --data "$OUT"
    $PY "$REPO/system/summarize_soft.py" --data "$OUT"
    PYTHON=$PY sh "$REPO/system/merge_decoders.sh" "$OUT"
    $PY "$REPO/system/system_cost.py" --data "$OUT" ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
