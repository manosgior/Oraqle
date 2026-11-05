#!/usr/bin/env bash
# Fetch the exact ECCentric revision the Oraqle paper used (§6 and the QEC appendix), over HTTPS, and overlay the
# Oraqle path plumbing.
#
#   qec/eccentric/fetch_eccentric.sh [DEST] [--with-7858c2c]
#
#   DEST            default ${ORAQLE_WORK:-<repo>/work}/eccentric  (work/ is gitignored)
#   --with-7858c2c  also create DEST-7858c2c, a git worktree at 7858c2c ("first qec results"), whose
#                   compare_readout_{eccentric,synchronization-artifact/artifact}.py hold the HERQULES_ACC tables
#                   behind the appendix comparison CSVs (fig:ler_current, fig:ler_futuristic)
#
# Pinned revisions (all public; checked with git ls-remote on 2026-11-05):
#   TUM-DSE/eccentric                      da7164c7f2eb5bc1e3ec46cfdc50111dbd42db42  (tip of branch mcm-latency-eccentric;
#                                          identical to 5658bb8 for every Oraqle driver and result; see PROVENANCE.md)
#   aswierkowska/qiskit-qec                7136727c201a8c76178a7222210f09080f2641aa  -> external/qiskit_qec
#   aswierkowska/more-bacon-less-threshold df98313954f51637e2dfbe70323587ca72ca31dd  -> external/more-bacon-less-threshold
# The .gitmodules of da7164c points at git@github.com (SSH) URLs, which need a GitHub key, so the two submodules are
# cloned directly over HTTPS instead of with `git submodule update`.
#
# After the checkout, the script copies into DEST:
#   overlay/run_ler_discriminators.py   (master_fidelity.csv path plumbing; the only driver with a host path)
#   ../../docker/eccentric/Dockerfile        -> DEST/Dockerfile.oraqle       (pinned base image + lock file)
#   ../../docker/eccentric/requirements.lock -> DEST/requirements.oraqle.lock
#   smoke/smoke_point.py                -> DEST/smoke_point.py           (one-point smoke test, no logic change)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DEST=""
WITH_OLD=0
for arg in "$@"; do
  case "$arg" in
    --with-7858c2c) WITH_OLD=1 ;;
    -h|--help) sed -n 2,26p "$0"; exit 0 ;;
    *) DEST="$arg" ;;
  esac
done
DEST=${DEST:-${ORAQLE_WORK:-$REPO/work}/eccentric}

ECC_URL=https://github.com/TUM-DSE/eccentric.git
ECC_BRANCH=mcm-latency-eccentric
ECC_SHA=da7164c7f2eb5bc1e3ec46cfdc50111dbd42db42
OLD_SHA=7858c2cc3c1778bf2fd26149c470f39f99d3211e
QQ_URL=https://github.com/aswierkowska/qiskit-qec.git
QQ_SHA=7136727c201a8c76178a7222210f09080f2641aa
MB_URL=https://github.com/aswierkowska/more-bacon-less-threshold.git
MB_SHA=df98313954f51637e2dfbe70323587ca72ca31dd

checkout() {  # url sha dir [branch]
  local url=$1 sha=$2 dir=$3 branch=${4:-}
  if [ ! -d "$dir/.git" ] && [ ! -f "$dir/.git" ]; then
    mkdir -p "$dir"
    if [ -n "$branch" ]; then
      git clone --quiet --single-branch --branch "$branch" --no-checkout "$url" "$dir"
    else
      git clone --quiet --no-checkout "$url" "$dir"
    fi
  fi
  if ! git -C "$dir" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git -C "$dir" fetch --quiet origin "$sha"
  fi
  git -C "$dir" -c advice.detachedHead=false checkout --quiet --force "$sha"
  local got
  got=$(git -C "$dir" rev-parse HEAD)
  if [ "$got" != "$sha" ]; then
    echo "ERROR: $dir is at $got, expected $sha" >&2
    exit 1
  fi
  echo "  $(basename "$dir"): $got"
}

echo "ECCentric -> $DEST"
checkout "$ECC_URL" "$ECC_SHA" "$DEST" "$ECC_BRANCH"
checkout "$QQ_URL" "$QQ_SHA" "$DEST/external/qiskit_qec"
checkout "$MB_URL" "$MB_SHA" "$DEST/external/more-bacon-less-threshold"

# The gitlinks recorded in the ECCentric tree must name the same submodule commits.
for sub in qiskit_qec:$QQ_SHA more-bacon-less-threshold:$MB_SHA; do
  path=external/${sub%%:*}; want=${sub##*:}
  rec=$(git -C "$DEST" ls-tree "$ECC_SHA" "$path" | awk '{print $3}')
  [ "$rec" = "$want" ] || { echo "ERROR: $path gitlink is $rec, expected $want" >&2; exit 1; }
done

echo "Overlay (path plumbing, docker pins, smoke wrapper)"
cp "$HERE/overlay/run_ler_discriminators.py" "$DEST/run_ler_discriminators.py"
cp "$REPO/docker/eccentric/Dockerfile" "$DEST/Dockerfile.oraqle"
cp "$REPO/docker/eccentric/requirements.lock" "$DEST/requirements.oraqle.lock"
cp "$HERE/smoke/smoke_point.py" "$DEST/smoke_point.py"
git -C "$DEST" status --short | sed 's/^/  /'

if [ "$WITH_OLD" = 1 ]; then
  OLD="$DEST-7858c2c"
  echo "ECCentric @7858c2c (appendix comparison drivers) -> $OLD"
  if [ ! -e "$OLD" ]; then
    git -C "$DEST" fetch --quiet origin "$OLD_SHA" 2>/dev/null || true
    git -C "$DEST" worktree add --quiet --detach "$OLD" "$OLD_SHA"
  fi
  [ "$(git -C "$OLD" rev-parse HEAD)" = "$OLD_SHA" ] || { echo "ERROR: $OLD not at $OLD_SHA" >&2; exit 1; }
  echo "  eccentric-7858c2c: $OLD_SHA"
  for s in qiskit_qec:$QQ_SHA:$QQ_URL more-bacon-less-threshold:$MB_SHA:$MB_URL; do
    IFS=: read -r name sha url <<<"$s"
    checkout "$url" "$sha" "$OLD/external/$name"
  done
fi
echo "done. Build the image with:  docker build -f $DEST/Dockerfile.oraqle -t eccentric $DEST"
