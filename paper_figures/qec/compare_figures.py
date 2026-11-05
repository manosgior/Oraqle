"""Compare regenerated §6 / appendix figures with the paper's PDFs (results/qec/reference_figures).

For every reference PDF that has a regenerated counterpart in the output directory:
  - page size (pdfinfo) must match,
  - text (pdftotext, whitespace removed) must match: titles, tick labels, legend entries, axis labels,
  - raster difference (pdftoppm, 100 dpi grayscale): share of pixels that differ by more than 64/255.
    Matplotlib versions differ (paper: 3.9.3; this repo's environment may differ), so a small raster
    difference from anti-aliasing is expected; a misplaced curve shows up as a large one.
  - drawing streams (informational): whether every decompressed PDF content stream is byte-identical. True means
    the same drawing commands; only metadata (creator, creation date) and byte offsets differ.

Only the top level of --ref is compared (the paper's figures); the superseded HERQULES-readout versions of
fig:ler_current / fig:ler_futuristic live in reference_figures/extras (compare with --ref .../extras after
`make_figure.py extras`).

Needs poppler-utils (pdfinfo, pdftotext, pdftoppm) and numpy.
Usage: python compare_figures.py [--out DIR] [--ref DIR] [--max-diff 0.01]
Exit status 1 if any page size or text differs, or any raster difference exceeds --max-diff.
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import hashlib
import zlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from oraqle_env import OUT, RESULTS  # noqa: E402


def page_size(pdf):
    info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True, check=True).stdout
    return re.search(r"Page size:\s+(.*)", info).group(1).strip()


def text(pdf):
    t = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True, check=True).stdout
    return re.sub(r"\s+", "", t)


def stream_digests(pdf):
    data = open(pdf, "rb").read()
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        s = m.group(1)
        try:
            s = zlib.decompress(s)
        except zlib.error:
            pass
        out.append(hashlib.md5(s).hexdigest())
    return out


def raster(pdf, tmp):
    stem = os.path.join(tmp, pathlib.Path(pdf).stem + str(abs(hash(pdf))))
    subprocess.run(["pdftoppm", "-gray", "-r", "100", "-singlefile", pdf, stem], check=True)
    with open(stem + ".pgm", "rb") as f:
        data = f.read()
    # binary PGM: P5\n<w> <h>\n<max>\n<bytes>
    m = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", data)
    w, h = int(m.group(1)), int(m.group(2))
    return np.frombuffer(data[m.end():], dtype=np.uint8)[: w * h].reshape(h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("ECC_FIG_OUT") or str(OUT / "paper_figures" / "qec"))
    ap.add_argument("--ref", default=str(RESULTS / "qec" / "reference_figures"))
    ap.add_argument("--max-diff", type=float, default=0.01, help="max share of differing pixels")
    a = ap.parse_args()
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for ref in sorted(pathlib.Path(a.ref).glob("*.pdf")):
            new = pathlib.Path(a.out) / ref.name
            if not new.exists():
                print(f"{ref.name:<45} MISSING (not regenerated)")
                bad += 1
                continue
            ps_ok = page_size(str(ref)) == page_size(str(new))
            tx_ok = text(str(ref)) == text(str(new))
            r0, r1 = raster(str(ref), tmp), raster(str(new), tmp)
            if r0.shape == r1.shape:
                diff = float(np.mean(np.abs(r0.astype(int) - r1.astype(int)) > 64))
            else:
                diff = float("nan")
            st_same = stream_digests(str(ref)) == stream_digests(str(new))
            ok = ps_ok and tx_ok and diff <= a.max_diff
            bad += not ok
            print(f"{ref.name:<45} page {'ok' if ps_ok else 'DIFF'}  text {'ok' if tx_ok else 'DIFF'}  "
                  f"raster {diff:.4%} of pixels differ  drawing streams {'identical' if st_same else 'differ'}"
                  f"  -> {'OK' if ok else 'MISMATCH'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
