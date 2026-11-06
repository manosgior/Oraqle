"""Download the public datasets of the cross-check and check their checksums. Python standard library only.

    python -I fetch.py --out /path/to/external            # default: $ORAQLE_WORK/external
    python -I fetch.py --only qick --verify               # also re-check the MD5 of files already present

Each dataset goes into its own directory and zip archives unpack into <dir>/unz. About 70 GB in all:
QICK 12.3 GB, Cao 50.6 GB (only the natural_decay folder of the 310 GB deposit), the rest under 1 GB each.
"""
import argparse
import hashlib
import json
import os
import pathlib
import urllib.request
import zipfile

ZENODO = dict(qick=(14427490, "qick_14427490"), hassani=(8004359, "hassani_8004359"),
              gunyho=(10000511, "gunyho_10000511"), zheng=(23367046, "zheng_23367046"))
CAO_DOI = "doi:10.7910/DVN/DPMID3"
UA = {"User-Agent": "oraqle-cross-dataset-fetch"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return json.load(r)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path, size, checksum, verify):
    if path.exists() and path.stat().st_size == size and (not verify or md5(path) == checksum):
        print("have", path.name, flush=True)
        return
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    got = md5(tmp)
    if got != checksum:
        raise SystemExit(f"checksum mismatch for {path.name}: {got} != {checksum}")
    os.replace(tmp, path)
    print("fetched", path.name, flush=True)


def zenodo(rec, dest, verify):
    dest.mkdir(parents=True, exist_ok=True)
    for f in get_json(f"https://zenodo.org/api/records/{rec}")["files"]:
        fetch(f["links"]["self"], dest / f["key"], f["size"], f["checksum"].split(":")[1], verify)
        if f["key"].endswith(".zip") and not (dest / "unz").is_dir():
            with zipfile.ZipFile(dest / f["key"]) as z:
                z.extractall(dest / "unz")


def cao(dest, verify):
    """natural_decay only. One file of the deposit (001_prep_g_natural_decay_59) matches its published
    checksum but is not valid HDF5; prepare.py skips it."""
    dest.mkdir(parents=True, exist_ok=True)
    meta = get_json(f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={CAO_DOI}")
    for f in meta["data"]["latestVersion"]["files"]:
        d = f["dataFile"]
        if f.get("directoryLabel") == "natural_decay" and d["filename"].startswith("001_prep_"):
            fetch(f"https://dataverse.harvard.edu/api/access/datafile/{d['id']}?format=original", dest / d["filename"],
                  d["filesize"], d["md5"], verify)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(os.environ.get("ORAQLE_WORK", "work"), "external"))
    ap.add_argument("--only", nargs="*", default=[*ZENODO, "cao"])
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    for name in a.only:
        if name == "cao":
            cao(out / "cao_DPMID3", a.verify)
        else:
            rec, d = ZENODO[name]
            zenodo(rec, out / d, a.verify)
    print("done")


if __name__ == "__main__":
    main()
