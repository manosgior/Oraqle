#!/usr/bin/env bash
# usage: fetch_zenodo.sh <record_id> <dest_dir>   Downloads every file of a Zenodo record and checks its md5.
set -u
rec=$1; dest=$2; mkdir -p "$dest"; cd "$dest" || exit 1
curl -sS --retry 5 "https://zenodo.org/api/records/$rec" -o .record.json
python3 - <<PY > .manifest.tsv
import json
r=json.load(open(".record.json"))
for f in r["files"]:
    print(f["key"], f["checksum"].split(":")[1], f["size"], f["links"]["self"], sep="\t")
PY
while IFS=$'\t' read -r key md5 size url; do
  echo "$(date +%T) fetching $key ($size bytes)"
  curl -sS -L --fail --retry 8 --retry-delay 20 -C - -o "$key" "$url" || echo "CURL FAILED $key"
  got=$(md5sum "$key" | cut -d' ' -f1)
  [ "$got" = "$md5" ] && echo "$(date +%T) OK $key" || echo "$(date +%T) MD5 MISMATCH $key $got != $md5"
done < .manifest.tsv
echo "$(date +%T) done record $rec"
