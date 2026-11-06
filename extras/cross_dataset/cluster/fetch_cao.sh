#!/usr/bin/env bash
# usage: fetch_cao.sh <manifest.tsv> <dest_dir> [max per state]   Downloads Cao et al. Dataverse files, checks md5.
man=$1; dest=$2; maxn=${3:-100000}; mkdir -p "$dest"; cd "$dest" || exit 1
declare -A cnt
while IFS=$'\t' read -r id md5 size dir name; do
  key=$(echo "$name" | sed -E 's/_[0-9]+$//'); cnt[$key]=$(( ${cnt[$key]:-0} + 1 ))
  [ "$dir" != root ] && [ "${cnt[$key]}" -gt "$maxn" ] && continue
  if [ -f "$name" ] && [ "$(stat -c %s "$name")" = "$size" ]; then echo "have $name"; continue; fi
  curl -sS -L --fail --retry 6 --retry-delay 20 -o "$name" "https://dataverse.harvard.edu/api/access/datafile/$id?format=original" || { echo "CURL FAILED $name"; continue; }
  got=$(md5sum "$name" | cut -d' ' -f1)
  [ "$got" = "$md5" ] && echo "$(date +%T) OK $name" || echo "$(date +%T) MD5 MISMATCH $name"
done < "$man"
echo "$(date +%T) done cao"
