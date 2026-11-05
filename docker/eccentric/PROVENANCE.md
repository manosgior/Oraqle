# Provenance: docker/eccentric

| File | Origin | Changes |
|---|---|---|
| `Dockerfile` | TUM-DSE/eccentric @da7164c `Dockerfile` (identical to graham `~/eccentric/Dockerfile`, which built the `eccentric` image used for every §6 run on 2026-06-09; that image no longer exists on graham or steve) | header comment; `FROM` pinned to the digest recorded in that build's log; installs `requirements.oraqle.lock` (pip 26.1.2, as in the log) instead of `requirements.txt` |
| `requirements.lock` | parsed from the pip "Successfully installed …" line of graham `~/eccentric/docker_build.log` (2026-06-09 21:54; laptop mirror `/home/manosgior/Documents/eccentric/docker_build.log`, 1168 lines). Base digest from line 16 of the same log. | new file; 235 `name==version` pins |

Verification (tier 2, graham, 2026-11-05): `fetch_eccentric.sh` then
`docker build -f work/eccentric/Dockerfile.oraqle -t unify-qec-eccentric work/eccentric` succeeded in 3 min 8 s
(image 5c39fba0fc12, 2.87 GB; upstream sanity step "eccentric image OK"). `pip freeze` in the image lists exactly the
locked versions (setuptools is hidden by pip freeze; `svg.py` = `svg-py`). The image `unify-qec-eccentric` is left on
graham for the coordinator to remove (`docker rmi unify-qec-eccentric`).
