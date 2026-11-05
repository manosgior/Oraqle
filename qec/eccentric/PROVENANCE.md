# Provenance: qec/eccentric

Assembled 2026-11-05 for the Oraqle artifact. Paper = oraqle-ics working tree of that date.

## Pinned upstream (fetched, not vendored)

| Repository | Commit | Role |
|---|---|---|
| github.com/TUM-DSE/eccentric, branch `mcm-latency-eccentric` | `da7164c7f2eb5bc1e3ec46cfdc50111dbd42db42` (branch tip, 2026-10-19) | library + Oraqle drivers + CSVs of §6 |
| same | `5658bb8774266fa2230fd5dda64397d502949060` (2026-07-12, "MCMit and Oraqle results") | the commit the results were committed in; da7164c differs from it only in README.md and `synchronization-artifact/evaluate_mcm_{error,latency}.py` (MCMit) |
| same | `7858c2cc3c1778bf2fd26149c470f39f99d3211e` (2026-06-11, "first qec results") | `plot_readout_compare.py` and the HERQULES-readout comparison drivers behind fig:ler_current / fig:ler_futuristic |
| github.com/aswierkowska/qiskit-qec (`main`) | `7136727c201a8c76178a7222210f09080f2641aa` | submodule `external/qiskit_qec` |
| github.com/aswierkowska/more-bacon-less-threshold (`main`) | `df98313954f51637e2dfbe70323587ca72ca31dd` | submodule `external/more-bacon-less-threshold` |

Public availability checked with `git ls-remote` on 2026-11-05 (the three SHAs are branch tips or ancestors of
`mcm-latency-eccentric`; the submodule SHAs are the `main` tips). `fetch_eccentric.sh` was run for real (local and on
graham): all five SHAs verified; the fetched tree equals `git archive da7164c` byte for byte except the overlay files.
The library and driver files of the fetch equal graham `~/eccentric` (the non-git directory the runs used; md5 of
run_ler_*.py, compare_readout_*.py, explore_readout.py, make_readout_tradeoff.py, gen_noise_results_md.py,
run_cnn_all.sh, synchronization-artifact/{compare_readout_artifact,run_herqules_both,run_herqules_futuristic}.py,
codes/*.py, noise/*.py all identical).

## Vendored files

| File | Origin (host:path @commit) | md5 of origin | Changes |
|---|---|---|---|
| `overlay/run_ler_discriminators.py` | eccentric @da7164c (= graham `~/eccentric/run_ler_discriminators.py`) | see `git show da7164c:run_ler_discriminators.py` | `MASTER_CSV` resolution (env `ORAQLE_MASTER_FIDELITY` / `ORAQLE_RESULTS` / original `../oraqle_reports/`), clear error if missing; docstring line. No logic, seed or noise change. |
| `plots/original/plot_discriminators.py` | eccentric @5658bb8 = @da7164c | 6965d6a1a8964d3a6cd5c0754ea01683 | none (the local eccentric checkout was at 5658bb8, clean, when plot_paper_pairs.py imported it on 2026-10-15; transcript eb84b470 line 575). graham's copy (f437c8c8…) is an older variant. |
| `plots/original/plot_single_model.py` | eccentric @5658bb8 = @da7164c | d790c09c064f18a50a13317e02618864 | none |
| `plots/original/plot_readout_compare.py` | eccentric @7858c2c | b6e09092a4bc2e4beb869884cb2948c9 | none (graham and da7164c hold the later `_cnn` variant f4194279…) |
| `plots/original/plot_noise_models_paper.py` | graham `~/eccentric/plot_noise_models_paper.py` (non-git; laptop mirror `/home/manosgior/Documents/eccentric`) | a2f71a04ee4cd663aab5f59c9e70ff65 (checked on graham) | none |
| `plots/original/plot_single_model_paper.py` | graham `~/eccentric/plot_single_model_paper.py` (non-git) | 48c6cdcab3bbba59e17c5ee574dbe5b6 (checked on graham) | none |
| `plots/recovered/plot_paper_pairs.orig_recovered.py` | Claude transcript `/home/manosgior/.claude/projects/-home-manosgior-Documents-GitHub-oraqle-ics/eb84b470-86cf-448f-8d70-7f6127bf91db.jsonl`: Write tool call at line 600 (2026-10-15T07:47:59Z, 8,865 characters) to `/home/manosgior/Documents/GitHub/eccentric/plot_paper_pairs.py`, then the five in-place modifications of that file in the same session, applied in order: lines 627 (python replace list, 8 edits), 644 (legend upper right), 673 (finish() rewritten, legend at the bottom, 4 edits), 712 (legend centred, "Lower is better" on top, figure heights, 7 edits), 726 (`sed` top=0.815). Every replacement's `assert s.count(a)==1` held. The file was never committed and no longer exists anywhere else. | b2caae6e84d77b9ad8a37e250f145a38 (recovered) | none |
| `plots/plot_*.py`, `plots/plot_paper_pairs.py` | the originals above | — | path plumbing only: `from _paths import res, out`; input paths `experiment_results/...` → `res(...)` (`$ECC_RESULTS`), output dirs → `$ECC_FIG_OUT`. `diff plots/original/X plots/X` shows every change (2-8 lines each). |
| `plots/_paths.py`, `paper_numbers.py`, `smoke/smoke_point.py`, `fetch_eccentric.sh` | new | — | — |
| `cluster_only/{build_thesis_results,mcm_qec_stats,plot_mcm_qec_thesis}.py` | graham `~/eccentric/` (non-git, 2026-06-13) | 9806de18…, 7c8e9d41…, 1ea22e7d… (checked on graham) | none. MCMit thesis scripts, not used by the Oraqle paper; they expect a `results/` dir produced by build_thesis_results.py. |

## Verification of the recovery

`plot_paper_pairs.py` (path-plumbed) run with numpy 2.4.4 / matplotlib 3.10.9 on `results/qec` regenerates
`ler_pair_{discriminators,noise,future}.pdf` with the paper's page sizes (504 × 133.2 / 190.8 / 190.8 pt), identical
pdftotext text, and **0 differing pixels** when both are rasterized at 300 dpi (pdftoppm, grayscale). The same holds
for the four other §6/appendix figures, and every decompressed PDF drawing stream is byte-identical to the paper's.
The paper PDFs say "Matplotlib v3.9.3" (eccentric `.venv`); only metadata (creator, creation date) and byte offsets
differ. Re-run on graham inside the pinned eccentric image (matplotlib 3.9.3): same result.
