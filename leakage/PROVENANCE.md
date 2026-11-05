# Provenance of the leakage / error-mode files

Generated 2026-11-05. Sources: local `oraqle-ics` working tree (untracked `sims/leakage/`, `data/leakage/`, `plots/iq_error_gallery_q5_v2.pdf`; Overleaf git HEAD d7cd4e9), graham `~/oraqle-ics-sims/leakage` (shared NFS home, md5 listing of 2026-11-05). The local copies are canonical (a superset, newer for the local-only assembly scripts); for the 82 scripts present in both places the md5 is identical. md5 = of the original file.

Changes are limited to path plumbing: absolute laptop paths and container mount points replaced by `lkpaths` names (see `README.md` section 2), three import lines added at the top (`import lkpaths as LP` and the script directory on `sys.path`), and for the e3herq scripts `discriminators/` added to `sys.path` (was `PYTHONPATH`). One output directory is now created instead of assumed (`em_e3herq_assemble_v3.py`). No analysis constant, threshold, seed or formula was touched; `diff` against the originals shows only the lines listed in `README.md` section 2. Remaining `/home/manosgior` strings in canonical scripts are in docstrings/comments only. Outputs written as provenance labels inside JSON (e.g. `"source": "data/leakage/..."`) were left as they are, so the regenerated JSON stays byte-identical.

## leakage/ (canonical scripts)

| file | original (local `oraqle-ics/sims/leakage/`) md5 | identical cluster copy (`~/oraqle-ics-sims/leakage/`) | change |
|---|---|---|---|
| `em_e1hmm_attrib_v3.py` | `3c66f70dce35685b495d1c0ed9ef7c2f` | `em/e1hmm/scripts/em_e1hmm_attrib_v3.py` | path plumbing: 4 line(s) replaced, 3 import line(s) added |
| `em_e1hmm_attrib_v4.py` | `cf2db71ad2d3cb2890de4d42167f0ab0` | `em/e1hmm/scripts/em_e1hmm_attrib_v4.py` | path plumbing: 3 line(s) replaced, 3 import line(s) added |
| `em_e1hmm_attrib_v6.py` | `681540be61261b39b6836c22aeafb1d6` | `em/e1hmm/scripts/em_e1hmm_attrib_v6.py` | path plumbing: 3 line(s) replaced, 3 import line(s) added |
| `em_e1hmm_noisecheck.py` | `f4bcd5077cbef8e231e2da277768d1e6` | `em/e1hmm/scripts/em_e1hmm_noisecheck.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `em_e1hmm_oldproc.py` | `43eb6e9125f1317f47291ea63c591e60` | `em/e1hmm/scripts/em_e1hmm_oldproc.py` | path plumbing: 7 line(s) replaced, 3 import line(s) added |
| `em_e1hmm_summarize.py` | `4e1562e2606c08101fa17462633e5c77` | - | unchanged |
| `em_e2seg_assemble.py` | `0c88a93f70e6a15dd46acc69d3a3b34a` | - | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `em_e2seg_attr_v2.py` | `0f0391c45fde8e4649aed4db17f91fea` | `em/e2seg/scripts/em_e2seg_attr_v2.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `em_e2seg_attr_v3.py` | `6504ff74932f0cdbd84bbdeda9c9d995` | `em/e2seg/scripts/em_e2seg_attr_v3.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `em_e2seg_oldmethod.py` | `3375f83a99d859f6797ee4fd5a1d7d96` | `em/e2seg/scripts/em_e2seg_oldmethod.py` | path plumbing: 7 line(s) replaced, 3 import line(s) added |
| `em_e2seg_oldmethod_l200scan.py` | `483306a6ac34b300b4d6b8ca6ef7f37d` | `em/e2seg/scripts/em_e2seg_oldmethod_l200scan.py` | path plumbing: 4 line(s) replaced, 3 import line(s) added |
| `em_e3herq_assemble_v3.py` | `0699a5b6996f54d3be9e54ed7c4ab44f` | - | path plumbing: 3 line(s) replaced, 3 import line(s) added |
| `em_e3herq_regen_demux.py` | `330fa44c126e44a32dfb03d46e620dec` | `em/e3herq/em_e3herq_regen_demux.py` | path plumbing: 1 line(s) replaced, 4 import line(s) added |
| `em_e3herq_regen_pertrace_ckpt.py` | `827e179040c7528cfb268b557c56972e` | `em/e3herq/em_e3herq_regen_pertrace_ckpt.py` | path plumbing: 3 line(s) replaced, 4 import line(s) added |
| `em_e3herq_regen_scrambled.py` | `065f746af41299477c19cdd4174bfd2d` | `em/e3herq/em_e3herq_regen_scrambled.py` | path plumbing: 1 line(s) replaced, 4 import line(s) added |
| `em_e3herq_ycheck.py` | `bc9f53a9ee79a9acd13dbcacac23fca2` | `em/e3herq/em_e3herq_ycheck.py` | path plumbing: 4 line(s) replaced, 3 import line(s) added |
| `em_e4recon_analyze.py` | `667bfd404c90c4b85a42b63a10fec265` | `em/e4recon/scripts/em_e4recon_analyze.py` | path plumbing: 6 line(s) replaced, 3 import line(s) added |
| `em_e4recon_assemble.py` | `5d489f377ea77a9f0f42880e1c19ebcf` | - | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `em_e4recon_glrt.py` | `ec14f7fb6055ccd155b68db59b459dd1` | `em/e4recon/scripts/em_e4recon_glrt.py` | path plumbing: 3 line(s) replaced, 3 import line(s) added |
| `em_e4recon_make_herq_in.py` | new | - | RECONSTRUCTED producer of graham `em/e4recon/in/em_e4recon_herq_in.npz` (no script survived; the inline step is documented in its docstring; output arrays verified byte-identical to the graham file) |
| `em_e5vrepro_analyze.py` | `668dc37149a00c404986a1a413fba582` | `em/e5vrepro/scripts/em_e5vrepro_analyze.py` | path plumbing: 5 line(s) replaced, 3 import line(s) added |
| `em_e5vrepro_assemble.py` | `96e8d9f478996eedf34c264c14aaca77` | - | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `em_e5vrepro_demod.py` | `00e134afc7c674eac883a56260f40868` | `em/e5vrepro/scripts/em_e5vrepro_demod.py` | path plumbing: 3 line(s) replaced, 3 import line(s) added |
| `em_e5vrepro_diag_partners.py` | `fcbcc177c09b1225fe35574acdbad2db` | `em/e5vrepro/scripts/em_e5vrepro_diag_partners.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `em_e5vrepro_diag_q3relax.py` | `2736d72b9d2050d37813de3885a0ae57` | `em/e5vrepro/scripts/em_e5vrepro_diag_q3relax.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `em_e5vrepro_diag_xtalk.py` | `6e49a0c018f4e6f3db9235a12ff51a8a` | `em/e5vrepro/scripts/em_e5vrepro_diag_xtalk.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `em_e5vrepro_ifscan.py` | `f09b88e2e65d936c40719d522f96a94f` | `em/e5vrepro/scripts/em_e5vrepro_ifscan.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `em_e5vstat_check.py` | `0bfd5e706509e738d803ee12ad3a4961` | `em/e5vstat/scripts/em_e5vstat_check.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `em_e5vstat_lda.py` | `7689817456a81808d32f4f0d9f2fc76e` | `em/e5vstat/scripts/em_e5vstat_lda.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `em_e5vstat_q3core.py` | `3b32bfcf6bfb9c79a8e0f8ec05698929` | `em/e5vstat/scripts/em_e5vstat_q3core.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `em_e5vstat_summarize.py` | `c70c35178bc44fa2542bc495701b3033` | - | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `em_e6edit_finalize.py` | `7472f2a01a5714ed3c7c8d47952c118f` | - | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `em_e6edit_iq_gallery.py` | `41e80f2afab7c582933e8f5b2f6782d0` | - | path plumbing: 3 line(s) replaced, 3 import line(s) added |
| `em_e8fix_finalize.py` | `357f67e60d4fca9f1f1ed4f584f5d704` | - | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `f_cnn_leak_xtab.py` | `9156758ccc12e707da53e74c85d3f0cc` | `f/scripts/f_cnn_leak_xtab.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `lkpaths.py` | new | - | path module (this repository) |
| `m1tail_v2.py` | `c19695620e79b7d0986d205b5b3b5b8b` | `m1tail/scripts/m1tail_v2.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `m1tail_v2_cooc.py` | `ac6b7cb65ad13d2290924bf97e1e3735` | `m1tail/scripts/m1tail_v2_cooc.py` | unchanged |
| `m1tail_v2_models.py` | `3dbfd0ce8ce13c160badcf744bba0859` | `m1tail/scripts/m1tail_v2_models.py` | unchanged |
| `m1tail_v2_percfg.py` | `d04e3f5f1fe40c12f3b4b31be614df11` | `m1tail/scripts/m1tail_v2_percfg.py` | unchanged |
| `m1tail_v2_plot.py` | `710e8c4f58170d14a5934f80f2746af4` | `m1tail/scripts/m1tail_v2_plot.py` | unchanged |
| `m1tail_v2_summary.py` | `a377fe837dfa80291ff17ff40a2b4abf` | - | unchanged |
| `m1tail_v2_xtab.py` | `bec9774bac908b4a06d21f43c2b274ea` | `m1tail/scripts/m1tail_v2_xtab.py` | unchanged |
| `m2mix_boot2.py` | `f2c734861057070f897f8adca2c70d5a` | `m2mix/scripts/m2mix_boot2.py` | unchanged |
| `m2mix_coinc2.py` | `22fb2b98c33dad276d178bdd1ab89b7e` | `m2mix/scripts/m2mix_coinc2.py` | unchanged |
| `m2mix_core2.py` | `0c52dadfd494b7d0322f1710dc82ed05` | `m2mix/scripts/m2mix_core2.py` | unchanged |
| `m2mix_feature2.py` | `611cc6dddec47d7c4933a6adf8381e9e` | - | unchanged |
| `m2mix_fit2.py` | `b31d82449886605d461fdc11e58981a6` | `m2mix/scripts/m2mix_fit2.py` | unchanged |
| `m2mix_plot2.py` | `255f67713b8096ff9f52a0dc56bfefb8` | - | unchanged |
| `m2mix_prep.py` | `8b8d6be8bdc4493c75440cff922d9b5d` | `m2mix/scripts/m2mix_prep.py` | unchanged |
| `m2mix_summarize2.py` | `bb338d63d00088b6ca8f720966bf9199` | - | unchanged |
| `m2mix_traj2.py` | `4fb17d02de428067f63e11be759bd9ff` | `m2mix/scripts/m2mix_traj2.py` | unchanged |
| `m3time_examples_re.py` | `aaede12acb92e995e7a6daaf515c5db4` | `m3time/scripts/m3time_examples_re.py` | path plumbing: 4 line(s) replaced, 3 import line(s) added |
| `m3time_hmm.py` | `892c5d544cc7ccc2e7cff97ad0e98368` | `m3time/scripts/m3time_hmm.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `m3time_joint.py` | `0c7694e56c382b566f9e4b1fbf9a7e94` | `m3time/scripts/m3time_joint.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `m3time_plot_re.py` | `94ab5e4a04d83db9c0937fac6aafd27d` | - | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `m3time_reanalyze.py` | `24cef420bacc3b141e4ee2baf2433abe` | `m3time/scripts/m3time_reanalyze.py` | unchanged |
| `m3time_summary.py` | `01a4ac68776a33ae45f6fc55cc6093cf` | - | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `m3time_xchk.py` | `1c0c597b04c65fb0ce78bc44e6a2072f` | `m3time/scripts/m3time_xchk.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `m3time_xchk2.py` | `50c9830369b9e7eb32ac683fb3a5224e` | `m3time/scripts/m3time_xchk2.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `m4conf_analysis.py` | `9157eee3708089f29438e748a52d8f7a` | `m4conf/m4conf_analysis.py` | unchanged |
| `m4conf_dccheck.py` | `3d1d3adcbbdb7764664300c126db58a2` | `m4conf/m4conf_dccheck.py` | unchanged |
| `m4conf_final.py` | `f43b9d01fbdec6439ed1514e9b894045` | `m4conf/m4conf_final.py` | unchanged |
| `m4conf_followup.py` | `86587857f5ea8c5982816ca4b7a4e295` | `m4conf/m4conf_followup.py` | unchanged |
| `m4conf_followup2.py` | `bf7cd015d4ccd9f918faa44c4c8ba5ae` | `m4conf/m4conf_followup2.py` | unchanged |
| `m4conf_pairflip.py` | `1a504138acf80f8de38866d438d7db4e` | `m4conf/m4conf_pairflip.py` | unchanged |
| `m4conf_plots.py` | `f80b34eb7afdb7edc56c43e1864dee04` | - | unchanged |
| `m4conf_rawprobe.py` | `b911cc0d15998244215f77ba158397b1` | `m4conf/m4conf_rawprobe.py` | unchanged |
| `m4conf_rawstats.py` | `642ed75b1580e778838de34d4ab455cf` | `m4conf/m4conf_rawstats.py` | unchanged |
| `m4conf_summary.py` | `10b86bbbcae91e129b8bc3ef751689c9` | - | unchanged |
| `r_build_summary.py` | `c4089eecd226371f13a93b7db6153dc5` | - | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `r_spot2.py` | `74e6bac4de9cca0bfbb521895f3915e5` | `r/r_spot2.py` | path plumbing: 4 line(s) replaced, 3 import line(s) added |
| `r_spotcheck.py` | `25e804c3bc5909a1e2dd288e8a515a72` | `r/r_spotcheck.py` | path plumbing: 7 line(s) replaced, 3 import line(s) added |
| `s1local_survey_numbers.py` | `46f3e655e344ed5bc2fc3e8827bbff6b` | - | path plumbing: 3 line(s) replaced, 3 import line(s) added |
| `s2cluster_cnn_err_vs_label.py` | `397efc1420341fa0231af13ba32b76a6` | `s2cluster/s2cluster_cnn_err_vs_label.py` | path plumbing: 5 line(s) replaced, 3 import line(s) added |
| `s2cluster_label_stats.py` | `dff9319da17d9d367ccba48a6154363b` | `s2cluster/s2cluster_label_stats.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `s2cluster_recompute_perp.py` | `098903b3dec200ced0231f844c6de374` | `s2cluster/s2cluster_recompute_perp.py` | path plumbing: 4 line(s) replaced, 3 import line(s) added |
| `s2cluster_xtalk_probe.py` | `861c1e94ee4d7615baf4f89c68af40bb` | `s2cluster/s2cluster_xtalk_probe.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `s3lit_eccentric_p1_leakage_dem_check.py` | `e0a4b71042913d4237669eaebf223cae` | - | unchanged |
| `s3lit_thermal_estimate.py` | `957f0c19b06a086ea34fa1012e1f80e9` | - | unchanged |
| `vphysics_circle.py` | `e000aaf0e43084ebb66083cd000be00f` | - | unchanged |
| `vphysics_circle2.py` | `c948f97ed6a6b6bf1f3dfeb9df4ec22e` | - | unchanged |
| `vphysics_dispersive_locus.py` | `79c7fbf1a87176f5a345a8210fc2d88a` | - | unchanged |
| `vphysics_implied_level.py` | `ae66ee603b2d238013b094b5c7a9d069` | - | unchanged |
| `vphysics_plot.py` | `6035f278570b8dd17e0faa55d99148d1` | - | unchanged |
| `vphysics_raw.py` | `7f8c067935b50dfcb84957f0812d84c5` | `vphysics/scripts/vphysics_raw.py` | path plumbing: 6 line(s) replaced, 3 import line(s) added |
| `vphysics_raw2.py` | `1f2aa06ba5871ee783693be6420b5c55` | `vphysics/scripts/vphysics_raw2.py` | path plumbing: 6 line(s) replaced, 3 import line(s) added |
| `vphysics_ringup.py` | `f0a3a9806fe29d7bc180418dd7abbb75` | - | unchanged |
| `vreproduction_demod.py` | `9990b4a10b4e73914d644fcbe9b77d2c` | `vreproduction/scripts/vreproduction_demod.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `vreproduction_estimate.py` | `d7f49056332e66cffba1cb901526785a` | `vreproduction/scripts/vreproduction_estimate.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `vreproduction_followup.py` | `a4dce65e57dbc796deb48caeee011754` | `vreproduction/scripts/vreproduction_followup.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `vreproduction_followup2.py` | `afc9168e50fccefeed6abf8ec641e48d` | `vreproduction/scripts/vreproduction_followup2.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `vreproduction_followup3.py` | `28803be8c8b335766087a1f8e747068c` | `vreproduction/scripts/vreproduction_followup3.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `vstatistics_line.py` | `3d267dc9a4332c0890debb16f3cabf67` | `vstatistics/vstatistics_line.py` | path plumbing: 4 line(s) replaced, 3 import line(s) added |
| `vstatistics_main.py` | `ab4a1308f1f3706924bc15bb4b2c6b7b` | `vstatistics/vstatistics_main.py` | path plumbing: 4 line(s) replaced, 3 import line(s) added |
| `x_build_features.py` | `2b51418ae3e9bf07ffb1b0266cb88f15` | `scripts/x_build_features.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `x_calib.py` | `3ba376ea9112e2fe0e958a6d1f520a95` | `scripts/x_calib.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `x_colleague_prep_verbatim.py` | `c0e0a6ffcdf73f9ebd2a293b8b6959b9` | `scripts/x_colleague_prep_verbatim.py` | unchanged |
| `x_common.py` | `e0cc426e3e0c7dc7122cf7752c4a6274` | `scripts/x_common.py` | path plumbing: 2 line(s) replaced, 3 import line(s) added |
| `x_crosscheck.py` | `986a0b95e0e3c96308dca5d7783203ab` | `scripts/x_crosscheck.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `x_plot_iq.py` | `cfb1b07675f323b4fae5fb543eb723fb` | `scripts/x_plot_iq.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `x_sanity.py` | `8bc9950dbb01c9958241cc6de10d40fa` | `scripts/x_sanity.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |
| `x_xtalk_diag.py` | `fa3de2a7a93b6a24e2456d0cb948347e` | `scripts/x_xtalk_diag.py` | path plumbing: 1 line(s) replaced, 3 import line(s) added |

## leakage/iterations/ (unchanged)

| file | origin | md5 |
|---|---|---|
| `em_e1hmm_attrib.py` | local `sims/leakage/em_e1hmm_attrib.py`; cluster: `em/e1hmm/scripts/em_e1hmm_attrib.py` | `383b2256763d8aae577ea211be2fee8e` |
| `em_e1hmm_attrib_v2.py` | local `sims/leakage/em_e1hmm_attrib_v2.py`; cluster: `em/e1hmm/scripts/em_e1hmm_attrib_v2.py` | `345abff349818af41fbdc3e3db6bd2e2` |
| `em_e1hmm_attrib_v5.py` | local `sims/leakage/em_e1hmm_attrib_v5.py`; cluster: `em/e1hmm/scripts/em_e1hmm_attrib_v5.py` | `35a1f4965873705cabe7f7a203e816cf` |
| `em_e2seg_attr.py` | local `sims/leakage/em_e2seg_attr.py`; cluster: - | `fff18629a553e512ef116689d933c236` |
| `em_e2seg_explore.py` | **cluster only**: `~/oraqle-ics-sims/leakage/em/e2seg/scripts/em_e2seg_explore.py` | `234882ced55338579372ef91d58111b5` |
| `em_e3herq_assemble.py` | local `sims/leakage/em_e3herq_assemble.py`; cluster: - | `3e66fa2b6eead1db273b6f257b63bb68` |
| `em_e3herq_assemble_v2.py` | local `sims/leakage/em_e3herq_assemble_v2.py`; cluster: - | `aaafe490ef2ffe8f2c417965770a920a` |
| `em_e5vstat_coredebug.py` | local `sims/leakage/em_e5vstat_coredebug.py`; cluster: `em/e5vstat/scripts/em_e5vstat_coredebug.py` | `21b08a524646edb01e295ae5a066a19b` |
| `m1tail_analysis.py` | local `sims/leakage/m1tail_analysis.py`; cluster: `m1tail/scripts/m1tail_analysis.py` | `5ad15236f3860b45965ee5c26b901453` |
| `m1tail_plot.py` | local `sims/leakage/m1tail_plot.py`; cluster: - | `fe2579898e44e35301bad1514d88b067` |
| `m2mix_core.py` | local `sims/leakage/m2mix_core.py`; cluster: `m2mix/scripts/m2mix_core.py` | `25890ecf839111724ccba9e4ea50dda4` |
| `m2mix_fit.py` | local `sims/leakage/m2mix_fit.py`; cluster: `m2mix/scripts/m2mix_fit.py` | `6be2d3875a6ac5da7325f40ae0a692eb` |
| `m2mix_smoke2.py` | **cluster only**: `~/oraqle-ics-sims/leakage/m2mix/scripts/m2mix_smoke2.py` | `469d9e5bc2e838af39bfb440f9af89c1` |
| `m3time_examples.py` | local `sims/leakage/m3time_examples.py`; cluster: `m3time/scripts/m3time_examples.py` | `7a016b5a61db0d02835878362a2be1b6` |
| `m3time_plot.py` | local `sims/leakage/m3time_plot.py`; cluster: - | `489be3348699d4128c6e0f5159a12771` |

## Other files in leakage/

| file | origin | md5 of original | change |
|---|---|---|---|
| `vendor/networks/HybridCNN.py` | cluster only: `em/e2seg/scripts/networks/HybridCNN.py` (= graham `~/Oraqle/Discriminators/networks/HybridCNN.py` = `discriminators/networks/HybridCNN.py`) | `3f51902f65186753c075cce04b072018` | unchanged; vendored because `em_e2seg_oldmethod.py` imports `networks.HybridCNN` without the full `networks/__init__.py` |
| `vendor/networks/__init__.py` | cluster only: `em/e2seg/scripts/networks/__init__.py` (empty) | `d41d8cd98f00b204e9800998ecf8427e` | unchanged |
| `docker/Dockerfile` | graham `~/oraqle-ics-sims/docker-torch/Dockerfile` | `31e91a05fc2d70a641abc72dd9125323` | comment header added (pip freeze of the image); build lines unchanged |
| `docs/LEAKAGE_REPORT.md` | local `oraqle-ics/data/leakage/LEAKAGE_REPORT.md` | `a40f35d9c4fe8d600b3c3defc6e8d984` | a location note (old -> new paths) inserted after the title; body unchanged |
| `README.md`, `PROVENANCE.md`, `iterations/README.md` | new | - | - |
| (cluster) `features/README.md` | `~/oraqle-ics-sims/leakage/features/README.md` | `676547b742d3ed10afc05b731e38b9fa` | identical to `results/leakage/x_features_README.md`; not duplicated |

## results/leakage/

All files are byte copies of `oraqle-ics/data/leakage/<same name>` (md5 checked after copying), except `paper_values.json` and `README.md` (new). Not copied: `em_e3herq_preds_{1us,600ns,400ns}.npz` (> 50 MB; regeneration in `results/leakage/README.md`) and `LEAKAGE_REPORT.md` (moved to `leakage/docs/`).

| file | md5 (= original) |
|---|---|
| `em_e1hmm_results.json` | `b000fba6bba3345a1895066f41f5488b` |
| `em_e1hmm_summary.md` | `1a5968705091c82d08b0e93fa956a9a7` |
| `em_e2seg_oldmethod.json` | `57f42787cc05e9add80d183f015733d8` |
| `em_e2seg_oldmethod_l200scan.json` | `ed06fca6052ba0f4e341f2fedf9f02bb` |
| `em_e2seg_results.json` | `98756c5b108fdafa9fcdf0b781e9c2bb` |
| `em_e2seg_summary.md` | `a530aca83664946277afe6c20c393853` |
| `em_e2seg_v2_raw_main.json` | `a0b73103f4c4beb5786f6a19fbe40f9f` |
| `em_e2seg_v3_raw_main.json` | `38cb846da694d5d1df32e8765f7a1538` |
| `em_e3herq_summary_v3.json` | `d98b780d62555aa7dcfd684bf28daee6` |
| `em_e4recon_analysis.json` | `7d5194784327f0380c8cc1d91ca5adb0` |
| `em_e4recon_final.json` | `1c4897dd402786e5636e2062c0042ff3` |
| `em_e4recon_final.md` | `db5804413a18bde55989e7c7eadd0422` |
| `em_e5vrepro_diag_partners.json` | `d285ef945b3890c865ac1c6b3035485b` |
| `em_e5vrepro_diag_q3relax.json` | `f0ddefc85dd17047b707769336d0e3e3` |
| `em_e5vrepro_diag_xtalk.json` | `b73e7a760ae424e46338dfbd014f86e0` |
| `em_e5vrepro_if2.json` | `f1b38b45524ceb888b73bbb1575cb492` |
| `em_e5vrepro_raw_v2.json` | `6ec79d50a39464211b93a93b6f5b2191` |
| `em_e5vrepro_results.json` | `2712bb64f144e4cb41e3a6af1654d34e` |
| `em_e5vstat_check.json` | `67e3696a399c41c6580aa70ed04e8a39` |
| `em_e5vstat_lda.json` | `5aa7e4e8f396d98f95f511738d775aa3` |
| `em_e5vstat_q3core.json` | `288430f98986febc59c87473cdb53c0c` |
| `em_e5vstat_summary.json` | `e1514f7e21f85dd3829748dbcec52678` |
| `em_e6edit_final_numbers.json` | `546fd133e85e11192c9240f4758621b3` |
| `em_e6edit_final_numbers.md` | `eb414b5430bcf9bc2984292ef5568a56` |
| `em_e8fix_final_numbers.json` | `8493003153ddbc3258c363be747cba89` |
| `f_cnn_leak_xtab.json` | `99213b5a530f01fd657d921a4e069744` |
| `m1tail_q0.png` | `a2d580d69137790fff94c2f6cd7239a2` |
| `m1tail_q1.png` | `9342e13cacd2e662849754807f504ea2` |
| `m1tail_q2.png` | `f8a0d383bca2f2f39ae127115cf78381` |
| `m1tail_q3.png` | `9c5cbc14312182674a48fb1a497203cd` |
| `m1tail_q4.png` | `ae224ad2a02560e17f214346da38980b` |
| `m1tail_summary.json` | `24eff1105386a6a12dbf6b5c3dda03ba` |
| `m1tail_time.png` | `a5ac1f96c15870da457a8d5622e288b8` |
| `m1tail_traj.png` | `023013562d348a200a99776523fcff12` |
| `m1tail_v2_cooc_test.json` | `41b8f57a999f99deae98d71b104e42c7` |
| `m1tail_v2_models_test.json` | `d1ffb2073d66eccf5f78eb7472bd289b` |
| `m1tail_v2_percfg_test.json` | `9a2aae67c09f0f2127097e4cc24401e8` |
| `m1tail_v2_results_test.json` | `67f4b55e5e1d33408ea7cdd15bb16008` |
| `m1tail_v2_results_train.json` | `add15f347ae9a7957113d5318e898609` |
| `m1tail_v2_sliding_test.json` | `67f8e0b302b76cac8e2b55cbb1872811` |
| `m1tail_v2_store_test.npz` | `e48eca137e402fffb0d488df26b484e3` |
| `m1tail_v2_xtab_test.json` | `a0f4300437281c2f41ca95eb74764679` |
| `m2mix_coinc2.json` | `f3aea03cdef672115c5593bc657a67e4` |
| `m2mix_feature_table.csv` | `23ce5c3dc7216183a3b523e1f15b6657` |
| `m2mix_feature_table.json` | `1e1fed43e98547eade3066cc9a55090a` |
| `m2mix_fit2/m2mix_fit2_test_q0_1us.json` | `a5e09028292bf6f84082774998dd8a44` |
| `m2mix_fit2/m2mix_fit2_test_q0_2us.json` | `c3ecff413c10c46a501b3b510d7f64ea` |
| `m2mix_fit2/m2mix_fit2_test_q0_early.json` | `28a85b31fbcb50af97f115d2283960dc` |
| `m2mix_fit2/m2mix_fit2_test_q1_1us.json` | `7b57c85c63f327a76701d128d43a07ae` |
| `m2mix_fit2/m2mix_fit2_test_q1_2us.json` | `608f4334847776f82267e5f7368cfa48` |
| `m2mix_fit2/m2mix_fit2_test_q1_early.json` | `b09dea669c3fe01a2e0e494a6b41dfb7` |
| `m2mix_fit2/m2mix_fit2_test_q2_1us.json` | `6f1498b47358053e3f0410f01d4fb73d` |
| `m2mix_fit2/m2mix_fit2_test_q2_2us.json` | `4722a19f9d99b7e5d8fd36f04e8c8bc7` |
| `m2mix_fit2/m2mix_fit2_test_q2_early.json` | `02e71f5c8a43bd34d0d179ce51f67302` |
| `m2mix_fit2/m2mix_fit2_test_q3_1us.json` | `e0b061423efe25f909d02ed287472fd1` |
| `m2mix_fit2/m2mix_fit2_test_q3_2us.json` | `7d6431320458e31f68c9d4165704d0e5` |
| `m2mix_fit2/m2mix_fit2_test_q3_early.json` | `dffa74fbd1bb4e617ccef7ddef0911a0` |
| `m2mix_fit2/m2mix_fit2_test_q4_1us.json` | `c5c5816694196c8f0c0b49f6b82bc728` |
| `m2mix_fit2/m2mix_fit2_test_q4_2us.json` | `d6268aacc189204f9c13349178a23bd7` |
| `m2mix_fit2/m2mix_fit2_test_q4_early.json` | `8a49643abb5089a78a0058618e3981a1` |
| `m2mix_fit2/m2mix_fit2_train_q0_1us.json` | `ecd52ff4c2dc788e95eaab4e2b1a2605` |
| `m2mix_fit2/m2mix_fit2_train_q0_2us.json` | `014624c4bcee4813e2bbc1b2052197ab` |
| `m2mix_fit2/m2mix_fit2_train_q0_early.json` | `52554f412ece387cc2f1a5ef22e4053b` |
| `m2mix_fit2/m2mix_fit2_train_q1_1us.json` | `60b629b3cac22176153c8d46e4af8b22` |
| `m2mix_fit2/m2mix_fit2_train_q1_2us.json` | `64bac6ec1c3ecf232ee2a8930af4b11d` |
| `m2mix_fit2/m2mix_fit2_train_q1_early.json` | `1e26b342ff005d76f06041f98ec57596` |
| `m2mix_fit2/m2mix_fit2_train_q2_1us.json` | `3fd38f208271ca70f943708e3d605421` |
| `m2mix_fit2/m2mix_fit2_train_q2_2us.json` | `4d64796ac7afd40d6e13237178ba2ce9` |
| `m2mix_fit2/m2mix_fit2_train_q2_early.json` | `19f4fb88626082a4c59a8b4461b7f7e8` |
| `m2mix_fit2/m2mix_fit2_train_q3_1us.json` | `e486a8dd8a21d3d13dcef5b737fb1136` |
| `m2mix_fit2/m2mix_fit2_train_q3_2us.json` | `14b48c125ca7064c35fb29a9aeca9e29` |
| `m2mix_fit2/m2mix_fit2_train_q3_early.json` | `5a7593bb39dbb7347ca83ded29deec0d` |
| `m2mix_fit2/m2mix_fit2_train_q4_1us.json` | `d0d09ced769d3143ccfafa174976c1fb` |
| `m2mix_fit2/m2mix_fit2_train_q4_2us.json` | `f7aae17574fc9d209235a6f5511e493f` |
| `m2mix_fit2/m2mix_fit2_train_q4_early.json` | `aca9f632eb2cc8063ee217886517b977` |
| `m2mix_q0.png` | `319026ea2d42958abb6002521b113dca` |
| `m2mix_q1.png` | `d42d852a524ef27aea1e2ea993928f4b` |
| `m2mix_q2.png` | `cc701e94c47a0c2533c2dd3858b8fd12` |
| `m2mix_q3.png` | `89965923630ceb29509c3ff46b9320a5` |
| `m2mix_q4.png` | `6480af7085027973e8ac0394cc934f91` |
| `m2mix_summary.csv` | `645049ddd8d4e16964198b0aacf75be2` |
| `m2mix_summary.json` | `66bf9e70950bd0948f72466aad0b45e8` |
| `m2mix_traj2_summary.json` | `a21ad0930aa13a4061c3de051424f85e` |
| `m3time_exchange.png` | `1d909d147e6a8751d209de442f83468f` |
| `m3time_joint_feat_xt0.json` | `2f3e42fa7207e06c13335e72abad9d7a` |
| `m3time_joint_feat_xt1.json` | `1b04f028f32f3fe08d20c3df007b2ea5` |
| `m3time_joint_featbox_xt1.json` | `da40c1f0c8bc28e19343f03e64de07e2` |
| `m3time_occupancy.png` | `a6f1cff06acbba9d74a6492c440a24c3` |
| `m3time_q0_examples.png` | `ab25c49eae0f34ca3260357abf3c6881` |
| `m3time_q1_examples.png` | `0b80d8774aa9e4ce96b5692ff055fa01` |
| `m3time_q2_examples.png` | `e65f8f4e6df6d5fe6cd5971fe033a17f` |
| `m3time_q3_examples.png` | `fbcd604d6b6f2d795cae81fd2e5dee0f` |
| `m3time_q4_examples.png` | `7ca70de2a6ca38c39bfdb8c1558e7e4e` |
| `m3time_re_cooccurrence_feat_xt0.json` | `65813da5fe73eee9d0ad9ea99db249db` |
| `m3time_re_cooccurrence_feat_xt1.json` | `ce791925f5e377b362f5ed5072c9ed86` |
| `m3time_re_cooccurrence_featbox_xt1.json` | `c2f5494fc17d4333bdf57c9dabff02fe` |
| `m3time_re_examples_raw.npz` | `b87a21943dc4c2439b277d32fed1c090` |
| `m3time_re_feat_xt0_q0.json` | `612ecb187183b5e9e2bff41afe4401a2` |
| `m3time_re_feat_xt0_q1.json` | `2fcc8bac1928f20c25c7416a10a6a550` |
| `m3time_re_feat_xt0_q2.json` | `9b9e226c4267f31d1701c1acbc611f05` |
| `m3time_re_feat_xt0_q3.json` | `4e13457f22193b273c34cb0ce28a3e6d` |
| `m3time_re_feat_xt0_q4.json` | `6d47642931925fca3b34abbe57b8822f` |
| `m3time_re_feat_xt1_q0.json` | `522f8ca137ee00daec57f54d7e1844d1` |
| `m3time_re_feat_xt1_q1.json` | `1ead5b381f276d6752314a9f820c4e7b` |
| `m3time_re_feat_xt1_q2.json` | `4969e030173dcf8cebc89fc4dfdeb14c` |
| `m3time_re_feat_xt1_q3.json` | `9099582aadaa16838e1ac8c21d6d857b` |
| `m3time_re_feat_xt1_q4.json` | `e0898df6902a4e848a944233e6505b37` |
| `m3time_re_featbox_xt1_q0.json` | `41b37f926de3a9499fc6dd4aaa7ba478` |
| `m3time_re_featbox_xt1_q1.json` | `cdd72bbcae436fc74afa3c5283b366d9` |
| `m3time_re_featbox_xt1_q2.json` | `0b525e3b87d31958acc13b4c8bd9b6b6` |
| `m3time_re_featbox_xt1_q3.json` | `946667643677870f663740a820e5efb1` |
| `m3time_re_featbox_xt1_q4.json` | `1aa59aa2241a0283358115621c0f90f4` |
| `m3time_summary.json` | `72b28db5983af76e5174b7a56f073b18` |
| `m3time_survival.png` | `e85914992fb0cb0a594eb50b46d0a928` |
| `m3time_xchk2_feat_xt1.json` | `b3b3d7632f320f33593fa2f35c03161d` |
| `m3time_xchk_feat_xt1.json` | `e7d71bb67ea2c1a2e6deaa91a3652130` |
| `m4conf_dccheck.json` | `8042882306afacae847e5b7813d7ee08` |
| `m4conf_final.json` | `d1470c3ae937b1298becb8e4490f54fc` |
| `m4conf_followup.json` | `eb168674e1e805b6692f57712267fdaa` |
| `m4conf_followup2.json` | `62ff547dbef022d84fc00c12914fd350` |
| `m4conf_funnel.png` | `54b919ae2c34e60aff724a9d54648570` |
| `m4conf_overlap.png` | `865bacc9363ca4a14842aad9a6d0a5e3` |
| `m4conf_pairflip.json` | `d43db53729515f4fa0df4b519febf9fa` |
| `m4conf_q5q3.png` | `2929faafe723d3ca381cd3f41a932f8a` |
| `m4conf_rawprobe.json` | `d1a79ee69b477bff2828e31d1c8bb918` |
| `m4conf_rawprobe.png` | `99c5ce178cd15a82966be7aaca0fbe3d` |
| `m4conf_results.json` | `3fbc4053e7c86b94f8c045c5391db0dc` |
| `m4conf_simultaneity.png` | `f161500a3427e5b21aa0263e757ace15` |
| `m4conf_summary.json` | `e4a85c3abb6c57d2a096e3a9d8bc7f5f` |
| `m4conf_time.png` | `fe0cb23782a19cdcc1c16e9cdcc46b7a` |
| `m4conf_xtalk.png` | `e1b239a83f820ecef6aee619035f3732` |
| `r_leakage_summary.json` | `d93b6f10869319256bb12fa5f1088676` |
| `r_leakage_summary.md` | `68b118ef6f0fd3f1c40d765896459acd` |
| `r_spot2.json` | `879202b81f605ea9ab84368a7b4d7d19` |
| `r_spotcheck.json` | `59ebb5fcb88ac196b57097528fc83918` |
| `s1local_existing_leakage_numbers.csv` | `9099abec2e0f86d1401770b75fc7b1f9` |
| `s1local_q5_gallery_offaxis.json` | `b5e5c17997a854bddbba04258050276a` |
| `s2cluster_cnn_err_vs_label.json` | `57912c5b64ac746aa9b4257a83412187` |
| `s2cluster_label_stats.json` | `d5127c7db9fe8f8341833efe3cd46094` |
| `s2cluster_recompute_perp.json` | `6e9d20f44bac66e3e035eb40b6e599a8` |
| `s2cluster_xtalk_probe.json` | `7eae9f5c028fbb543ea00e8c1af5f402` |
| `vphysics_circle.json` | `fca6869f7c2720ee96498488e8c93ac9` |
| `vphysics_circle.png` | `189611f988c03935955cd51fd68eabfd` |
| `vphysics_circle2.json` | `ec4f93e07458700b29709afe2725847e` |
| `vphysics_classmeans.npz` | `fd7e05a0a9d44137fa41fdc844b57170` |
| `vphysics_classmeans2.npz` | `804dff2bcecc53edc90ef3701c8bf21f` |
| `vphysics_dispersive_locus.json` | `b1c5e98139aa240d7cf111c9ed057de8` |
| `vphysics_freqcheck.json` | `1b70c016e0ccc5e3ff6e6fad40fd0114` |
| `vphysics_implied_level.json` | `fe0dd3118dee2ca7e99754d6bfce97b8` |
| `vphysics_raw_classes.json` | `841b00f5f7c62513ac5996359e5671d4` |
| `vphysics_raw_classes2.json` | `c308c423b7c816ee21930a446acc7414` |
| `vphysics_ringup.json` | `a0887003907288f57e080415f5288b27` |
| `vreproduction_calib.json` | `ae1ad243615c572338a1c1f586005d8d` |
| `vreproduction_followup.json` | `7d4bd46915e632598fef4dc2b788b5d2` |
| `vreproduction_followup2.json` | `8f640cb90e6184f30c596cf11690762e` |
| `vreproduction_followup3.json` | `f03f25e2f66556e9c45db391e9ef2f8a` |
| `vreproduction_results.json` | `1098926f26740f5fb7c82635dbac6b04` |
| `vstatistics_line.json` | `efd0b0e5dc8e7a7390e9e1748819dd09` |
| `vstatistics_main.json` | `1613d2db50433a415695cd25478595d1` |
| `x_build_meta.json` | `fe9fc7c38b7c0a4ae3e78e64c321348e` |
| `x_calib.json` | `d851a007e2696dd7a51dcb559f1b749e` |
| `x_crosscheck.json` | `1845e65abf2bba3c366c969a031b7d49` |
| `x_features_README.md` | `676547b742d3ed10afc05b731e38b9fa` |
| `x_hist.npz` | `76fc89c9d3f2541b9f4ff7efaf0202b1` |
| `x_iq_q0.png` | `4df33b2001f7caa46ad58b9b4934bc8e` |
| `x_iq_q1.png` | `b1d249d3e6c837f0e0fb8c0758e2404e` |
| `x_iq_q2.png` | `5b65e5efd2d987a616571dc3c87d11d4` |
| `x_iq_q3.png` | `e23612875ad8f74ee67f2cef046c3560` |
| `x_iq_q4.png` | `d1a5e68d9636a60532032921fe3ed844` |
| `x_sanity.json` | `cd8c39e9433e27e97da88af32f8eb972` |
| `x_xtalk.json` | `230dc51a131e7e37359b603311940156` |
| ~~`em_e3herq_preds_1us.npz`~~ (not committed) | `eef537b8f784107d4970446625c862a4` |
| ~~`em_e3herq_preds_600ns.npz`~~ (not committed) | `c418cd15aa2e1a5f7e7eecd0c96a2898` |
| ~~`em_e3herq_preds_400ns.npz`~~ (not committed) | `1c717adb1dec7eb65c75c97aabac411e` |

## paper_figures/leakage/

| file | origin |
|---|---|
| `make_error_modes_table.py`, `make_iq_taxonomy.py` | new (the table had no generator; the figure wrapper repeats `em_e6edit_iq_gallery.py` without its absolute paths and refuse-to-overwrite guard) |
| `reference/iq_error_gallery_q5_v2.pdf` | `oraqle-ics/plots/iq_error_gallery_q5_v2.pdf` (untracked there), md5 `c240f180f7071bdd2e65b56acda129af` |
| `reference/tab_error_modes.tex` | `oraqle-ics/chapters/ml_discriminators.tex` lines 392-412 (2026-11-05 working tree) |
