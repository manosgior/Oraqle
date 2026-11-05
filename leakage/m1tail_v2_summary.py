"""m1tail_v2_summary.py (agent tag m1tail): collect the key per-qubit numbers of the v2 off-axis tail
analysis into data/leakage/m1tail_summary.json. usage: python m1tail_v2_summary.py <data/leakage dir>"""
import json, os, sys
import numpy as np
from scipy.special import ndtr
D = sys.argv[1]
L = lambda f: json.load(open(os.path.join(D, f)))
R, RT, M, P, C, X, SL = (L("m1tail_v2_results_test.json"), L("m1tail_v2_results_train.json"), L("m1tail_v2_models_test.json"),
                         L("m1tail_v2_percfg_test.json"), L("m1tail_v2_cooc_test.json"), L("m1tail_v2_xtab_test.json"),
                         L("m1tail_v2_sliding_test.json"))
PAPER = ["Q1", "Q2", "Q3", "Q4", "Q5"]
WIN = {"early": "raw samples 0-99 (first 200 ns; FIR sees to 149)", "1us": "raw samples 0-499 (paper '1 us', strictly causal FIR = CNN input)",
       "2us": "raw samples 0-999 (full stored 2 us)"}
out = dict(tag="m1tail", split="test (train replicates within CI)", N_per_prepared_bit=560000,
           definitions=dict(
               z_b="(perp - mu_s)/sigma_noise; mu_s = truncated-Gaussian perp centre of the full prepared 5-bit state s (prepared bit x 16 neighbour configs); sigma_noise = perp width of the prepared-|0> state with the same neighbours (H0: state-independent amplifier noise)",
               E3="mirror excess [N(z_b>3) - N(z_b<-3)]/N: one-sided off-axis excess beyond 3 sigma, needs only a symmetric core; 95% CI = +-1.96 sqrt(N+ + N-)/N",
               gauss_excess_k="N(z_b>k) - N*Phi(-k) per side",
               dseg="2-D distance (noise sigma) of the window mean to the c0-c1 segment; H0 bound N*exp(-k^2/2)",
               frame="c0 = KDE mode of prepared-|0> state, c1 = fixed-width EM (c0 fixed, isotropic noise sigma, flat background), rotation = median per-config c0->c1 angle"),
           qubits={})
for q in range(5):
    Q = dict(paper=PAPER[q], code=f"q{q}", windows={})
    for w in ("early", "1us", "2us"):
        Rq = R["windows"][w][f"q{q}"]
        W = dict(samples=WIN[w], sep_raw=Rq["frame"]["sep_mode_median"], sep_over_noise_sigma=Rq["frame"]["sep_over_sigma_median"],
                 sigma_perp_prep1_over_prep0_median=float(np.median(Rq["frame"]["sigma_perp_prep1_over_prep0"])))
        cnt = dict(p=0, m=0, N=0)
        for b in (0, 1):
            RB = Rq[f"prep{b}"]; N = RB["N"]; t = RB["tails_zb"]
            cnt["p"] += t["3"]["plus"]["obs"]; cnt["m"] += t["3"]["minus"]["obs"]; cnt["N"] += N
            tr = RT["windows"][w][f"q{q}"][f"prep{b}"]["tails_zb"]["3"]
            mod = M[f"{w}_q{q}_b{b}"]["zb"]
            fe = RB["mode_census"]["far_excess"]
            W[f"prep{b}"] = dict(
                N=N, E3=t["3"]["mirror_excess_frac"], E3_ci95=t["3"]["mirror_excess_frac_ci95"],
                E3_train=tr["mirror_excess_frac"], E3_train_ci95=tr["mirror_excess_frac_ci95"],
                E2=t["2"]["mirror_excess_frac"], E4=t["4"]["mirror_excess_frac"], E5=t["5"]["mirror_excess_frac"],
                counts_plus_minus={k: [t[k]["plus"]["obs"], t[k]["minus"]["obs"], round(t[k]["plus"]["exp_gauss"], 3)] for k in ("3", "4", "5", "6", "8")},
                dseg5=dict(obs=RB["dseg"]["5"]["obs"], H0_bound=RB["dseg"]["5"]["rayleigh_bound"],
                           perp_only_null_lenient=RB["dseg"]["5"]["null_lenient_resampled"], frac=RB["dseg"]["5"]["frac"],
                           frac_ci95=RB["dseg"]["5"]["frac_ci95"], side=RB["dseg5_side"]),
                student_t_df=mod["student_t"]["df"], dBIC_t_minus_gauss=mod["dBIC_t_minus_gauss"], best_model_by_bic=mod["best_by_bic"],
                gmm2_minor=[mod["gmm2_minor_weight"], mod["gmm2_minor_mean"]],
                far_excess_mode_minus_cut={k: (round(v["mode_minus_cut"], 2) if "mode_minus_cut" in v else None) for k, v in fe.items()},
                far_excess_mode_uv={k: v.get("mode_uv") for k, v in fe.items()},
                far_excess_centroid_uv_cut5=fe.get("5", {}).get("centroid_uv"),
                noncore_modes=[dict(uv_sep=md["uv_sep_units"], xy_sigma=[md["x_ms"], md["y_ms"]], basin_count=md["basin_count"],
                                    excess_frac_lenient=md["excess_frac_lenient"], excess_frac_lenient_ci95=md["excess_frac_lenient_ci95"],
                                    excess_frac_strict=md["excess_frac_strict"], z_dip=md.get("z_dip"), ratio=md.get("ratio"),
                                    per_cfg_loc_sd_sep=md.get("per_cfg_loc_sd_sep"), per_cfg_frac_rel_sd=md["per_cfg_frac_rel_sd"],
                                    per_cfg_chi2_p=md["per_cfg_chi2_p"], frac_cfg_bootstrap95=md["frac_cfg_bootstrap95"],
                                    other_prep=md.get("other_prep"))
                               for md in RB["mode_census"]["modes"] if not (md["is_core0"] or md["is_core1"])],
                percfg=dict(E3_min=P[f"{w}_q{q}"][f"prep{b}"]["E3_cfg_min"], E3_max=P[f"{w}_q{q}"][f"prep{b}"]["E3_cfg_max"],
                            chi2_p=P[f"{w}_q{q}"][f"prep{b}"]["E3_cfg_chi2_p"],
                            offaxis_centroid_uv_mean=P[f"{w}_q{q}"][f"prep{b}"]["centroid_uv_cfg_mean"],
                            offaxis_centroid_uv_sd=P[f"{w}_q{q}"][f"prep{b}"]["centroid_uv_cfg_sd"],
                            E3_by_other_qubit_bit=P[f"{w}_q{q}"][f"prep{b}"]["by_other_qubit_bit"]),
                midpoint_threshold_errors=RB["midpoint_threshold_errors"])
        se = np.sqrt(cnt["p"] + cnt["m"]) / cnt["N"]
        e = (cnt["p"] - cnt["m"]) / cnt["N"]
        W["pooled"] = dict(N=cnt["N"], E3=e, E3_ci95=[e - 1.96 * se, e + 1.96 * se])
        Q["windows"][w] = W
    Q["sliding_500ns_E3"] = {f"prep{b}": dict(window_centre_us=[(s * 100 + 250) / 1000 for s in SL[f"q{q}"]["start_bin"]],
                                              E3=SL[f"q{q}"][f"prep{b}"]["mirror3"]) for b in (0, 1)}
    if q != 2:
        Q["xtab_vs_Q3_both_prep1"] = {w: {k: dict(frac_of_shots=v["frac_of_shots"], P_target_offaxis=v["P_target_offaxis"],
                                                  mirror_E3_target=v["mirror_E3_target"])
                                          for k, v in X[w][f"t{q}_s2_srcprep1"].items() if k.startswith("src_")} for w in ("1us", "2us")}
    out["qubits"][PAPER[q]] = Q
json.dump(out, open(os.path.join(D, "m1tail_summary.json"), "w"), indent=1, default=float)
for q in range(5):
    W = out["qubits"][PAPER[q]]["windows"]["1us"]
    print(PAPER[q], " ".join(f"b{b}: E3={100*W[f'prep{b}']['E3']:.3f}% [{100*W[f'prep{b}']['E3_ci95'][0]:.3f},{100*W[f'prep{b}']['E3_ci95'][1]:.3f}]" for b in (0, 1)),
          f"pooled {100*W['pooled']['E3']:.3f}% [{100*W['pooled']['E3_ci95'][0]:.3f},{100*W['pooled']['E3_ci95'][1]:.3f}]")
