#!/usr/bin/env python3
"""Extract the paper's discriminator/readout numbers into results/discriminators/paper_values.json.
[unify, new file]

Run once against the paper source (oraqle-ics checkout); verify/discriminators.py then
checks every entry from results/ without needing the paper:

    python discriminators/tables/extract_paper_values.py --paper-root ../oraqle-ics

For each value it records the chapter:line where it is printed, the printed text, the
metric that recomputes it from results/discriminators/, and the data value
(`expected`). If the printed value disagrees with the data, the entry becomes a
"known_discrepancy" and MUST carry an explanatory note from KNOWN_NOTES below;
otherwise the script stops, so no disagreement is recorded silently. Paper typos are
never "fixed": expected is always the data value.

Scope: abstract, intro, background, motivation, overview, qubit_readout,
ml_discriminators (without tab:error_modes / fig:iq_taxonomy / Sec. 5.3, which the
leakage part covers) and the HERQULES appendix tables.
"""
import argparse
import importlib.util
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "discriminators", "tables"))
import build_table2 as BT  # noqa: E402

spec = importlib.util.spec_from_file_location("verify_disc", os.path.join(REPO, "verify", "discriminators.py"))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

QR, ML, OV, AP = ("chapters/qubit_readout.tex", "chapters/ml_discriminators.tex", "chapters/overview.tex",
                  "chapters/appendix_herqules.tex")
FMT3 = {"mode": "fmt", "decimals": 3}

# ------------------------------------------------------------------ prose values
# (id, chapter, paper_text (substring of the line), paper_value, metric, compare[, line hint])
PROSE = [
    # ---- Sec. 4: fig:readout_sweeps caption
    ("readout/caption/250ns_window_peak", QR, r"peaks at $\sim$0.86 fidelity", 0.86, "win_peak:250:val", {"mode": "tol", "tol": 0.01}),
    ("readout/caption/250ns_window_peak_center", QR, r"near a $0.3$--$0.4\,\mu$s center", [0.3, 0.4], "win_peak:250:center", {"mode": "within"}),
    ("readout/caption/500ns_window_peak", QR, r"($\sim$0.89 peak)", 0.89, "win_peak:500:val", {"mode": "tol", "tol": 0.01}),
    ("readout/caption/info_gain_peak_t", QR, r"peaks at $t\!\approx\!0.3\,\mu$s (dotted)", 0.3, "dfdt_peak_t_all", {"mode": "eq"}),
    ("readout/caption/info_gain_turns_negative", QR, r"turns negative beyond $t\!\approx\!1\,\mu$s", [1.0, 1.0], "dfdt_zero_range", {"mode": "tol", "tol": 0.1}),
    # ---- Sec. 4.1 results / analysis
    ("readout/4.1/sweep_range_end", QR, r"$T \in [0.1, 2.0]\,\mu$s in $0.1\,\mu$s steps", 2.0, "trunc_last_t", {"mode": "eq"}),
    ("readout/4.1/gmean_at_0.1us", QR, r"$\sim0.60$ at $T=0.1\,\mu$s", 0.60, "trunc_g:0.1", {"mode": "tol", "tol": 0.01}),
    ("readout/4.1/gmean_at_0.3us", QR, r"$\sim0.86$ at $0.3\,\mu$s", 0.86, "trunc_g:0.3", {"mode": "tol", "tol": 0.01}),
    ("readout/4.1/gmean_at_0.5us", QR, r"$\sim0.88$ at $0.5\,\mu$s", 0.88, "trunc_g:0.5", {"mode": "tol", "tol": 0.01}),
    ("readout/4.1/Q1_at_1.0us", QR, r"$\sim0.97$", 0.97, "trunc_q:1:1.0", {"mode": "tol", "tol": 0.01}),
    ("readout/4.1/Q2_at_1.0us", QR, r"$\sim0.73$", 0.73, "trunc_q:2:1.0", {"mode": "tol", "tol": 0.01}),
    ("readout/4.1/Q4_at_1.0us", QR, r"$\sim0.94$", 0.94, "trunc_q:4:1.0", {"mode": "tol", "tol": 0.01}),
    ("readout/4.1/Q3_peak_position", QR, r"peak around $0.7$--$1.0\,\mu$s", [0.7, 1.0], "trunc_peak:3:t", {"mode": "within"}),
    ("readout/4.1/Q5_peak_position", QR, r"peak around $0.7$--$1.0\,\mu$s", [0.7, 1.0], "trunc_peak:5:t", {"mode": "within"}),
    ("readout/4.1/Q3_peak_value", QR, r"$\sim0.90$", 0.90, "trunc_peak:3:val", {"mode": "tol", "tol": 0.01}),
    ("readout/4.1/Q5_peak_value", QR, r"$\sim0.96$", 0.96, "trunc_peak:5:val", {"mode": "tol", "tol": 0.01}),
    ("readout/4.1/Q3Q5_relative_drop", QR, r"relative drop of up to $\sim$40\%", 0.40, "trunc_reldrop_max:3,5", {"mode": "tol", "tol": 0.05}),
    ("readout/4.1/info_gain_peak_all_qubits", QR, r"peaks at $T = 0.3\,\mu$s on all five qubits", 0.3, "dfdt_peak_t_all", {"mode": "eq"}),
    ("readout/4.1/info_gain_zero_crossing", QR, r"crosses zero between $T \approx 0.8$ and $1.0\,\mu$s", [0.8, 1.0], "dfdt_zero_range", {"mode": "tol", "tol": 0.05}),
    # ---- Sec. 4.2
    ("readout/4.2/peak_center_250", QR, r"centered around $0.3$--$0.4\,\mu$s", [0.3, 0.4], "win_peak:250:center", {"mode": "within"}),
    ("readout/4.2/peak_center_500", QR, r"centered around $0.3$--$0.4\,\mu$s", [0.3, 0.4], "win_peak:500:center", {"mode": "within"}),
    ("readout/4.2/250ns_peak", QR, r"$\sim$0.86 at the peak", 0.86, "win_peak:250:val", {"mode": "tol", "tol": 0.01}),
    ("readout/4.2/250ns_at_1.8us", QR, r"$\sim$0.77 at a $1.8\,\mu$s center", 0.77, "win_at:250:1.8", {"mode": "tol", "tol": 0.01}),
    ("readout/4.2/250ns_relative_drop", QR, r"$\sim$39\% relative drop", 0.39, "win_reldrop:250:1.8", {"mode": "tol", "tol": 0.015}),
    ("readout/4.2/500ns_ceiling", QR, r"higher ceiling ($\sim$0.89)", 0.89, "win_peak:500:val", {"mode": "tol", "tol": 0.01}),
    ("readout/4.2/Q2_weakest_250", QR, r"Q2 weakest) at every position", True, "win_q2_weakest_all:250", {"mode": "eq"}),
    ("readout/4.2/Q2_weakest_500", QR, r"Q2 weakest) at every position", True, "win_q2_weakest_all:500", {"mode": "eq"}),
    ("readout/4.2/Q1Q5_strongest_250", QR, r"(Q1 and Q5 strongest", 1.0, "win_q1q5_top2_frac:250", {"mode": "tol", "tol": 1e-9}),
    ("readout/4.2/Q1Q5_strongest_500", QR, r"(Q1 and Q5 strongest", 1.0, "win_q1q5_top2_frac:500", {"mode": "tol", "tol": 1e-9}),
    ("readout/4.2/gap_250ns_vs_plateau", QR, r"peaks $\sim$21\% (relative) below", 0.21, "win_gap_vs_plateau:250", {"mode": "tol", "tol": 0.015}),
    ("readout/4.2/truncation_plateau", QR, r"$\sim$0.89 plateau", 0.89, "trunc_plateau", {"mode": "tol", "tol": 0.01}),
    ("readout/4.2/500ns_never_exceeds_plateau", QR, r"even the $500\,$ns window never exceeds it", True, "win_below_plateau:500", {"mode": "eq"}),
    # ---- Sec. 5.1 (ml_discriminators.tex:85, :91, Takeaway #3)
    ("ml/5.1/Q2_range_1000ns", ML, r"$\sim$0.73--0.75 regardless of architecture", [0.73, 0.75], "range:Q2:1000:all6", {"mode": "tol", "tol": 0.005}),
    ("ml/5.1/F5Q_neural_band_1000ns", ML, r"($F_{5Q}=0.901$--$0.911$)", [0.901, 0.911], "range:F5Q:1000:neural", FMT3),
    ("ml/5.1/CNN_F4Q_1000ns", ML, r"leads on $F_{4Q}$ ($0.957$)", 0.957, "master:MCMit-CNN:1000:F4Q", FMT3),
    ("ml/5.1/F4Q_leader_1000ns", ML, r"leads on $F_{4Q}$ ($0.957$)", "MCMit-CNN", "argmax_model:F4Q:1000", {"mode": "eq"}),
    ("ml/5.1/CNN_params", ML, r"at $\sim$3k parameters", 3000, "master:MCMit-CNN:1000:params", {"mode": "tol", "tol": 500}),
    ("ml/5.1/QubiCML_F5Q_1000ns", ML, r"\textit{QubiCML} ($0.893$)", 0.893, "master:QubiCML:1000:F5Q", FMT3),
    ("ml/5.1/linear_F5Q_1000ns", ML, r"($F_{5Q}=0.892$, Table~\ref{tab:fidelity_linear})", 0.892, "master:Linear Threshold:1000:F5Q", FMT3),
    ("ml/5.1/F5Q_neural_band_600ns", ML, r"0.894$--$0.901", [0.894, 0.901], "range:F5Q:600:neural", FMT3),
    ("ml/5.1/error_increase_1000_to_600", ML, r"every model loses only $\sim$10\%", 0.10, "err_increase:mean_all6:1000:600", {"mode": "tol", "tol": 0.02}),
    ("ml/5.1/F5Q_band_200ns", ML, r"$0.738$--$0.767$ band", [0.738, 0.767], "range:F5Q:200:noherq", FMT3),
    ("ml/5.1/HERQULES_Q2_200ns", ML, r"coin-flip $0.513$", 0.513, "master:HERQULES:200:Q2", FMT3),
    ("ml/takeaway3/error_increase", ML, r"costs only $\sim$10\% more per-shot error", 0.10, "err_increase:mean_all6:1000:600", {"mode": "tol", "tol": 0.02}),
    ("ml/takeaway3/Q2_range_400_1000ns", ML, r"stays at $\sim$0.69--0.75", [0.69, 0.75], "range:Q2:400,600,800,1000:all6", {"mode": "tol", "tol": 0.005}),
    # ---- Sec. 5.2
    ("ml/5.2/orders_of_magnitude", ML, r"span five orders of magnitude", 5, "orders_of_magnitude", {"mode": "tol", "tol": 0.5}),
    ("ml/5.2/linear_params", ML, r"$\sim$5-parameter Linear threshold", 5, "master:Linear Threshold:1000:params", {"mode": "eq"}),
    ("ml/5.2/MCMitT_params", ML, r"$\sim$5M-parameter MCMit-T", 5e6, "master:MCMit-T:1000:params", {"mode": "tol", "tol": 0.5e6}),
    ("ml/5.2/error_reduction_T_vs_linear", ML, r"only $\sim$18\% in $F_{5Q}$ error reduction", 0.18, "errred_vs:MCMit-T:Linear Threshold:1000", {"mode": "tol", "tol": 0.015}),
    ("ml/5.2/CNN_params", ML, r"MCMit-CNN ($\sim$3k params)", 3000, "master:MCMit-CNN:1000:params", {"mode": "tol", "tol": 500}),
    ("ml/5.2/CNN_F4Q", ML, r"leads the suite on $F_{4Q}$ ($0.957$)", 0.957, "master:MCMit-CNN:1000:F4Q", FMT3),
    ("ml/5.2/HERQULES_params", ML, r"HERQULES ($\sim$1k)", 1000, "master:HERQULES:1000:params", {"mode": "tol", "tol": 100}),
    ("ml/5.2/giants_weight_ratio", ML, r"extra $100$--$1000\times$ weights", [100, 1000], "param_ratios_giants", {"mode": "orders"}),
    ("ml/takeaway4/thousand_times_larger", ML, r"models a thousand times larger", 1000, "param_ratio:MCMit-T:MCMit-CNN:1000", {"mode": "orders"}),
    ("ml/5.2/fpga_latency", ML, r"($\leq74$\,ns)", 74, "fpga_max_latency", {"mode": "eq"}),
    ("ml/5.2/fpga_klinq_share", ML, r"($\sim$70\% of LUTs/DSPs)", [70, 70], "fpga_pct_pair:KLiNQ:LUT,DSP", {"mode": "tol", "tol": 1.5}),
    ("ml/5.2/fpga_fnn_T_do_not_fit", ML, r"do not fit on the QubiC ZCU216 RFSoC at all", None, None, {"mode": "unverifiable"}),
    # ---- overview: tab:model_comparison and Stage 1 dataset
    ("overview/params/BaselineFNN", OV, r"$\sim 635$K", 635000, "master:Baseline FNN:1000:params", {"mode": "tol", "tol": 5000}),
    ("overview/params/HERQULES", OV, r"$\sim 1$K", 1000, "master:HERQULES:1000:params", {"mode": "tol", "tol": 100}),
    ("overview/params/QubiCML_per_qubit", OV, r"$\sim 65$ (per qubit)", 65, "params_div:QubiCML:1000:5", {"mode": "eq"}),
    ("overview/params/KLiNQ", OV, r"$\sim 9$K", 9000, "master:KLiNQ:1000:params", {"mode": "tol", "tol": 500}),
    ("overview/params/MCMitT_range", OV, r"$\sim 0.1$--$4.8$M", [0.1e6, 4.8e6], "params_range:MCMit-T", {"mode": "tol", "tol": 0.05e6}),
    ("overview/params/MCMitCNN", OV, r"$\sim 3$K", 3000, "master:MCMit-CNN:1000:params", {"mode": "tol", "tol": 500}),
    ("overview/params/MCMitT_min_at_800ns", OV, r"from $\sim$0.1M (800\,ns)", 800, "params_argmin_ns:MCMit-T", {"mode": "eq"}),
    ("overview/params/MCMitT_1us", OV, r"$\sim$4.8M (1\,$\mu$s", 4.8e6, "master:MCMit-T:1000:params", {"mode": "tol", "tol": 0.05e6}),
    ("overview/features/QubiCML_per_qubit", OV, r"from $\sim$65 per qubit for QubiCML", 65, "params_div:QubiCML:1000:5", {"mode": "eq"}),
    ("overview/features/MCMitT_max", OV, r"to $\sim$4.8M for the MCMit Transformer", 4.8e6, "master:MCMit-T:1000:params", {"mode": "tol", "tol": 0.05e6}),
    ("overview/dataset/shots", OV, r"$\sim$1.6\,M shots", 1.6e6, "dataset:shots_total", {"mode": "tol", "tol": 0.05e6}),
    ("overview/dataset/rate", OV, r"500\,MSamples/s", 500, "dataset:rate_MSps", {"mode": "eq"}),
    ("overview/dataset/window", OV, r"over a 2\,$\mu$s window", 2.0, "dataset:window_us", {"mode": "eq"}),
    ("overview/dataset/states", OV, r"$2^5 = 32$ states", 32, "dataset:n_states", {"mode": "eq"}),
    ("overview/dataset/traces_per_state", OV, r"50,000 traces each", 50000, "dataset:shots_per_state", {"mode": "eq"}),
    # ---- appendix: HERQULES protocol
    ("app/herqules/seed_variance", AP, r"variance is $\approx0.0002$", 0.0002, "herq_seed:std:F5Q", {"mode": "fmt", "decimals": 4}),
    ("app/herqules/mixed_gain_1us", AP, r"$\sim$21\% of the per-trace error", 0.21, "herq_errred:1000", {"mode": "tol", "tol": 0.01}),
    ("app/herqules/mixed_gain_below_600", AP, r"$\sim$31--35\% below", [0.31, 0.35], "herq_errred_range:200,400,600", {"mode": "tol", "tol": 0.006}),
    ("app/herqules/released_200ns", AP, r"($F_{5Q}=0.794$ vs.\ $0.702$", 0.794, "herq:demux_subsample:200:gmean5", FMT3),
    ("app/herqules/pertrace_200ns", AP, r"($F_{5Q}=0.794$ vs.\ $0.702$", 0.702, "herq:deployment_pertrace:200:gmean5", FMT3),
    ("app/noretrain/1us_F5Q", AP, r"($F_{5Q}=0.905$)", 0.905, "freeze:freeze:1000:gmean5", FMT3),
    ("app/noretrain/800ns_F5Q", AP, r"falling to $0.881$ at $800$\,ns", 0.881, "freeze:freeze:800:gmean5", FMT3),
    ("app/noretrain/600ns_F5Q", AP, r"$0.775$ at $600$\,ns", 0.775, "freeze:freeze:600:gmean5", FMT3),
    ("app/noretrain/400ns_F5Q", AP, r"$0.628$ at", 0.628, "freeze:freeze:400:gmean5", FMT3),
    ("app/noretrain/200ns_F5Q", AP, r"$0.504$ at $200$\,ns", 0.504, "freeze:freeze:200:gmean5", FMT3),
    ("app/noretrain/retrained_600", AP, r"$0.894 / 0.867 / 0.702$", 0.894, "master:HERQULES:600:F5Q", FMT3),
    ("app/noretrain/retrained_400", AP, r"$0.894 / 0.867 / 0.702$", 0.867, "master:HERQULES:400:F5Q", FMT3),
    ("app/noretrain/retrained_200", AP, r"$0.894 / 0.867 / 0.702$", 0.702, "master:HERQULES:200:F5Q", FMT3),
]

# Notes for every value where the paper text disagrees with the data (required).
KNOWN_NOTES = {
    "readout/caption/info_gain_turns_negative": "First negative dF/dT per qubit: Q3 0.7, Q5 0.9, Q4 1.0, Q1 1.4, Q2 1.7 us; 'beyond ~1 us' holds for Q3-Q5 only.",
    "readout/4.1/sweep_range_end": "The sweep stops at 950 samples = 1.9 us (baseline_no_ML.py caps T_END at T_max-1 = 999 samples); there is no 2.0 us point.",
    "readout/4.1/gmean_at_0.3us": "Geometric mean at 0.3 us is 0.824; ~0.86 is the 0.4 us value (0.8635).",
    "readout/4.1/Q3_peak_position": "Q3 peaks at 0.5 us (0.9116), not in 0.7-1.0 us.",
    "readout/4.1/Q5_peak_position": "Q5 peaks at 0.6 us (0.9655; 0.9649 at 0.7 us), just outside 0.7-1.0 us.",
    "readout/4.1/Q3_peak_value": "Q3 peak is 0.912 (at 0.5 us), not ~0.90.",
    "readout/4.1/Q3Q5_relative_drop": "With the paper's error-reduction metric (peak-last)/(1-last) to the last point (1.9 us): Q3 37%, Q5 50%; 'up to ~40%' understates Q5.",
    "readout/4.1/info_gain_zero_crossing": "First negative dF/dT per qubit: Q3 0.7, Q5 0.9, Q4 1.0, Q1 1.4, Q2 1.7 us; 'between 0.8 and 1.0' holds for Q4/Q5 only.",
    "readout/4.2/Q1Q5_strongest_250": "Q2 is weakest everywhere, but Q1 and Q5 are the top two at only this fraction of window positions (Q3 or Q4 is sometimes second).",
    "readout/4.2/Q1Q5_strongest_500": "Q2 is weakest everywhere, but Q1 and Q5 are the top two at only this fraction of window positions.",
    "readout/4.2/gap_250ns_vs_plateau": "21% follows from the rounded 0.89 and 0.86; with the unrounded truncation plateau 0.8931 (0.9 us) and 250 ns window peak 0.8579 it is 24.8%.",
    "ml/5.2/orders_of_magnitude": "5 parameters (Linear) to 4,773,408 (MCMit-T, 1000 ns) is 10^5.98, i.e. six orders of magnitude.",
    "app/noretrain/1us_F5Q": "The freeze run's own 1000 ns row gives F5Q 0.9045 (printed 0.905 = tab:fidelity's 0.9046).",
    "app/noretrain/1us_row": "The printed 1000 ns row is copied from tab:fidelity (HERQULES 1000 ns); the no-retrain (freeze) run's own 1000 ns row is Q1 0.9685, Q2 0.7401, Q3 0.9247, Q4 0.942, Q5 0.9697, F5Q 0.9045.",
    "tab:fidelity_linear/800/Q5": "Typo: the CSV value is 0.9633 (printed 0.936).",
    "tab:fidelity_linear/800/F4Q": "Follows from the Q5 typo: with Q5 = 0.9633, F4Q at 800 ns is 0.941 (and becomes the best F4Q, so the bold should move from 600 to 800 ns).",
}
UNVERIFIABLE_NOTE = ("No synthesis report exists for the Baseline FNN or MCMit-T (fpga/hls4ml/reports covers "
                     "QubiCML, HERQULES, MCMit CNN m8/m16 and KLiNQ only), so 'do not fit' cannot be checked.")
PROSE_NOTES = {   # notes on values that agree but need context
    "tab:fpga_resources/QubiCML": "5 x qubic.txt (one per-qubit model per qubit; checkpoint = Arxiv_q0_len500_trial3.pth, the Table 2 QubiCML Q1 model). Vitis HLS C-synthesis estimate, 2 ns clock.",
    "tab:fpga_resources/HERQULES": "herqules.txt (Net_rmf, 1002 params; checkpoint = HERQULES_len500.pth, the June-2 per-trace model, not the Table 2 sweep run, which saved no checkpoint). MF/RMF front-end not included.",
    "tab:fpga_resources/MCMit CNN": "cnn-m8-parallelizationfactors.txt: m_param=8 CNN with 957 parameters on ds40 (10,13) input (runners/_cnn_small_hls.py), NOT the 3061-parameter m16 model of tab:fidelity (that one, cnn-m16.txt, needs 65% LUT / 54% FF / 47% DSP, 198 ns).",
    "tab:fpga_resources/KLiNQ": "3 x klinq.txt (705-param student, = KLiNQ_student_paper_q0_1000ns.pth) + 2 x klinq-larger-reuse4.txt (3425-param student, = KLiNQ_student_paper_q2_1000ns.pth, ReuseFactor 4); latency = slowest instance (42 ns). The run script for the reuse-4 variant was not committed (fpga/README.md).",
    "ml/5.2/fpga_latency": "Slowest deployable row is MCMit CNN (37 cycles x 2 ns).",
    "ml/5.2/fpga_klinq_share": "KLiNQ composition: 70.2% LUT, 69.0% DSP.",
    "app/herqules/seed_variance": "The CSV value is the standard deviation of F5Q over 5 seeds (not a variance), measured with the released (demux) feature assembly (mean F5Q 0.9250), not with the per-trace model.",
    "app/herqules/mixed_gain_below_600": "Table rows 600/400/200 ns: 31.6%, 35.3%, 30.7%. The CSV also has 250 ns (40.8%) and 500 ns (33.5%), which the paper does not show.",
    "ml/5.1/error_increase_1000_to_600": "Mean over the six models; per model +3.9% (QubiCML) to +12.1% (MCMit-CNN).",
    "ml/takeaway3/error_increase": "Mean over the six models; per model +3.9% (QubiCML) to +12.1% (MCMit-CNN).",
    "ml/takeaway3/Q2_range_400_1000ns": "Holds for 400-1000 ns; at 200 ns Q2 is 0.513-0.648.",
    "ml/5.1/F5Q_neural_band_1000ns": "Neural = Baseline FNN, HERQULES, KLiNQ, MCMit-T, MCMit-CNN (QubiCML 0.893 is discussed separately).",
    "ml/5.2/giants_weight_ratio": "Baseline FNN / MCMit-CNN = 208x, MCMit-T / MCMit-CNN = 1559x (orders of magnitude 2 and 3).",
}


def find_line(lines, text, after=0):
    for i, l in enumerate(lines, start=1):
        if i <= after or l.lstrip().startswith("%"):
            continue
        if text in l:
            return i
    raise SystemExit(f"text not found: {text!r}")


def norm(v):
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, list):
        return [norm(x) for x in v]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-root", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "discriminators", "paper_values.json"))
    a = ap.parse_args()
    D = V.Data()
    entries, missing_notes = [], []

    def add(eid, ref, paper_text, paper_value, metric, compare, note=None):
        e = {"id": eid, "ref": ref, "paper_text": paper_text, "paper_value": paper_value, "metric": metric,
             "compare": compare}
        if compare["mode"] == "unverifiable":
            e.update(expected=None, status="unverifiable", note=UNVERIFIABLE_NOTE)
            entries.append(e)
            return
        val = D.metric(metric)
        e["expected"] = norm(val)
        cons = V.paper_consistent(e, val)
        e["status"] = "match" if cons else "known_discrepancy"
        n = KNOWN_NOTES.get(eid) if not cons else (note or PROSE_NOTES.get(eid))
        if not cons and not n:
            missing_notes.append(f"{eid} ({ref}): paper {paper_text!r} vs data {val!r}")
        if n:
            e["note"] = n
        entries.append(e)

    texts = {c: open(os.path.join(a.paper_root, c)).read().splitlines() for c in (QR, ML, OV, AP)}

    # ---- prose
    for eid, ch, ptxt, pval, metric, cmp_ in PROSE:
        add(eid, f"{ch}:{find_line(texts[ch], ptxt)}", ptxt, pval, metric, cmp_)

    # ---- tab:fidelity (180 cells)
    row_re = r"^(?:\\multirow\{5\}\{\*\}\{(?P<model>[^}]*?)(?:\\footnotemark)?\})?\s*&\s*(?P<t>\d+)\s*&(?P<rest>.*)$"
    cells = BT.parse_paper_table(os.path.join(a.paper_root, ML), "tab:fidelity", BT.TABLE2_MODELS, BT.COLS, row_re)
    for (m, t, c), (v, b, s, txt, line) in sorted(cells.items(), key=lambda kv: kv[1][4]):
        add(f"tab:fidelity/{m}/{t}/{c}", f"{ML}:{line}", txt, float(txt), f"master:{m}:{t}:{c}", FMT3)

    # ---- tab:fidelity_linear (35 cells)
    row_re = r"^(?P<model>)(?P<t>\d+)\\,ns\s*&(?P<rest>.*)$"
    cells = BT.parse_paper_table(os.path.join(a.paper_root, AP), "tab:fidelity_linear", ["Linear Threshold"],
                                 BT.COLS + ["F4Q"], row_re)
    for (_, t, c), (v, b, s, txt, line) in sorted(cells.items(), key=lambda kv: kv[1][4]):
        add(f"tab:fidelity_linear/{t}/{c}", f"{AP}:{line}", txt, float(txt), f"master:Linear Threshold:{t}:{c}", FMT3)

    # ---- tab:herqules-pertrace (20 cells): per-trace F5Q, F4Q | released F5Q, F4Q
    cols = [("deployment_pertrace", "gmean5"), ("deployment_pertrace", "gmean4_no_q2"),
            ("demux_subsample", "gmean5"), ("demux_subsample", "gmean4_no_q2")]
    cells = BT.parse_paper_table(os.path.join(a.paper_root, AP), "tab:herqules-pertrace", ["HERQULES"],
                                 ["c0", "c1", "c2", "c3"], row_re)
    for (_, t, c), (v, b, s, txt, line) in sorted(cells.items(), key=lambda kv: kv[1][4]):
        var, col = cols[int(c[1])]
        add(f"tab:herqules-pertrace/{t}/{var}/{col}", f"{AP}:{line}", txt, float(txt), f"herq:{var}:{t}:{col}", FMT3)

    # ---- tab:herqules-noretrain (30 cells) from the freeze run
    qcols = ["Q1", "Q2", "Q3", "Q4", "Q5", "gmean5"]
    cells = BT.parse_paper_table(os.path.join(a.paper_root, AP), "tab:herqules-noretrain", ["HERQULES"],
                                 qcols, row_re)
    for (_, t, c), (v, b, s, txt, line) in sorted(cells.items(), key=lambda kv: kv[1][4]):
        eid = f"tab:herqules-noretrain/{t}/{c}"
        if t == 1000:
            KNOWN_NOTES.setdefault(eid, KNOWN_NOTES["app/noretrain/1us_row"])
        add(eid, f"{AP}:{line}", txt, float(txt), f"freeze:freeze:{t}:{c}", FMT3)

    # ---- tab:fpga_resources (16 cells) from the Vitis HLS reports
    lines = texts[ML]
    start = find_line(lines, r"\label{tab:fpga_resources}")
    for i in range(start, start + 20):
        l = lines[i - 1]
        m = re.match(r"^\s*(QubiCML|HERQULES|MCMit CNN|KLiNQ)(?:\\textsuperscript\{\\(?:dag|ddag)\})?\s*&"
                     r"\s*(\d+)\\%\s*&\s*(\d+)\\%\s*&\s*(\d+)\\%\s*&\s*\$(\d+)\$\\,ns", l)
        if not m:
            continue
        row = m.group(1)
        for col, txt in zip(["LUT", "FF", "DSP"], m.group(2, 3, 4)):
            add(f"tab:fpga_resources/{row}/{col}", f"{ML}:{i}", txt + r"\%", int(txt), f"fpga:{row}:{col}",
                {"mode": "fmt", "decimals": 0},
                note=PROSE_NOTES.get(f"tab:fpga_resources/{row}"))
        add(f"tab:fpga_resources/{row}/latency_ns", f"{ML}:{i}", f"${m.group(5)}$", int(m.group(5)),
            f"fpga:{row}:latency_ns", {"mode": "eq"}, note=PROSE_NOTES.get(f"tab:fpga_resources/{row}"))
    if sum(e["id"].startswith("tab:fpga_resources/") for e in entries) != 16:
        raise SystemExit("tab:fpga_resources: expected 16 cells")

    if missing_notes:
        print("Disagreements without a note in KNOWN_NOTES (add one, do not change the data):")
        for m in missing_notes:
            print("  ", m)
        sys.exit(1)
    n_kd = sum(e["status"] == "known_discrepancy" for e in entries)
    out = {"_about": ("Paper values for verify/discriminators.py. expected = value recomputed from "
                      "results/discriminators/ (the data); paper_text/paper_value = as printed at ref. "
                      "status: match | known_discrepancy (paper text disagrees with the data; see note) | "
                      "unverifiable (no artifact). Generated by discriminators/tables/extract_paper_values.py "
                      "from the oraqle-ics working tree."),
           "paper_source": "oraqle-ics working tree, 2026-11-05 (git HEAD d7cd4e9 + uncommitted chapter edits)",
           "entries": entries}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(entries)} entries, {n_kd} known discrepancies, "
          f"{sum(e['status'] == 'unverifiable' for e in entries)} unverifiable")


if __name__ == "__main__":
    main()
