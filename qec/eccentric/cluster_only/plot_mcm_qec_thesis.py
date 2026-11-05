import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants
WIDE_FIGSIZE = 6
HEIGHT_FIGSIZE = 2
FONTSIZE = 12
BAR_WIDTH = 0.2
group_spacing = 0.4
code_palette = sns.color_palette("pastel", n_colors=8)

def _latency_col(df):
    return 'mcm_latency_ns' if 'mcm_latency_ns' in df.columns else 'measure_latency_ns'


def print_fig_comparison(df_mcm, df_herq, label):
    """Print how MCMit compares to HERQULES for one panel (e.g. (a) current,
    (b) futuristic).

    1) Same readout length: % by which MCMit's LER is lower than HERQULES',
       per (distance, duration) and averaged across distances per duration.
    2) MCMit best LER (lowest, expected at 500 ns) vs MCMit at 1000 ns,
       per distance and averaged across distances.
    """
    lat_col = _latency_col(df_mcm)
    distances = sorted(df_mcm['distance'].unique())
    durations = sorted(df_mcm[lat_col].unique())

    print(f"\n=== Figure {label}: MCMit vs HERQULES ===")
    print("\n[1] Same readout length (MCMit improvement over HERQULES):")
    for dur in durations:
        improvements = []
        for d in distances:
            mcm_row = df_mcm[(df_mcm['distance'] == d) & (df_mcm[lat_col] == dur)]
            herq_row = df_herq[(df_herq['distance'] == d) & (df_herq[lat_col] == dur)]
            if mcm_row.empty or herq_row.empty:
                continue
            mcm_ler = mcm_row['logical_error_rate'].values[0]
            herq_ler = herq_row['logical_error_rate'].values[0]
            if herq_ler == 0:
                print(f"  {int(dur)} ns, d={int(d)}: "
                      f"MCMit={mcm_ler:.5f} vs HERQULES={herq_ler:.5f} "
                      f"-> N/A (HERQULES LER is 0)")
                continue
            improvement = (herq_ler - mcm_ler) / herq_ler * 100
            improvements.append(improvement)
            print(f"  {int(dur)} ns, d={int(d)}: "
                  f"MCMit={mcm_ler:.5f} vs HERQULES={herq_ler:.5f} "
                  f"-> {improvement:.2f}% lower")
        if improvements:
            print(f"  {int(dur)} ns, AVG across {len(improvements)} distances: "
                  f"{np.mean(improvements):.2f}% lower")

    print("\n[2] MCMit best LER vs MCMit at 1000 ns:")
    improvements = []
    for d in distances:
        mcm_d = df_mcm[df_mcm['distance'] == d]
        best_row = mcm_d.loc[mcm_d['logical_error_rate'].idxmin()]
        best_ler = best_row['logical_error_rate']
        best_dur = int(best_row[lat_col])
        row_1000 = mcm_d[mcm_d[lat_col] == 1000]
        if row_1000.empty:
            continue
        ler_1000 = row_1000['logical_error_rate'].values[0]
        if ler_1000 == 0:
            print(f"  d={int(d)}: best={best_ler:.5f} @ {best_dur} ns "
                  f"vs 1000 ns={ler_1000:.5f} -> N/A (1000 ns LER is 0)")
            continue
        improvement = (ler_1000 - best_ler) / ler_1000 * 100
        improvements.append(improvement)
        print(f"  d={int(d)}: best={best_ler:.5f} @ {best_dur} ns "
              f"vs 1000 ns={ler_1000:.5f} -> {improvement:.2f}% lower")
    if improvements:
        print(f"  AVG across {len(improvements)} distances: "
              f"{np.mean(improvements):.2f}% lower")


def plot_four_panel(single_csv, patch_csv, mcm_error_csv):

    # Read CSVs
    df_all = pd.read_csv(single_csv)
    if 'mcm_latency_ns' in df_all.columns:
        df_all = df_all.sort_values(by=['distance', 'mcm_latency_ns'])
    elif 'measure_latency_ns' in df_all.columns:
        df_all = df_all.sort_values(by=['distance', 'measure_latency_ns'])
    df_mcm_a = df_all[(df_all['system'] == 'MCMit') & (df_all['noise_model'] == 'current')].copy()
    df_herq_a = df_all[(df_all['system'] == 'HERQULES') & (df_all['noise_model'] == 'current')].copy()
    df_mcm_b = df_all[(df_all['system'] == 'MCMit') & (df_all['noise_model'] == 'futuristic')].copy()
    df_herq_b = df_all[(df_all['system'] == 'HERQULES') & (df_all['noise_model'] == 'futuristic')].copy()
    df_patch = pd.read_csv(patch_csv)
    df_mcm_err = pd.read_csv(mcm_error_csv)

    for df in [df_mcm_a, df_herq_a, df_mcm_b, df_herq_b, df_patch, df_mcm_err]:
        df['logical_error_rate'] = pd.to_numeric(df['logical_error_rate'])

    print_fig_comparison(df_mcm_a, df_herq_a, "(a) current")
    print_fig_comparison(df_mcm_b, df_herq_b, "(b) futuristic")

    fig, axes = plt.subplots(
        1, 4,
        figsize=(WIDE_FIGSIZE * 2.8, HEIGHT_FIGSIZE * 1.3)
    )

    LOWER_TEXT_Y = 1.25

    # ==================================================
    # (a) Comparison 1 (MCMit vs HERQULES)
    # ==================================================
    distances = sorted(df_mcm_a['distance'].unique())

    for i, d in enumerate(distances):
        subset = df_mcm_a[df_mcm_a['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[0].plot(
            x_vals,
            subset['logical_error_rate'].replace(0, np.nan),
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"MCMit d={d}"
        )

        subset = df_herq_a[df_herq_a['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[0].plot(
            x_vals,
            subset['logical_error_rate'].replace(0, np.nan),
            '--s',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"HERQULES d={d}"
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Duration (ns)", fontsize=FONTSIZE)
    axes[0].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[0].set_title("(a) IBM Heron", fontsize=FONTSIZE, fontweight="bold")
    axes[0].grid(color='gray')
    axes[0].set_ylim(bottom=0, top=0.7)  # Set y-limits for better visualization

    # X-ticks for plot a
    durations_a = sorted(set(
        df_mcm_a.get('mcm_latency_ns', df_mcm_a.get('measure_latency_ns')).tolist() +
        df_herq_a.get('mcm_latency_ns', df_herq_a.get('measure_latency_ns')).tolist()
    ))
    axes[0].set_xticks(durations_a)
    axes[0].set_xticklabels([str(int(d)) for d in durations_a])

    # ==================================================
    # (b) Comparison 2 (MCMit vs HERQULES)
    # ==================================================
    distances = sorted(df_mcm_b['distance'].unique())

    for i, d in enumerate(distances):
        subset = df_mcm_b[df_mcm_b['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[1].plot(
            x_vals,
            subset['logical_error_rate'].replace(0, np.nan),
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"MCMit d={d}"
        )

        subset = df_herq_b[df_herq_b['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[1].plot(
            x_vals,
            subset['logical_error_rate'].replace(0, np.nan),
            '--s',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"HERQULES d={d}"
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Duration (ns)", fontsize=FONTSIZE)
    axes[1].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[1].set_title("(b) Futuristic noise model", fontsize=FONTSIZE, fontweight="bold")
    axes[1].grid(color='gray')

    # X-ticks for plot b
    durations_b = sorted(set(
        df_mcm_b.get('mcm_latency_ns', df_mcm_b.get('measure_latency_ns')).tolist() +
        df_herq_b.get('mcm_latency_ns', df_herq_b.get('measure_latency_ns')).tolist()
    ))
    axes[1].set_xticks(durations_b)
    axes[1].set_xticklabels([str(int(d)) for d in durations_b])

    # ==================================================
    # (c) Patches vs duration (MCMit)
    # ==================================================
    patch_vals = sorted(df_patch['total_patches'].unique())
    distances = sorted(df_patch['distance'].unique())

    for i, dist in enumerate(distances):
        y_vals = []
        for n in patch_vals:
            subset = df_patch[
                (df_patch['total_patches'] == n) &
                (df_patch['distance'] == dist)
            ]
            y_vals.append(
                subset['logical_error_rate'].values[0]
                if not subset.empty else np.nan
            )

        axes[2].plot(
            patch_vals,
            y_vals,
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"d = {int(dist)}"
        )

    axes[2].set_yscale("log")
    axes[2].set_xlabel("Number of QEC patches", fontsize=FONTSIZE)
    axes[2].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[2].set_title("(c) QEC patches", fontsize=FONTSIZE, fontweight="bold")
    axes[2].grid(color='gray')

    # X-ticks for plot c
    axes[2].set_xticks(patch_vals)
    axes[2].set_xticklabels([str(int(p)) for p in patch_vals])
    axes[2].legend(fontsize=FONTSIZE-2,
                   loc="center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.46),
    frameon=False)
    # ==================================================
    # (d) MCM error vs latency (MCMit)
    # ==================================================
    x_vals = sorted(df_mcm_err['measure_error'].unique())
    latencies = sorted(df_mcm_err['measure_latency_ns'].unique())

    for i, lat in enumerate(latencies):
        y_vals = []
        for xm in x_vals:
            subset = df_mcm_err[
                (df_mcm_err['measure_error'] == xm) &
                (df_mcm_err['measure_latency_ns'] == lat)
            ]
            y_vals.append(subset['logical_error_rate'].values[0] if not subset.empty else np.nan)

        axes[3].plot(
            x_vals,
            y_vals,
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"{int(lat)} ns"
        )

    axes[3].set_yscale("log")
    axes[3].set_xscale("log")
    axes[3].set_xlabel("MCM error probability", fontsize=FONTSIZE)
    axes[3].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[3].set_title("(d) QEC threshold", fontsize=FONTSIZE, fontweight="bold")
    axes[3].grid(color='gray')
    axes[3].text(2.55, 1.25, "Lower is better ↓",
                 transform=axes[0].transAxes,
                 fontsize=FONTSIZE, fontweight="bold",
                 color="blue", va="top", ha="center")
    xmin, xmax = axes[3].get_xlim()
    ymin, ymax = axes[3].get_ylim()

    # Determine overlap range
    low = max(xmin, ymin)
    high = min(xmax, ymax)

    # Generate diagonal line
    x_diag = np.logspace(np.log10(low), np.log10(high), 200)
    axes[3].plot(
        x_diag,
        x_diag,
        linestyle='--',
        linewidth=2,
        color='black',
        label='Threshold'
    )
    axes[3].legend(fontsize=FONTSIZE-2,
                   loc="center",
    ncol=3,
    bbox_to_anchor=(0.48, -0.55),
    frameon=False
                   )

    # ==================================================
    # Legends between plots
    # ==================================================
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    #handles_c, labels_c = axes[3].get_legend_handles_labels()

    # Legend between (a) and (b)
    fig.legend(
        handles_a, labels_a,
        loc="center",
        ncol=3,
        bbox_to_anchor=(0.28, 0.08),
        fontsize=FONTSIZE - 2,
        frameon=False
    )

    # Legend between (c) and (d)
    #fig.legend(
    #    handles_c, labels_c,
    #    loc="center",
    #    ncol=2,
    #    bbox_to_anchor=(0.78, 0.08),
    #    fontsize=FONTSIZE - 2,
    #    frameon=False
    #)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('gray')  # change border color to gray
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.35, top=0.85, wspace=0.35)
    fig.patch.set_linewidth(3)
    plt.savefig("mcm_qec_evaluation.pdf", format="pdf")
    plt.close(fig)

def plot_thesis_qec_panels(single_csv, patch_csv, mcm_error_csv):
    # Read CSVs
    df_all = pd.read_csv(single_csv)
    if 'mcm_latency_ns' in df_all.columns:
        df_all = df_all.sort_values(by=['distance', 'mcm_latency_ns'])
    elif 'measure_latency_ns' in df_all.columns:
        df_all = df_all.sort_values(by=['distance', 'measure_latency_ns'])
    df_mcm_a = df_all[(df_all['system'] == 'MCMit') & (df_all['noise_model'] == 'current')].copy()
    df_herq_a = df_all[(df_all['system'] == 'HERQULES') & (df_all['noise_model'] == 'current')].copy()
    df_mcm_b = df_all[(df_all['system'] == 'MCMit') & (df_all['noise_model'] == 'futuristic')].copy()
    df_herq_b = df_all[(df_all['system'] == 'HERQULES') & (df_all['noise_model'] == 'futuristic')].copy()
    df_patch = pd.read_csv(patch_csv)
    df_mcm_err = pd.read_csv(mcm_error_csv)

    for df in [df_mcm_a, df_herq_a, df_mcm_b, df_herq_b, df_patch, df_mcm_err]:
        df['logical_error_rate'] = pd.to_numeric(df['logical_error_rate'])

    print_fig_comparison(df_mcm_a, df_herq_a, "(a) current")
    print_fig_comparison(df_mcm_b, df_herq_b, "(b) futuristic")

    TEXT_WIDTH = 10
    THESIS_SIZE = (TEXT_WIDTH, TEXT_WIDTH * 0.4)

    fig, axes = plt.subplots(
        1, 2,
        figsize=THESIS_SIZE
    )
    fig.subplots_adjust(wspace=0.25)

    # ==================================================
    # (a) Comparison 1 (MCMit vs HERQULES)
    # ==================================================
    distances = sorted(df_mcm_a['distance'].unique())

    for i, d in enumerate(distances):
        subset = df_mcm_a[df_mcm_a['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[0].plot(
            x_vals,
            subset['logical_error_rate'].replace(0, np.nan),
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"MCMit d={d}"
        )

        subset = df_herq_a[df_herq_a['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[0].plot(
            x_vals,
            subset['logical_error_rate'].replace(0, np.nan),
            '--s',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"HERQULES d={d}"
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Duration (ns)", fontsize=FONTSIZE)
    axes[0].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[0].set_title("(a) IBM Heron", fontsize=FONTSIZE, fontweight="bold")
    axes[0].grid(color='gray')

    # X-ticks for plot a
    durations_a = sorted(set(
        df_mcm_a.get('mcm_latency_ns', df_mcm_a.get('measure_latency_ns')).tolist() +
        df_herq_a.get('mcm_latency_ns', df_herq_a.get('measure_latency_ns')).tolist()
    ))
    axes[0].set_xticks(durations_a)
    axes[0].set_xticklabels([str(int(d)) for d in durations_a])
    fig.text(0.5, 0.95, "Lower is better ↓", ha="center",
                 fontsize=FONTSIZE, fontweight="bold",
                 color="blue")

    # ==================================================
    # (b) Comparison 2 (MCMit vs HERQULES)
    # ==================================================
    distances = sorted(df_mcm_b['distance'].unique())

    for i, d in enumerate(distances):
        subset = df_mcm_b[df_mcm_b['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[1].plot(
            x_vals,
            subset['logical_error_rate'].replace(0, np.nan),
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"MCMit d={d}"
        )

        subset = df_herq_b[df_herq_b['distance'] == d]
        x_vals = subset.get('mcm_latency_ns', subset.get('measure_latency_ns'))
        axes[1].plot(
            x_vals,
            subset['logical_error_rate'].replace(0, np.nan),
            '--s',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"HERQULES d={d}"
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Duration (ns)", fontsize=FONTSIZE)
    axes[1].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[1].set_title("(b) Futuristic noise model", fontsize=FONTSIZE, fontweight="bold")
    axes[1].grid(color='gray')

    # X-ticks for plot b
    durations_b = sorted(set(
        df_mcm_b.get('mcm_latency_ns', df_mcm_b.get('measure_latency_ns')).tolist() +
        df_herq_b.get('mcm_latency_ns', df_herq_b.get('measure_latency_ns')).tolist()
    ))
    axes[1].set_xticks(durations_b)
    axes[1].set_xticklabels([str(int(d)) for d in durations_b])

    handles_a, labels_a = axes[0].get_legend_handles_labels()
    fig.legend(
        handles_a, labels_a,
        loc="center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.1),
        fontsize=FONTSIZE - 2,
        frameon=False
    )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('gray')  # change border color to gray
    #plt.subplots_adjust(left=0.06, right=0.99, bottom=0.35, top=0.85, wspace=0.35)
    fig.patch.set_linewidth(3)
    plt.savefig("thesis_mcm_qec_evaluation1.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    TEXT_WIDTH = 10
    THESIS_SIZE = (TEXT_WIDTH, TEXT_WIDTH * 0.4)

    fig, axes = plt.subplots(
        1, 2,
        figsize=THESIS_SIZE
    )


    patch_vals = sorted(df_patch['total_patches'].unique())
    distances = sorted(df_patch['distance'].unique())

    for i, dist in enumerate(distances):
        y_vals = []
        for n in patch_vals:
            subset = df_patch[
                (df_patch['total_patches'] == n) &
                (df_patch['distance'] == dist)
            ]
            y_vals.append(
                subset['logical_error_rate'].values[0]
                if not subset.empty else np.nan
            )

        axes[0].plot(
            patch_vals,
            y_vals,
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"d = {int(dist)}"
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of QEC patches", fontsize=FONTSIZE)
    axes[0].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[0].set_title("(a) QEC patches", fontsize=FONTSIZE, fontweight="bold")
    axes[0].grid(color='gray')

    # X-ticks for plot c
    axes[0].set_xticks(patch_vals)
    axes[0].set_xticklabels([str(int(p)) for p in patch_vals])
    axes[0].legend(fontsize=FONTSIZE-2,
                   loc="center",
                   ncol=3,
                   bbox_to_anchor=(0.5, -0.3),
                   frameon=False)
    # ==================================================
    # (d) MCM error vs latency (MCMit)
    # ==================================================
    x_vals = sorted(df_mcm_err['measure_error'].unique())
    latencies = sorted(df_mcm_err['measure_latency_ns'].unique())

    for i, lat in enumerate(latencies):
        y_vals = []
        for xm in x_vals:
            subset = df_mcm_err[
                (df_mcm_err['measure_error'] == xm) &
                (df_mcm_err['measure_latency_ns'] == lat)
            ]
            y_vals.append(subset['logical_error_rate'].values[0] if not subset.empty else np.nan)

        axes[1].plot(
            x_vals,
            y_vals,
            '-o',
            color=code_palette[i],
            markeredgecolor='black',
            markersize=4,
            label=f"{int(lat)} ns"
        )

    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("MCM error probability", fontsize=FONTSIZE)
    axes[1].set_ylabel("Log. err. rate (log)", fontsize=FONTSIZE)
    axes[1].set_title("(b) QEC threshold", fontsize=FONTSIZE, fontweight="bold")
    axes[1].grid(color='gray')
    fig.text(0.5, 0.95, "Lower is better ↓", ha="center",
                 fontsize=FONTSIZE, fontweight="bold",
                 color="blue")

    xmin, xmax = axes[1].get_xlim()
    ymin, ymax = axes[1].get_ylim()

    # Determine overlap range
    low = max(xmin, ymin)
    high = min(xmax, ymax)

    # Generate diagonal line
    x_diag = np.logspace(np.log10(low), np.log10(high), 200)
    axes[1].plot(
        x_diag,
        x_diag,
        linestyle='--',
        linewidth=2,
        color='black',
        label='Threshold'
    )
    axes[1].legend(fontsize=FONTSIZE-2,
                   loc="center",
    ncol=3,
    bbox_to_anchor=(0.48, -0.3),
    frameon=False)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('gray')  # change border color to gray
    #plt.subplots_adjust(left=0.06, right=0.99, bottom=0.35, top=0.85, wspace=0.35)
    fig.patch.set_linewidth(3)
    plt.savefig("thesis_mcm_qec_evaluation2.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


import matplotlib
matplotlib.use("Agg")

plot_four_panel("results/mcm_latency_error_ler.csv", "results/mcm_patch_dist.csv", "results/mcm_error_latency_ler.csv")
plot_thesis_qec_panels("results/mcm_latency_error_ler.csv", "results/mcm_patch_dist.csv", "results/mcm_error_latency_ler.csv")
