import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import sys  # [unify]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_results import fetch_results
from disc_paths import RESULTS_DISC, REPORTS_OUT  # [unify]


def _first_existing(candidates, fallback):
    """Return the first path that exists, else `fallback`. Lets the same code
    run in the container (e.g. /oraqle_reports) and on the host
    (/home/manosgior/oraqle_reports) without edits."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return fallback


# Input table: read from the oraqle_reports dir (container or host path).
MASTER_FIDELITY_CSV = os.environ.get(  # [unify] was /app/optimization_reports or ~/oraqle_reports
    "ORAQLE_MASTER_FIDELITY", str(RESULTS_DISC / "master_fidelity.csv"))

# Output: this is a report/figure (NOT a model), so it goes into the
# oraqle_reports dir alongside the master fidelity table (container or host).
OUTPUT_DIR = str(REPORTS_OUT)  # [unify] was /app/optimization_reports or ~/oraqle_reports

# Authored on a large (16x6) canvas, so use generous sizes that stay readable
# once the figure is scaled into the paper (titles are axes + 2).
AXIS_FS = 18
TITLE_FS = AXIS_FS + 2   # 20
SUPTITLE_FS = AXIS_FS + 4  # 22
ANNOT_FS = 14
LEGEND_FS = 16
TICK_FS = 15

# Distinct markers cycled per model (legend replaces inline labels).
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]


def _pareto_frontier(df, x_col, y_col):
    """Return the subset of rows on the Pareto frontier that MINIMISES x_col
    (parameters) while MAXIMISING y_col (fidelity), sorted by x_col ascending.
    """
    ordered = df.sort_values([x_col, y_col], ascending=[True, False])
    frontier, best_y = [], float("-inf")
    for _, row in ordered.iterrows():
        if row[y_col] > best_y:
            frontier.append(row)
            best_y = row[y_col]
    return pd.DataFrame(frontier)


def _draw_pareto_panel(ax, df_dur, fidelity_col, panel_label, description,
                       style_map, line_color):
    """Draw one Pareto panel: a neutral frontier line plus one distinctly
    marked point per model (identified via the shared legend, not inline
    labels). `style_map` maps model name -> (color, marker)."""
    df_dur = df_dur.sort_values("parameters")

    # One marker per model (consistent colour+marker across panels).
    for _, row in df_dur.iterrows():
        color, marker = style_map[row["model"]]
        ax.scatter(row["parameters"], row[fidelity_col],
                   s=150, marker=marker, color=color,
                   edgecolor="gray", linewidth=0.5, zorder=3)

    ax.set_xscale("log")
    ax.margins(x=0.12, y=0.10)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_xlabel("Number of Parameters (Log Scale)", fontsize=AXIS_FS)
    ax.set_ylabel("5-Qubit Fidelity (Geometric Mean)", fontsize=AXIS_FS)
    ax.set_title(f"({panel_label}) {description}", fontsize=TITLE_FS,
                 fontweight="bold")
    ax.grid(False)


def plot_pareto_params_vs_fidelity(csv_path=MASTER_FIDELITY_CSV,
                                   durations_ns=(600, 1000),
                                   fidelity_col="F5Q_gmean",
                                   out_dir=None,
                                   fig_width_in=None,
                                   fig_height_in=6.0):
    """Side-by-side Pareto fronts of model size vs. 5-qubit fidelity.

    One panel per entry in `durations_ns` (default 600 ns and 1000 ns), sharing
    a common y-axis.

    x-axis : number of parameters (log scale).
    y-axis : geometric-mean 5-qubit fidelity (`fidelity_col`).

    The `parameters` column is taken as the full multi-qubit model size as-is.
    """
    df = pd.read_csv(csv_path)

    panels = []
    for ns in durations_ns:
        sub = df[df["duration_ns"] == ns].copy()
        if sub.empty:
            print(f"No rows at duration_ns={ns} in {csv_path}; skipping panel.")
            continue
        panels.append((ns, sub))
    if not panels:
        print("No data to plot.")
        return

    # No grid (white background); neutral frontier line, coloured markers.
    sns.set_style("white")
    line_color = "0.35"  # neutral grey so it never clashes with model colours

    # Stable model -> (colour, marker) map shared across panels, ordered by
    # model size (smallest first) so the legend reads small -> large.
    ref_ns = max(ns for ns, _ in panels)
    ref_df = df[df["duration_ns"] == ref_ns]
    models = list(ref_df.sort_values("parameters")["model"])
    # Append any models missing from the reference duration.
    for _, sub in panels:
        for m in sub["model"]:
            if m not in models:
                models.append(m)
    colors = sns.color_palette("deep", len(models))
    style_map = {m: (colors[i], MARKERS[i % len(MARKERS)])
                 for i, m in enumerate(models)}

    # Match the existing side-by-side convention (plot_pareto_front in
    # evaluate_truncated.py): ~8 in per panel at 6 in tall, scaled down by
    # includegraphics[width=\textwidth] in the paper.
    width = fig_width_in if fig_width_in is not None else 8.0 * len(panels)
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(16, 6),
                             sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for i, (ax, (ns, sub)) in enumerate(zip(axes, panels)):
        panel_label = chr(ord("a") + i)
        description = f"Model Size vs. Fidelity: Readout {ns} ns"
        _draw_pareto_panel(ax, sub, fidelity_col, panel_label, description,
                           style_map, line_color)

    # With a shared y-axis, only the leftmost panel keeps the y-label.
    for ax in axes[1:]:
        ax.set_ylabel("")

    # Single legend, placed inside the right-hand panel (empty lower-right).
    handles = [Line2D([0], [0], linestyle="None", marker=mk, color=col,
                      markeredgecolor="gray", markersize=10, label=m)
               for m, (col, mk) in style_map.items()]
    axes[-1].legend(handles=handles, title="Model", fontsize=LEGEND_FS,
                    title_fontsize=LEGEND_FS, loc="lower right", frameon=True)

    fig.tight_layout()

    if out_dir is None:
        out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(f"{ns}ns" for ns, _ in panels)
    out_path = os.path.join(out_dir, f"pareto_params_vs_fidelity_{tag}.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved Pareto plot: {out_path}")
    return out_path


def plot_geometric_mean_accuracies(df, exclude_2nd_qubit=False, plot_error_rate=True):
    """
    Plots readout duration (trace_length) on the x-axis and geometric mean accuracy 
    on the y-axis. Plots a line for each model.
    """
    if df.empty:
        print("DataFrame is empty. Nothing to plot.")
        return
        
    plt.figure(figsize=(10, 6))
    
    y = 'geometric_mean_accuracy_excl_q1' if exclude_2nd_qubit else 'geometric_mean_accuracy'

    df_plot = df.copy()

    # Convert accuracy to error rate
    if plot_error_rate:
        df_plot[y] = 100.0 - df_plot[y]

    sns.lineplot(
        data=df_plot,
        x='trace_length',
        y=y,
        hue='model_name',
        marker='o'
    )

    if plot_error_rate:
        plt.title('Geometric Mean Error Rate vs Readout Duration')
        plt.ylabel('Geometric Mean Error Rate (%)')
        plt.yscale('log') # Set the log scale here!
    else:
        plt.title('Geometric Mean Accuracy vs Readout Duration')
        plt.ylabel('Geometric Mean Accuracy (%)')
    
    plt.xlabel('Readout Duration (Trace Length)')
    plt.grid(True)
    plt.legend(title='Model')
    plt.tight_layout()
    filename_metric = "error_rates" if plot_error_rate else "accuracies" 
    filename_metric += "_excl_q1" if exclude_2nd_qubit else ""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # [unify]
    plt.savefig(f"{OUTPUT_DIR}/multi-model_{filename_metric}.pdf", bbox_inches="tight", dpi=600)

def plot_single_model_qubit_accuracies(df, model_name):
    """
    Plots readout duration on the x-axis and individual qubit accuracies on the y-axis,
    for a single specified model.
    """
    if df.empty:
        print("DataFrame is empty. Nothing to plot.")
        return
        
    model_df = df[df['model_name'] == model_name].copy()
    if model_df.empty:
        print(f"No data found for model: {model_name}")
        return
        
    plt.figure(figsize=(10, 6))
    
    qubit_cols = [c for c in model_df.columns if c.startswith('qubit_') and c.endswith('_accuracy')]
    
    melted_df = model_df.melt(
        id_vars=['trace_length'], 
        value_vars=qubit_cols,
        var_name='Qubit', 
        value_name='Accuracy'
    )
    
    # Clean up the 'Qubit' labels for the legend
    melted_df['Qubit'] = melted_df['Qubit'].apply(lambda x: f"Qubit {x.split('_')[1]}")
    
    sns.lineplot(
        data=melted_df,
        x='trace_length',
        y='Accuracy',
        hue='Qubit',
        marker='o'
    )
    
    plt.title(f'Individual Qubit Accuracies vs Readout Duration ({model_name})')
    plt.xlabel('Readout Duration (Trace Length)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend(title='Qubit')
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # [unify]
    plt.savefig("{}/{}_model_accuracies.pdf".format(OUTPUT_DIR, model_name), bbox_inches="tight", dpi=600)

if __name__ == "__main__":
    # Pareto front of parameters vs fidelity from the master fidelity table.
    print(f"[plot_fidelities] cwd            = {os.getcwd()}", flush=True)
    print(f"[plot_fidelities] this file      = {os.path.abspath(__file__)}", flush=True)
    print(f"[plot_fidelities] MASTER CSV     = {MASTER_FIDELITY_CSV} "
          f"(exists={os.path.exists(MASTER_FIDELITY_CSV)})", flush=True)
    print(f"[plot_fidelities] OUTPUT_DIR     = {OUTPUT_DIR}", flush=True)
    if os.path.exists(MASTER_FIDELITY_CSV):
        plot_pareto_params_vs_fidelity()
    else:
        print("[plot_fidelities] !! master_fidelity.csv NOT FOUND -> Pareto plot "
              "SKIPPED. Mount the oraqle_reports dir into the container or pass "
              "csv_path=... explicitly.", flush=True)

    # Check common locations for the optimization_reports directory
    csv_directory = os.environ.get("ORAQLE_REPORTS_IN", str(RESULTS_DISC))  # [unify] was ./optimization_reports
        
    df = fetch_results(csv_dir=csv_directory)
    
    if not df.empty:
        # Plot 1: Geometric mean of all models
        plot_geometric_mean_accuracies(df)
        plot_geometric_mean_accuracies(df, True)
        
        # Plot 2: Individual qubit accuracies for a specific model
        # By default we plot the first available model
        models_available = df['model_name'].unique()
        if len(models_available) > 0:
            for model in models_available:
                plot_single_model_qubit_accuracies(df, model_name=model)
