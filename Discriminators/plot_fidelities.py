import os
import matplotlib.pyplot as plt
import seaborn as sns
from fetch_results import fetch_results


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
    
    plt.savefig(f"./optimization_reports/multi-model_{filename_metric}.pdf", bbox_inches="tight", dpi=600)

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
    plt.savefig("./optimization_reports/{}_model_accuracies.pdf".format(model_name), bbox_inches="tight", dpi=600)

if __name__ == "__main__":
    # Check common locations for the optimization_reports directory
    csv_directory = "./optimization_reports"
    if not os.path.exists(csv_directory):
        csv_directory = "./runners/optimization_reports"
        
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
