import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import gmean

def fetch_results(csv_dir="./optimization_reports"):
    """
    Fetches the hyper-optimization results from CSV files.
    Groups by model_name and trace_length to aggregate per-qubit results.
    Returns a pandas DataFrame containing model names, readout durations (trace lengths),
    individual qubit accuracies, and their geometric mean.
    """
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {csv_dir}")
        return pd.DataFrame()
    
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Identify qubit columns
    qubit_cols = [f"qubit_{i}_accuracy" for i in range(5)]
    
    # Ensure numerical columns are handled correctly
    for col in qubit_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        else:
            combined_df[col] = np.nan
            
    # Group by model_name and trace_length
    # Using 'max' allows us to merge the per-qubit runs into a single row
    agg_funcs = {col: 'max' for col in qubit_cols}
    grouped_df = combined_df.groupby(['model_name', 'trace_length'], as_index=False).agg(agg_funcs)
    
    # Calculate geometric mean of the 5 qubits
    def calc_gmean(row):
        accs = [row[col] for col in qubit_cols if pd.notna(row[col])]
        if len(accs) > 0:
            return gmean(accs)
        return np.nan
        
    # Calculate geometric mean excluding the 2nd qubit (qubit_1)
    def calc_gmean_excl_q1(row):
        accs = [row[col] for col in qubit_cols if col != "qubit_1_accuracy" and pd.notna(row[col])]
        if len(accs) > 0:
            return gmean(accs)
        return np.nan
        
    grouped_df['geometric_mean_accuracy'] = grouped_df.apply(calc_gmean, axis=1)
    grouped_df['geometric_mean_accuracy_excl_q1'] = grouped_df.apply(calc_gmean_excl_q1, axis=1)
    
    return grouped_df

if __name__ == "__main__":
    # Example usage
    df = fetch_results()
    if not df.empty:
        print(df.head())
