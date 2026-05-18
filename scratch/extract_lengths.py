import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import gmean
import glob

def fetch_results(csv_dir):
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            pass
            
    if not df_list:
        return pd.DataFrame()
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    qubit_cols = [f"qubit_{i}_accuracy" for i in range(5)]
    
    for col in qubit_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        else:
            combined_df[col] = np.nan
            
    agg_funcs = {col: 'max' for col in qubit_cols}
    # Group by model_name and trace_length
    grouped_df = combined_df.groupby(['model_name', 'trace_length'], as_index=False).agg(agg_funcs)
    
    def calc_gmean(row):
        accs = [row[col] for col in qubit_cols if pd.notna(row[col])]
        if len(accs) == 5:
            return gmean(accs)
        return np.nan
        
    grouped_df['geometric_mean_accuracy'] = grouped_df.apply(calc_gmean, axis=1) / 100.0
    
    return grouped_df

if __name__ == "__main__":
    csv_dir = "/home/manosgior/Documents/GitHub/Oraqle/Discriminators/training_results"
    df = fetch_results(csv_dir)
    
    lengths = [100, 200, 300, 400, 500]
    models = [
        "Baseline_Threshold",
        "Arxiv240618807FNN",
        "CNN",
        "FNN",
        "HERQULES_Net_rmf",
        "Transformer",
        "KLiNQ_Student"
    ]
    
    # Filter and pivot
    df_filtered = df[df['trace_length'].isin(lengths) & df['model_name'].isin(models)]
    table = df_filtered.pivot(index='model_name', columns='trace_length', values='geometric_mean_accuracy')
    
    # Reorder models to match previous list
    table = table.reindex(models)
    
    print(table.to_string())
