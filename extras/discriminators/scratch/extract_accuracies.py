import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
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
    grouped_df = combined_df.groupby(['model_name', 'trace_length'], as_index=False).agg(agg_funcs)
    
    def calc_gmean(row):
        accs = [row[col] for col in qubit_cols if pd.notna(row[col])]
        if len(accs) == 5:
            return gmean(accs)
        return np.nan
        
    def calc_gmean_excl_q1(row):
        accs = [row[col] for col in qubit_cols if col != "qubit_1_accuracy" and pd.notna(row[col])]
        if len(accs) == 4:
            return gmean(accs)
        return np.nan
        
    grouped_df['geometric_mean_accuracy'] = grouped_df.apply(calc_gmean, axis=1)
    grouped_df['geometric_mean_accuracy_excl_q1'] = grouped_df.apply(calc_gmean_excl_q1, axis=1)
    
    return grouped_df

if __name__ == "__main__":
    csv_dir = str(RESULTS_DISC / "legacy_training_results")
    df = fetch_results(csv_dir)
    df_500 = df[df['trace_length'] == 500]
    print(df_500.to_string())
