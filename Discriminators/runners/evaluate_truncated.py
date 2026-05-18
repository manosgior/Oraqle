import os
import csv
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
from loguru import logger
import sys
from scipy.stats import gmean

# Add parent directory to sys.path so we can import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks import (
    SingleQubitFNN,
    CNN,
    Arxiv240618807FNN,
    Net_rmf,
    KLiNQTeacherModel,
    KLiNQStudentModel,
    QubitClassifierTransformer,
)
from helpers.cnn_helpers import prepare_cnn_data

# ============================================================================
# Configuration
# ============================================================================
RAW_TRAIN_FILE = "/data/five_qubit_data/DRaw_C_Tr_v0-001"
RAW_TEST_FILE = "/data/five_qubit_data/DRaw_C_Te_v0-002"
CNN_TEST_FILE = "/data/cnn/Qubit_5Channel_ds20_test.h5"
CNN_TRAIN_FILE = "/data/cnn/Qubit_5Channel_ds20_train.h5"

NUM_QUBITS = 5
MAX_LENGTH = 500
TRACE_LENGTHS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_DIR = "./optimization_reports"
MAX_TEST_SAMPLES = 100_000
SAMPLE_SEED = 42

FREQ_READOUT = -np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])

# ============================================================================
# Helper Functions
# ============================================================================
def load_hdf5_data(filepath, trace_length, is_test=False, max_samples=None):
    key_suffix = "test" if is_test else "train"
    with h5py.File(filepath, "r") as hf:
        total = hf[f"X_{key_suffix}"].shape[0]
        if max_samples is not None and max_samples < total:
            rng = np.random.RandomState(SAMPLE_SEED)
            indices = np.sort(rng.choice(total, size=max_samples, replace=False))
            X = hf[f"X_{key_suffix}"][indices, :trace_length, :]
            y = hf[f"y_{key_suffix}"][indices]
        else:
            X = hf[f"X_{key_suffix}"][:, :trace_length, :]
            y = hf[f"y_{key_suffix}"][:]
    return X, y

def demodulate_and_average(traces, freq_readout, dt=2e-9):
    N, T, _ = traces.shape
    num_qubits = len(freq_readout)
    t = np.arange(T) * dt
    demodulated_acc = np.zeros((N, num_qubits, 2))
    
    for i, freq in enumerate(freq_readout):
        lo_i = np.cos(2 * np.pi * freq * t)
        lo_q = -np.sin(2 * np.pi * freq * t)
        I_trace = traces[:, :, 0]
        Q_trace = traces[:, :, 1]
        I_demod = I_trace * lo_i - Q_trace * lo_q
        Q_demod = I_trace * lo_q + Q_trace * lo_i
        demodulated_acc[:, i, 0] = np.mean(I_demod, axis=1)
        demodulated_acc[:, i, 1] = np.mean(Q_demod, axis=1)
    return demodulated_acc

def extract_qubit_labels(y_packed, target_qubit):
    return (y_packed >> target_qubit) & 1

def evaluate_test_accuracy(model, X_test, y_test, task_type, batch_size=512):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size)
    all_preds = []
    with torch.no_grad():
        for (X_b,) in loader:
            X_b = X_b.to(DEVICE)
            out = model(X_b)
            all_preds.append(out.cpu())
    all_preds = torch.cat(all_preds, dim=0)

    if task_type == '32class':
        pred_labels = all_preds.argmax(dim=1).numpy()
        per_qubit_accs = []
        for q in range(NUM_QUBITS):
            pred_q = (pred_labels >> q) & 1
            true_q = (y_test >> q) & 1
            per_qubit_accs.append(100.0 * np.mean(pred_q == true_q))
        return per_qubit_accs
    elif task_type == 'multitask':
        pred_binary = (torch.sigmoid(all_preds) >= 0.5).int().numpy()
        y_int = y_test.astype(int)
        per_qubit_accs = [100.0 * np.mean(pred_binary[:, q] == y_int[:, q]) for q in range(NUM_QUBITS)]
        return per_qubit_accs
    elif task_type == 'binary':
        # Arxiv model already has a Sigmoid at the end and outputs probabilities.
        # Other binary models (like KLiNQ) output raw logits.
        if isinstance(model, Arxiv240618807FNN):
            preds = all_preds.squeeze()
        else:
            preds = torch.sigmoid(all_preds.squeeze())
            
        pred_binary = (preds >= 0.5).int().numpy()
        acc = 100.0 * np.mean(pred_binary == y_test)
        return acc

def get_best_models():
    """Finds the best model path for each architecture at trace_length=500."""
    all_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    df_list = []
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except:
            pass
    if not df_list:
        return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)
    df_500 = df[df['trace_length'] == MAX_LENGTH]
    
    # Sort by overall accuracy to get the best if multiple exist
    if 'overall_accuracy' in df_500.columns:
        df_500['overall_accuracy'] = pd.to_numeric(df_500['overall_accuracy'], errors='coerce')
        df_500 = df_500.sort_values('overall_accuracy', ascending=False)
        
    # Get the best entry per model and target_qubit
    best_models = df_500.drop_duplicates(subset=['model_name', 'target_qubit'])
    return best_models

# ============================================================================
# Main Evaluation Loop
# ============================================================================
def evaluate_all():
    best_models_df = get_best_models()
    if best_models_df.empty:
        logger.error("No CSV reports found for trace_length=500.")
        return

    # Load 500-length data for computing reference stats (envelopes, norms)
    logger.info("Loading reference 500-length data...")
    try:
        X_train_ref, y_train_ref = load_hdf5_data(RAW_TRAIN_FILE, MAX_LENGTH, is_test=False, max_samples=400_000)
        X_test_ref, y_test_ref = load_hdf5_data(RAW_TEST_FILE, MAX_LENGTH, is_test=True, max_samples=MAX_TEST_SAMPLES)
        
        # Flatten reference
        X_train_flat_ref = X_train_ref.reshape(X_train_ref.shape[0], -1)
    except FileNotFoundError:
        logger.error("HDF5 data files not found. Are you running this inside the docker container?")
        return

    results = []

    for length in TRACE_LENGTHS:
        logger.info(f"Evaluating truncated trace length: {length}")
        
        # We simulate short readout by keeping only 'length' samples, and padding the rest with zeros to reach 500.
        X_test_trunc = np.zeros_like(X_test_ref)
        X_test_trunc[:, :length, :] = X_test_ref[:, :length, :]
        X_test_trunc_flat = X_test_trunc.reshape(X_test_trunc.shape[0], -1)
        
        for idx, row in best_models_df.iterrows():
            model_name = row['model_name']
            model_path = row['model_path']
            target_qubit = row['target_qubit']
            
            if not isinstance(model_path, str) or not os.path.exists(model_path):
                logger.warning(f"Model file missing for {model_name}: {model_path}")
                continue

            try:
                accs = [np.nan] * 5
                
                # 1. FNN
                if model_name == "FNN":
                    fnn_mean = np.mean(X_train_flat_ref, axis=0)
                    fnn_std = np.std(X_train_flat_ref, axis=0) + 1e-10
                    X_test_norm = (X_test_trunc_flat - fnn_mean) / fnn_std
                    
                    model = SingleQubitFNN(input_size=MAX_LENGTH * 2, output_size=32).to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    accs = evaluate_test_accuracy(model, X_test_norm, y_test_ref, '32class')
                    
                # 2. HERQULES Net_rmf
                elif model_name == "HERQULES_Net_rmf":
                    mf_test = np.zeros((X_test_trunc.shape[0], NUM_QUBITS))
                    rmf_test = np.zeros((X_test_trunc.shape[0], NUM_QUBITS))
                    
                    for q in range(NUM_QUBITS):
                        y_q = extract_qubit_labels(y_train_ref, q)
                        gnd = X_train_flat_ref[y_q == 0]
                        ext = X_train_flat_ref[y_q == 1]
                        n = min(len(gnd), len(ext))
                        
                        diff_mf = gnd[:n] - ext[:n]
                        mf_envelope = np.mean(diff_mf, axis=0) / (np.var(diff_mf, axis=0) + 1e-10)
                        mf_out_train = X_train_flat_ref @ mf_envelope
                        
                        # Dot product of zero-padded trace with 500-length envelope
                        mf_test[:, q] = X_test_trunc_flat @ mf_envelope
                        
                        mf_gnd = mf_out_train[y_q == 0]
                        threshold = np.mean(mf_gnd)
                        sigma_gnd = np.std(mf_gnd) + 1e-10
                        relax_mask = (y_q == 1) & (np.abs(mf_out_train - threshold) < 2 * sigma_gnd)
                        relax_traces = X_train_flat_ref[relax_mask]
                        
                        if len(relax_traces) > 10:
                            n_rmf = min(len(relax_traces), len(gnd))
                            diff_rmf = relax_traces[:n_rmf] - gnd[:n_rmf]
                            rmf_envelope = np.mean(diff_rmf, axis=0) / (np.var(diff_rmf, axis=0) + 1e-10)
                            rmf_test[:, q] = X_test_trunc_flat @ rmf_envelope
                        else:
                            rmf_test[:, q] = 0.0

                    mf_rmf_test = np.concatenate([mf_test, rmf_test], axis=1)
                    model = Net_rmf().to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    accs = evaluate_test_accuracy(model, mf_rmf_test, y_test_ref, '32class')
                    
                # 3. Arxiv (QubiCML)
                elif model_name == "Arxiv240618807FNN":
                    q = int(target_qubit)
                    X_train_demod = demodulate_and_average(X_train_ref, FREQ_READOUT)
                    
                    # For Arxiv, zero-padding might distort averages, but actually we want to evaluate 
                    # shorter readout. It's better to average only over the active length!
                    X_test_trunc_demod = demodulate_and_average(X_test_ref[:, :length, :], FREQ_READOUT)
                    
                    X_q_train = X_train_demod[:, q, :]
                    X_min = np.min(X_q_train, axis=0)
                    X_range = np.max(X_q_train, axis=0) - X_min
                    X_range[X_range == 0] = 1e-10
                    
                    X_q_test_norm = (X_test_trunc_demod[:, q, :] - X_min) / X_range
                    y_q_test = extract_qubit_labels(y_test_ref, q)
                    
                    model = Arxiv240618807FNN().to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    acc = evaluate_test_accuracy(model, X_q_test_norm, y_q_test, 'binary')
                    accs[q] = acc
                    
                # 4. Transformer
                elif model_name == "Transformer":
                    # Parse hyperparameters safely handling NaNs from merged CSVs
                    patch_sz = int(row['patch_size']) if pd.notna(row.get('patch_size')) else 10
                    embed_dim = int(row['embedding_dim']) if pd.notna(row.get('embedding_dim')) else 128
                    n_heads = int(row['num_heads']) if pd.notna(row.get('num_heads')) else 8
                    n_layers = int(row['num_layers']) if pd.notna(row.get('num_layers')) else 4
                    drop = float(row['dropout']) if pd.notna(row.get('dropout')) else 0.1

                    model = QubitClassifierTransformer(
                        num_classes=32, 
                        patch_size=patch_sz,
                        embedding_dim=embed_dim,
                        num_heads=n_heads,
                        num_layers=n_layers,
                        dropout=drop
                    ).to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    accs = evaluate_test_accuracy(model, X_test_trunc, y_test_ref, '32class')
                    
                # 5. CNN
                elif model_name == "CNN":
                    try:
                        X_cnn_test, y_cnn_test = prepare_cnn_data(
                            CNN_TEST_FILE, downsample_factor=20, original_length=1000,
                            num_qubits=NUM_QUBITS, time_slice=(0, length), is_test=True)
                        
                        # Pad CNN data to length 25 (500/20) with zeros to match expected shape
                        X_cnn_padded = torch.zeros(X_cnn_test.shape[0], 10, 25)
                        trunc_len_ds = X_cnn_test.shape[2]
                        X_cnn_padded[:, :, :trunc_len_ds] = X_cnn_test
                        
                        model = CNN(in_channels=10, m_param=8, num_qubits=NUM_QUBITS).to(DEVICE)
                        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                        accs = evaluate_test_accuracy(model, X_cnn_padded.numpy(), y_cnn_test.numpy(), 'multitask')
                    except Exception as e:
                        logger.warning(f"Failed CNN evaluate: {e}")
                        
                # 6. KLiNQ Teacher
                elif model_name == "KLiNQ_Teacher":
                    q = int(target_qubit)
                    X_tea_mean = np.mean(X_train_flat_ref, axis=0)
                    X_tea_std = np.std(X_train_flat_ref, axis=0) + 1e-10
                    X_test_teacher = (X_test_trunc_flat - X_tea_mean) / X_tea_std
                    
                    model = KLiNQTeacherModel(input_size=MAX_LENGTH * 2, output_size=1).to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    acc = evaluate_test_accuracy(model, X_test_teacher, extract_qubit_labels(y_test_ref, q), 'binary')
                    accs[q] = acc
                    
                # 7. KLiNQ Student
                elif model_name == "KLiNQ_Student":
                    q = int(target_qubit)
                    target_length_klinq = 5
                    bin_size = max(1, MAX_LENGTH // target_length_klinq)
                    n_bins = MAX_LENGTH // bin_size
                    
                    X_avg_I = X_test_trunc[:, :n_bins * bin_size, 0].reshape(X_test_trunc.shape[0], n_bins, bin_size).mean(axis=2)
                    X_avg_Q = X_test_trunc[:, :n_bins * bin_size, 1].reshape(X_test_trunc.shape[0], n_bins, bin_size).mean(axis=2)
                    X_avg_test = np.concatenate([X_avg_I, X_avg_Q], axis=1)

                    y_q_train = extract_qubit_labels(y_train_ref, q)
                    gnd = X_train_flat_ref[y_q_train == 0]
                    ext = X_train_flat_ref[y_q_train == 1]
                    n = min(len(gnd), len(ext))
                    diff = gnd[:n] - ext[:n]
                    envelope = np.mean(diff, axis=0) / (np.var(diff, axis=0) + 1e-10)
                    mf_scalar_test = (X_test_trunc_flat @ envelope).reshape(-1, 1)

                    def _znorm(a):
                        return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-10)

                    X_combined_test = np.concatenate(
                        [_znorm(X_test_trunc_flat.copy()), _znorm(X_avg_test), _znorm(mf_scalar_test)], axis=1
                    )
                    
                    student_input_size = X_combined_test.shape[1]
                    model = KLiNQStudentModel(input_size=student_input_size).to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    acc = evaluate_test_accuracy(model, X_combined_test, extract_qubit_labels(y_test_ref, q), 'binary')
                    accs[q] = acc

                results.append({
                    "model_name": model_name,
                    "target_qubit": target_qubit,
                    "evaluated_length": length,
                    "qubit_0_accuracy": accs[0],
                    "qubit_1_accuracy": accs[1],
                    "qubit_2_accuracy": accs[2],
                    "qubit_3_accuracy": accs[3],
                    "qubit_4_accuracy": accs[4],
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} at length {length}: {e}")

    if results:
        df_res = pd.DataFrame(results)
        os.makedirs(CSV_DIR, exist_ok=True)
        out_path = os.path.join(CSV_DIR, "truncated_evaluation_results.csv")
        df_res.to_csv(out_path, index=False)
        logger.info(f"Saved truncated evaluation results to {out_path}")
        print(df_res.head())
        
if __name__ == "__main__":
    evaluate_all()
