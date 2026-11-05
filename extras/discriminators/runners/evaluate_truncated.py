import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
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
import re
import matplotlib.pyplot as plt
import seaborn as sns

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
from trainers.train_HERQULES import (
    demodulate_all_qubits,
    compute_all_envelopes,
    build_features as herqules_build_features,
)

# ============================================================================
# Configuration
# ============================================================================
RAW_TRAIN_FILE = str(RAW_TRAIN)
RAW_TEST_FILE = str(RAW_TEST)
CNN_TEST_FILE = str(CNN_DATA / "Qubit_5Channel_ds20_test.h5")
CNN_TRAIN_FILE = str(CNN_DATA / "Qubit_5Channel_ds20_train.h5")

NUM_QUBITS = 5
MAX_LENGTH = 500
TRACE_LENGTHS = [100, 200, 300, 400]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_DIR = str(REPORTS_OUT)
MAX_TEST_SAMPLES = None
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

def _prepare_herqules_features(X_train_ref, X_test_ref, y_train_ref):
    """Demodulate train/test once and compute full-length MF/RMF envelopes.

    Mirrors the train_HERQULES.py feature pipeline so eval features match what
    the Net_rmf checkpoint actually saw during training: per-qubit Butterworth
    demodulation -> geometric preclassifier -> Wiener MF/RMF envelopes with
    paper boxcars. Envelopes are full-length (1000 = 500*2); build_features
    slices them per evaluated length.
    """
    logger.info("HERQULES: demodulating train traces (one-time per model)...")
    demod_train = demodulate_all_qubits(X_train_ref)
    logger.info("HERQULES: demodulating test traces...")
    demod_test = demodulate_all_qubits(X_test_ref)
    logger.info("HERQULES: computing MF/RMF envelopes from purified train traces...")
    np.random.seed(SAMPLE_SEED)  # deterministic class-balancing subsampling in _get_mf_core
    mf_envs, rmf_envs = compute_all_envelopes(demod_train, y_train_ref)
    return demod_test, mf_envs, rmf_envs

def get_saved_models():
    """Fetches all models from the saved_models directory and parses their type and lengths."""
    # Inside Docker, ./saved_models is the host bind-mount (live checkpoints);
    # Discriminators/saved_models is whatever was baked into the image at build
    # time and never refreshed. Prefer the mount; fall back to the in-repo path
    # when running outside Docker from the project root.
    if True:  # [unify] was ./saved_models (docker mount) or Discriminators/saved_models
        search_dir = os.environ.get("ORAQLE_MODELS_IN", str(CHECKPOINTS))
    else:
        search_dir = "Discriminators/saved_models"
        
    # Same Docker-mount issue as saved_models: prefer the bind-mounted
    # ./optimization_reports over the baked-in Discriminators/optimization_reports.
    if True:  # [unify] was ./optimization_reports or Discriminators/optimization_reports
        csv_dir = os.environ.get("ORAQLE_REPORTS_IN", str(RESULTS_DISC))
    else:
        csv_dir = "Discriminators/optimization_reports"
        
    df_csvs = pd.DataFrame()
    if os.path.exists(csv_dir):
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        dfs = []
        for cf in csv_files:
            try:
                dfs.append(pd.read_csv(cf))
            except Exception:
                pass
        if dfs:
            df_csvs = pd.concat(dfs, ignore_index=True)
            
    models = []
    for f in glob.glob(os.path.join(search_dir, "*.pth")):
        fname = os.path.basename(f)
        
        # Extract trained length
        len_match = re.search(r'_len(\d+)', fname)
        if not len_match:
            logger.warning(f"Skipping model with ambiguous trained length in filename: {fname}")
            continue
        trained_length = int(len_match.group(1))
        
        # We only want to run truncated evaluation on models trained on the full length (500)
        if trained_length != MAX_LENGTH:
            continue
        
        # Extract target qubit (if binary)
        q_match = re.search(r'_q(\d+)', fname)
        target_qubit = int(q_match.group(1)) if q_match else None
        
        # Determine model name
        fname_lower = fname.lower()
        if "arxiv" in fname_lower:
            model_name = "Arxiv240618807FNN"
        elif "cnn" in fname_lower:
            model_name = "CNN"
        elif "fnn" in fname_lower:
            model_name = "FNN"
        elif "herqules" in fname_lower:
            model_name = "HERQULES_Net_rmf"
        #elif "klinq_teacher" in fname_lower:
            #model_name = "KLiNQ_Teacher"
        elif "klinq_student" in fname_lower:
            model_name = "KLiNQ_Student"
        elif "transformer" in fname_lower:
            model_name = "Transformer"
        else:
            continue
            
        extra_hparams = {}
        if not df_csvs.empty and 'model_path' in df_csvs.columns:
            # Match by basename to avoid path prefix issues
            matches = df_csvs[df_csvs['model_path'].apply(lambda x: os.path.basename(str(x))) == fname]
            if not matches.empty:
                extra_hparams = matches.iloc[0].to_dict()
                
        models.append({
            "model_name": model_name,
            "model_path": f,
            "target_qubit": target_qubit,
            "trained_length": trained_length,
            "hparams": extra_hparams,
            "_mtime": os.path.getmtime(f),
        })

    df = pd.DataFrame(models)
    if df.empty:
        return df

    # Dedupe checkpoints sharing (model_name, trained_length, target_qubit),
    # keeping the most recently modified .pth. target_qubit is NaN for
    # multi-qubit models; fill so drop_duplicates groups them correctly.
    df['_dedupe_q'] = df['target_qubit'].fillna(-1)
    df = df.sort_values('_mtime', ascending=False)
    before = len(df)
    df = df.drop_duplicates(
        subset=['model_name', 'trained_length', '_dedupe_q'], keep='first'
    )
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} older duplicate checkpoint(s); kept most recent by mtime.")
    return df.drop(columns=['_mtime', '_dedupe_q']).reset_index(drop=True)

# ============================================================================
# Main Evaluation Loop
# ============================================================================
def evaluate_all():
    models_df = get_saved_models()
    if models_df.empty:
        logger.error("No saved models found in the saved_models directory.")
        return

    # Load 500-length data for computing reference stats (envelopes, norms)
    logger.info("Loading reference 500-length data...")
    try:
        X_train_ref, y_train_ref = load_hdf5_data(RAW_TRAIN_FILE, MAX_LENGTH, is_test=False, max_samples=400_000)
        X_test_ref, y_test_ref = load_hdf5_data(RAW_TEST_FILE, MAX_LENGTH, is_test=True, max_samples=MAX_TEST_SAMPLES)
    except FileNotFoundError:
        logger.error("HDF5 data files not found. Are you running this inside the docker container?")
        return

    results = []

    for idx, row in models_df.iterrows():
        model_name = row['model_name']
        model_path = row['model_path']
        target_qubit = row['target_qubit']
        trained_length = row['trained_length']
        hparams = row.get('hparams', {})
        
        logger.info(f"Evaluating model: {model_name} (Trained Len: {trained_length}, Qubit: {target_qubit})")
        
        # Prepare training reference up to trained_length
        X_train_trunc_ref = X_train_ref[:, :trained_length, :]
        X_train_flat_ref = X_train_trunc_ref.reshape(X_train_trunc_ref.shape[0], -1)

        # HERQULES: paper-faithful feature pipeline must demodulate + build
        # envelopes once per model. Done outside the length loop because the
        # envelopes are full-length and only sliced per evaluated length.
        herqules_cache = None
        if model_name == "HERQULES_Net_rmf":
            herqules_cache = _prepare_herqules_features(X_train_ref, X_test_ref, y_train_ref)

        for length in TRACE_LENGTHS:
            if length > trained_length:
                continue # Cannot evaluate on longer traces than trained
                
            logger.info(f"  -> Truncated length: {length}")
            
            # We simulate short readout by keeping only 'length' samples.
            # To preserve the scale of the integration without artificially multiplying values,
            # we use Zero-Order Hold (ZOH) padding instead of Zero-Padding. 
            # We hold the last observed sample constant for the rest of the tensor.
            X_test_trunc = np.zeros((X_test_ref.shape[0], trained_length, 2))
            X_test_trunc[:, :length, :] = X_test_ref[:, :length, :]
            
            last_samples = X_test_ref[:, length-1:length, :]
            X_test_trunc[:, length:, :] = np.repeat(last_samples, trained_length - length, axis=1)
            
            X_test_trunc_flat = X_test_trunc.reshape(X_test_trunc.shape[0], -1)

            try:
                accs = [np.nan] * 5
                
                # 1. FNN
                if model_name == "FNN":
                    fnn_mean = np.mean(X_train_flat_ref, axis=0)
                    fnn_std = np.std(X_train_flat_ref, axis=0) + 1e-10
                    X_test_norm = (X_test_trunc_flat - fnn_mean) / fnn_std
                    
                    model = SingleQubitFNN(input_size=trained_length * 2, output_size=32).to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    accs = evaluate_test_accuracy(model, X_test_norm, y_test_ref, '32class')
                    
                # 2. HERQULES Net_rmf — paper-faithful: per-qubit demodulated
                # traces dotted with truncated MF/RMF envelopes (no ZOH).
                elif model_name == "HERQULES_Net_rmf":
                    demod_test, mf_envs, rmf_envs = herqules_cache
                    features_test = herqules_build_features(demod_test, mf_envs, rmf_envs, length)
                    model = Net_rmf().to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    accs = evaluate_test_accuracy(model, features_test, y_test_ref, '32class')
                    
                # 3. Arxiv (QubiCML)
                elif model_name == "Arxiv240618807FNN":
                    q = int(target_qubit)
                    X_train_demod = demodulate_and_average(X_train_trunc_ref, FREQ_READOUT)
                    X_test_trunc_demod = demodulate_and_average(X_test_trunc, FREQ_READOUT)
                    
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
                    patch_sz = hparams.get('patch_size', 10)
                    embedding_dim = hparams.get('embedding_dim', 128)
                    num_heads = hparams.get('num_heads', 8)
                    num_layers = hparams.get('num_layers', 4)
                    
                    # Handle NaNs from pandas
                    if pd.isna(patch_sz): patch_sz = 10
                    if pd.isna(embedding_dim): embedding_dim = 128
                    if pd.isna(num_heads): num_heads = 8
                    if pd.isna(num_layers): num_layers = 4
                    
                    # Fallback infer from state_dict if CSV didn't have it
                    state_dict = torch.load(model_path, map_location=DEVICE)
                    if 'cls_token' in state_dict:
                        embedding_dim = state_dict['cls_token'].shape[2]
                    if 'patch_embedding.weight' in state_dict:
                        patch_sz = state_dict['patch_embedding.weight'].shape[2]
                    elif 'patch_embed.projection.weight' in state_dict:
                        shape = state_dict['patch_embed.projection.weight'].shape
                        if len(shape) == 2:
                            patch_sz = shape[1] // 2
                        elif len(shape) == 3:
                            patch_sz = shape[2]
                    if 'transformer_encoder.layers.0.norm1.weight' in state_dict:
                        embedding_dim = state_dict['transformer_encoder.layers.0.norm1.weight'].shape[0]
                        
                    layer_indices = set()
                    for key in state_dict.keys():
                        m = re.match(r'transformer_encoder\.layers\.(\d+)\.', key)
                        if m:
                            layer_indices.add(int(m.group(1)))
                    if layer_indices:
                        num_layers = max(layer_indices) + 1
                        
                    if embedding_dim % num_heads != 0:
                        num_heads = 4 # Fallback

                    model = QubitClassifierTransformer(
                        num_classes=32, 
                        patch_size=int(patch_sz),
                        embedding_dim=int(embedding_dim),
                        num_heads=int(num_heads),
                        num_layers=int(num_layers),
                        dropout=0.1
                    ).to(DEVICE)
                    model.load_state_dict(state_dict)
                    accs = evaluate_test_accuracy(model, X_test_trunc, y_test_ref, '32class')
                    
                # 5. CNN
                elif model_name == "CNN":
                    try:
                        m_param = hparams.get('m_param', 16)
                        if pd.isna(m_param): m_param = 16
                        
                        state_dict = torch.load(model_path, map_location=DEVICE)
                        if 'conv1.weight' in state_dict:
                            m_param = state_dict['conv1.weight'].shape[0]

                        X_cnn_test, y_cnn_test = prepare_cnn_data(
                            CNN_TEST_FILE, downsample_factor=20, original_length=1000,
                            num_qubits=NUM_QUBITS, time_slice=(0, trained_length), is_test=True)
                        X_cnn_train, _ = prepare_cnn_data(
                            CNN_TRAIN_FILE, downsample_factor=20, original_length=1000,
                            num_qubits=NUM_QUBITS, time_slice=(0, trained_length), is_test=False)
                        
                        # Convert to numpy arrays if they are PyTorch tensors
                        if torch.is_tensor(X_cnn_train): X_cnn_train = X_cnn_train.numpy()
                        if torch.is_tensor(X_cnn_test): X_cnn_test = X_cnn_test.numpy()
                        if torch.is_tensor(y_cnn_test): y_cnn_test = y_cnn_test.numpy()

                        # CNN Training stats for normalization
                        cnn_mean = np.mean(X_cnn_train, axis=(0, 2), keepdims=True)
                        cnn_std = np.std(X_cnn_train, axis=(0, 2), keepdims=True) + 1e-10

                        # ZOH pad CNN test data
                        expected_ds_len = trained_length // 20
                        ds_length = length // 20
                        X_cnn_padded = np.zeros_like(X_cnn_test)
                        X_cnn_padded[:, :, :ds_length] = X_cnn_test[:, :, :ds_length]
                        last_samples_cnn = X_cnn_test[:, :, ds_length-1:ds_length]
                        X_cnn_padded[:, :, ds_length:] = np.repeat(last_samples_cnn, expected_ds_len - ds_length, axis=2)
                        X_cnn_norm = (X_cnn_padded - cnn_mean) / cnn_std
                        
                        model = CNN(in_channels=10, m_param=int(m_param), num_qubits=NUM_QUBITS).to(DEVICE)
                        model.load_state_dict(state_dict)
                        accs = evaluate_test_accuracy(model, X_cnn_norm, y_cnn_test, 'multitask')
                    except Exception as e:
                        logger.warning(f"Failed CNN evaluate: {e}")
                        
                # 6. KLiNQ Teacher
                elif model_name == "KLiNQ_Teacher":
                    q = int(target_qubit)
                    X_tea_mean = np.mean(X_train_flat_ref, axis=0)
                    X_tea_std = np.std(X_train_flat_ref, axis=0) + 1e-10
                    X_test_teacher = (X_test_trunc_flat - X_tea_mean) / X_tea_std
                    
                    model = KLiNQTeacherModel(input_size=trained_length * 2, output_size=1).to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    acc = evaluate_test_accuracy(model, X_test_teacher, extract_qubit_labels(y_test_ref, q), 'binary')
                    accs[q] = acc
                    
                # 7. KLiNQ Student
                # Paper-faithful student input: [binned avg | MF scalar] only
                # (= 2*n_bins + 1 features). Legacy checkpoints also prepended
                # the full flat ZOH trace (= 2*trained_length more features),
                # which gave the student the same information as the teacher and
                # blew fc1 up to ~16k weights per qubit. We detect which kind of
                # checkpoint we have by reading fc1.in_features and build a
                # matching input. The MF scalar uses the truncated trace dotted
                # with the truncated envelope (no ZOH * late-envelope bias).
                elif model_name == "KLiNQ_Student":
                    q = int(target_qubit)
                    target_length_klinq = 5
                    bin_size = max(1, trained_length // target_length_klinq)
                    n_bins = trained_length // bin_size

                    state_dict = torch.load(model_path, map_location=DEVICE)
                    fc1_in = state_dict['fc1.weight'].shape[1]
                    paper_faithful_in = 2 * n_bins + 1
                    legacy_in = 2 * trained_length + 2 * n_bins + 1

                    X_avg_I = X_test_trunc[:, :n_bins * bin_size, 0].reshape(X_test_trunc.shape[0], n_bins, bin_size).mean(axis=2)
                    X_avg_Q = X_test_trunc[:, :n_bins * bin_size, 1].reshape(X_test_trunc.shape[0], n_bins, bin_size).mean(axis=2)
                    X_avg_test = np.concatenate([X_avg_I, X_avg_Q], axis=1)

                    y_q_train = extract_qubit_labels(y_train_ref, q)
                    gnd = X_train_flat_ref[y_q_train == 0]
                    ext = X_train_flat_ref[y_q_train == 1]
                    n = min(len(gnd), len(ext))
                    diff = gnd[:n] - ext[:n]
                    envelope = np.mean(diff, axis=0) / (np.var(diff, axis=0) + 1e-10)

                    F_short = length * 2
                    trace_short_flat = X_test_ref[:, :length, :].reshape(X_test_ref.shape[0], -1)
                    mf_scalar_test = (trace_short_flat @ envelope[:F_short]).reshape(-1, 1)

                    def _znorm(a):
                        return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-10)

                    if fc1_in == paper_faithful_in:
                        X_combined_test = np.concatenate(
                            [_znorm(X_avg_test), _znorm(mf_scalar_test)], axis=1
                        )
                    elif fc1_in == legacy_in:
                        logger.warning(
                            f"{os.path.basename(model_path)}: legacy KLiNQ checkpoint "
                            f"(fc1.in={legacy_in} includes the full flat trace). Eval falls "
                            f"back to legacy input shape; retrain with the updated "
                            f"hyper_optimize.py to get a paper-faithful ~300-param student."
                        )
                        X_combined_test = np.concatenate(
                            [_znorm(X_test_trunc_flat.copy()), _znorm(X_avg_test), _znorm(mf_scalar_test)], axis=1
                        )
                    else:
                        raise ValueError(
                            f"Unexpected fc1.in_features={fc1_in} for "
                            f"{os.path.basename(model_path)}; expected {paper_faithful_in} "
                            f"(paper) or {legacy_in} (legacy)."
                        )

                    model = KLiNQStudentModel(input_size=fc1_in).to(DEVICE)
                    model.load_state_dict(state_dict)
                    acc = evaluate_test_accuracy(model, X_combined_test, extract_qubit_labels(y_test_ref, q), 'binary')
                    accs[q] = acc

                results.append({
                    "model_name": model_name,
                    "target_qubit": target_qubit,
                    "trained_length": trained_length,
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
        
def plot_pareto_front(csv_path):
    if not os.path.exists(csv_path):
        logger.error(f"Cannot find {csv_path} for plotting.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_theme(style="whitegrid")
    
    df_500 = df[df['trained_length'] == 500]
    df_200 = df[df['trained_length'] == 200]
    
    sns.scatterplot(
        ax=axes[0],
        data=df_500, 
        x="total_parameters_5_qubits", 
        y="mean_5_accuracy", 
        hue="model_name", 
        style="model_name", 
        palette="deep", 
        s=150,
        legend=False
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of Parameters (Log Scale)", fontsize=12)
    axes[0].set_ylabel("Fidelity / Mean(5) Accuracy", fontsize=12)
    axes[0].set_title("Trace Size: 1000 ns (500 samples)", fontsize=14)
    
    sns.scatterplot(
        ax=axes[1],
        data=df_200, 
        x="total_parameters_5_qubits", 
        y="mean_5_accuracy", 
        hue="model_name", 
        style="model_name", 
        palette="deep", 
        s=150
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of Parameters (Log Scale)", fontsize=12)
    axes[1].set_ylabel("Fidelity / Mean(5) Accuracy", fontsize=12)
    axes[1].set_title("Trace Size: 400 ns (200 samples)", fontsize=14)
    
    axes[1].legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.suptitle("Pareto Front: Model Parameters vs. Fidelity", fontsize=16)
    plt.tight_layout()
    
    out_pdf = os.path.join(os.path.dirname(csv_path), "params_vs_accuracy_pareto_side_by_side.pdf")
    plt.savefig(out_pdf, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved side-by-side pareto scatter plot to {out_pdf}")

def evaluate_params_and_accuracy():
    models_df = get_saved_models()
    if models_df.empty:
        logger.error("No saved models found in the saved_models directory.")
        return

    # Map model names to Mean(5) accuracy from the provided LaTeX table
    mean5_acc_map_500 = {
        "FNN": 0.909,
        "HERQULES_Net_rmf": 0.914,
        "Arxiv240618807FNN": 0.893,
        "KLiNQ_Student": 0.908,
        "Transformer": 0.906,
        "CNN": 0.908,
    }

    mean5_acc_map_200 = {
        "FNN": 0.877,
        "HERQULES_Net_rmf": 0.851,
        "Arxiv240618807FNN": 0.864,
        "KLiNQ_Student": 0.877,
        "Transformer": 0.876,
        "CNN": 0.857,
    }

    # Provide the exact total parameter counts from your external table here.
    # If set to None, the script will continue computing it dynamically.
    exact_params_map = {
        "FNN": None,               # Dynamic (depends on trace length)
        "HERQULES_Net_rmf": None, 
        "Arxiv240618807FNN": None, 
        "KLiNQ_Student": None,     # Dynamic (depends on trace length)
        "Transformer": None, 
        "CNN": None,                # Hardcoded to exactly match m=8 (1k params)
    }

    results = []

    for idx, row in models_df.iterrows():
        model_name = row['model_name']
        model_path = row['model_path']
        trained_length = row['trained_length']
        hparams = row.get('hparams', {})
        
        if model_name not in mean5_acc_map_500:
            continue
            
        for length, acc_map in [(500, mean5_acc_map_500), (200, mean5_acc_map_200)]:
            acc = acc_map[model_name]
            
            try:
                # 1. FNN
                if model_name == "FNN":
                    model = SingleQubitFNN(input_size=length * 2, output_size=32)
                    
                # 2. HERQULES Net_rmf
                elif model_name == "HERQULES_Net_rmf":
                    model = Net_rmf()
                    
                # 3. Arxiv (QubiCML)
                elif model_name == "Arxiv240618807FNN":
                    model = Arxiv240618807FNN()
                    
                # 4. Transformer
                elif model_name == "Transformer":
                    patch_sz = hparams.get('patch_size', 10)
                    embedding_dim = hparams.get('embedding_dim', 128)
                    num_heads = hparams.get('num_heads', 8)
                    num_layers = hparams.get('num_layers', 4)
                    
                    if pd.isna(patch_sz): patch_sz = 10
                    if pd.isna(embedding_dim): embedding_dim = 128
                    if pd.isna(num_heads): num_heads = 8
                    if pd.isna(num_layers): num_layers = 4
                    
                    state_dict = torch.load(model_path, map_location=DEVICE)
                    if 'cls_token' in state_dict:
                        embedding_dim = state_dict['cls_token'].shape[2]
                    if 'patch_embedding.weight' in state_dict:
                        patch_sz = state_dict['patch_embedding.weight'].shape[2]
                    elif 'patch_embed.projection.weight' in state_dict:
                        shape = state_dict['patch_embed.projection.weight'].shape
                        if len(shape) == 2:
                            patch_sz = shape[1] // 2
                        elif len(shape) == 3:
                            patch_sz = shape[2]
                    if 'transformer_encoder.layers.0.norm1.weight' in state_dict:
                        embedding_dim = state_dict['transformer_encoder.layers.0.norm1.weight'].shape[0]
                        
                    layer_indices = set()
                    for key in state_dict.keys():
                        m = re.match(r'transformer_encoder\.layers\.(\d+)\.', key)
                        if m:
                            layer_indices.add(int(m.group(1)))
                    if layer_indices:
                        num_layers = max(layer_indices) + 1
                        
                    if embedding_dim % num_heads != 0:
                        num_heads = 4 

                    model = QubitClassifierTransformer(
                        num_classes=32, 
                        patch_size=int(patch_sz),
                        embedding_dim=int(embedding_dim),
                        num_heads=int(num_heads),
                        num_layers=int(num_layers),
                        dropout=0.1
                    )
                    
                # 5. CNN
                elif model_name == "CNN":
                    m_param = hparams.get('m_param', 16)
                    if pd.isna(m_param): m_param = 16
                    
                    state_dict = torch.load(model_path, map_location=DEVICE)
                    if 'conv1.weight' in state_dict:
                        m_param = state_dict['conv1.weight'].shape[0]

                    model = CNN(in_channels=10, m_param=int(m_param), num_qubits=NUM_QUBITS)
                    
                # 6. KLiNQ Student
                elif model_name == "KLiNQ_Student":
                    target_length_klinq = 5
                    bin_size = max(1, length // target_length_klinq)
                    n_bins = length // bin_size
                    student_input_size = 2 * length + 2 * n_bins + 1
                    model = KLiNQStudentModel(input_size=student_input_size)
                    
                num_params = sum(p.numel() for p in model.parameters())
                
                # Since Arxiv and KLiNQ models are per-qubit, the total parameters for the 5-qubit 
                # classification task is 5 times the parameters of a single model.
                total_params = num_params * 5 if model_name in ["Arxiv240618807FNN", "KLiNQ_Student"] else num_params
                
                if exact_params_map.get(model_name) is not None:
                    total_params = exact_params_map[model_name]

                results.append({
                    "model_name": model_name,
                    "trained_length": length,
                    "num_parameters_per_model": num_params,
                    "total_parameters_5_qubits": total_params,
                    "mean_5_accuracy": acc,
                })
                
            except Exception as e:
                logger.error(f"Error computing params for {model_name} at length {length}: {e}")

    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.drop_duplicates(subset=["model_name", "trained_length", "total_parameters_5_qubits"])
        os.makedirs(CSV_DIR, exist_ok=True)
        out_path = os.path.join(CSV_DIR, "model_params_and_accuracy.csv")
        df_res.to_csv(out_path, index=False)
        logger.info(f"Saved parameter and accuracy results to {out_path}")
        print(df_res)

        plot_pareto_front(out_path)

def _compute_total_params(model_name, length, hparams, model_path):
    """Instantiate the architecture for (model_name, length) and return total
    parameters across the 5-qubit system. Returns None for unknown model_name."""
    hparams = hparams or {}
    if model_name == "FNN":
        model = SingleQubitFNN(input_size=length * 2, output_size=32)
    elif model_name == "HERQULES_Net_rmf":
        model = Net_rmf()
    elif model_name == "Arxiv240618807FNN":
        model = Arxiv240618807FNN()
    elif model_name == "Transformer":
        patch_sz = hparams.get('patch_size', 10)
        embedding_dim = hparams.get('embedding_dim', 128)
        num_heads = hparams.get('num_heads', 8)
        num_layers = hparams.get('num_layers', 4)
        if pd.isna(patch_sz): patch_sz = 10
        if pd.isna(embedding_dim): embedding_dim = 128
        if pd.isna(num_heads): num_heads = 8
        if pd.isna(num_layers): num_layers = 4

        state_dict = torch.load(model_path, map_location=DEVICE)
        if 'cls_token' in state_dict:
            embedding_dim = state_dict['cls_token'].shape[2]
        if 'patch_embedding.weight' in state_dict:
            patch_sz = state_dict['patch_embedding.weight'].shape[2]
        elif 'patch_embed.projection.weight' in state_dict:
            shape = state_dict['patch_embed.projection.weight'].shape
            if len(shape) == 2:
                patch_sz = shape[1] // 2
            elif len(shape) == 3:
                patch_sz = shape[2]
        if 'transformer_encoder.layers.0.norm1.weight' in state_dict:
            embedding_dim = state_dict['transformer_encoder.layers.0.norm1.weight'].shape[0]

        layer_indices = set()
        for key in state_dict.keys():
            m = re.match(r'transformer_encoder\.layers\.(\d+)\.', key)
            if m:
                layer_indices.add(int(m.group(1)))
        if layer_indices:
            num_layers = max(layer_indices) + 1
        if embedding_dim % num_heads != 0:
            num_heads = 4

        model = QubitClassifierTransformer(
            num_classes=32,
            patch_size=int(patch_sz),
            embedding_dim=int(embedding_dim),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            dropout=0.1,
        )
    elif model_name == "CNN":
        m_param = hparams.get('m_param', 16)
        if pd.isna(m_param): m_param = 16
        state_dict = torch.load(model_path, map_location=DEVICE)
        if 'conv1.weight' in state_dict:
            m_param = state_dict['conv1.weight'].shape[0]
        model = CNN(in_channels=10, m_param=int(m_param), num_qubits=NUM_QUBITS)
    elif model_name == "KLiNQ_Student":
        # Prefer inferring input_size from the checkpoint: evaluate_all() builds
        # a 1011-feature input at length=500 (flat trace + binned + MF scalar),
        # but other training scripts use smaller inputs (e.g. 31 or 201). Reading
        # fc1.weight.shape[1] is authoritative for the actual saved weights.
        state_dict = torch.load(model_path, map_location=DEVICE)
        if 'fc1.weight' in state_dict:
            student_input_size = state_dict['fc1.weight'].shape[1]
        else:
            target_length_klinq = 5
            bin_size = max(1, length // target_length_klinq)
            n_bins = length // bin_size
            student_input_size = 2 * length + 2 * n_bins + 1
        model = KLiNQStudentModel(input_size=student_input_size)
    else:
        return None

    num_params = sum(p.numel() for p in model.parameters())
    return num_params * 5 if model_name in ["Arxiv240618807FNN", "KLiNQ_Student"] else num_params


def summarize_with_gmean(csv_path=None):
    """Read the truncated_evaluation_results.csv produced by evaluate_all(),
    collapse per-qubit binary rows into one row per (model_name, trained_length,
    evaluated_length), and write a new CSV with geometric-mean fidelity over
    all 5 qubits and over 4 qubits (excluding qubit 1), plus param counts."""
    if csv_path is None:
        csv_path = os.path.join(CSV_DIR, "truncated_evaluation_results.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Cannot find {csv_path}; run evaluate_all() first.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.error("Results CSV is empty.")
        return

    qcols_all = [f"qubit_{i}_accuracy" for i in range(5)]
    qcols_4 = [f"qubit_{i}_accuracy" for i in [0, 2, 3, 4]]

    # Per-qubit models (Arxiv, KLiNQ_Student) emit one row per target qubit with
    # the other slots NaN; max collapses them. Multi-qubit models already have
    # all 5 populated in a single row, so max is a no-op for them.
    agg = (
        df.groupby(["model_name", "trained_length", "evaluated_length"])[qcols_all]
        .max()
        .reset_index()
    )

    missing = agg[qcols_all].isna().any(axis=1)
    if missing.any():
        bad = agg.loc[missing, ["model_name", "trained_length", "evaluated_length"]]
        logger.warning(
            f"{int(missing.sum())} row(s) have missing per-qubit accuracies; "
            f"gmean will be NaN there. Examples:\n{bad.head().to_string(index=False)}"
        )

    agg["gmean_5"] = gmean(agg[qcols_all].values, axis=1)
    agg["gmean_4_excl_q1"] = gmean(agg[qcols_4].values, axis=1)

    # Param count is per (model_name, trained_length) — independent of evaluated_length.
    models_df = get_saved_models()
    params_cache = {}
    for _, mrow in models_df.iterrows():
        key = (mrow["model_name"], int(mrow["trained_length"]))
        if key in params_cache:
            continue
        try:
            params_cache[key] = _compute_total_params(
                mrow["model_name"], int(mrow["trained_length"]),
                mrow.get("hparams", {}), mrow["model_path"],
            )
        except Exception as e:
            logger.warning(f"Failed param count for {key}: {e}")
            params_cache[key] = None

    agg["total_parameters_5_qubits"] = agg.apply(
        lambda r: params_cache.get((r["model_name"], int(r["trained_length"]))), axis=1
    )

    out_path = os.path.join(CSV_DIR, "truncated_evaluation_gmean_summary.csv")
    agg.to_csv(out_path, index=False)
    logger.info(f"Saved gmean summary to {out_path}")
    print(agg)
    return agg


if __name__ == "__main__":
    #evaluate_all()
    #evaluate_params_and_accuracy()
    summarize_with_gmean(os.path.join(CSV_DIR, "truncated_evaluation_results.csv"))  # [unify]
