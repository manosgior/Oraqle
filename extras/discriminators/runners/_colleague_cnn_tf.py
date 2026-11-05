"""Colleague's reference Hybrid CNN (TensorFlow/Keras), VERBATIM architecture +
preprocessing. Only changes vs the file she sent:
  - data paths -> /data/five_qubit_data/{DRaw_C_Tr_v0-001, DRaw_C_Te_v0-002}
  - global seed set for a representative run
  - evaluate() also reports geomean fidelity (bit-acc AND assignment-fidelity)
    and the run dumps them to optimization_reports/colleague_cnn_tf.csv
Run in the oraqle-tf image (tensorflow-cpu).
"""
import os as _os, sys as _sys  # [unify] repo layout: discriminators/ and extras/discriminators/ on sys.path
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'discriminators'))
_sys.path.insert(1, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from disc_paths import (RAW_TRAIN, RAW_TEST, THREE_LEVEL_LABELS, CNN_DATA, REPORTS_OUT, MODELS_OUT,  # [unify]
                        RESULTS_DISC, CHECKPOINTS, ensure_dir)
import os, csv
import h5py
import numpy as np
from scipy.signal import firwin
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

tf.random.set_seed(1); np.random.seed(1)

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLING_RATE = 500e6
NUM_QUBITS = 5
NUM_STATES = 32
CHUNK_SIZE = 5000
DOWNSAMPLE_FACTOR = 40
RAW_TIME_SLICE = (0, 1000)

M_PARAM = 16
START_LR = 5e-4
DECAY_RATE = 0.95
DECAY_STEP = 3
BATCH_SIZE = 64
EPOCHS = 30

TRAIN_CFG = {
    "file_path": str(RAW_TRAIN),
    "dataset_X": "X_train",
    "dataset_y": "y_train",
    "samples_per_state": 15000,
}
TEST_CFG = {
    "file_path": str(RAW_TEST),
    "dataset_X": "X_test",
    "dataset_y": "y_test",
    "samples_per_state": 35000,
}


def calibrate(cfg):
    print("Step 1: Calibrating frequencies using Train Dataset...", flush=True)
    with h5py.File(cfg["file_path"], "r") as hf:
        X_cal_list, y_cal_list = [], []
        for s in range(NUM_STATES):
            start_idx = s * cfg["samples_per_state"] + 5000
            X_tmp = hf[cfg["dataset_X"]][start_idx: start_idx + 500]
            s0, s1 = RAW_TIME_SLICE
            X_tmp = X_tmp[:, s0:s1, :]
            X_cal_list.append(X_tmp)
            y_cal_list.append(hf[cfg["dataset_y"]][start_idx: start_idx + 500])
    X_cal = np.concatenate(X_cal_list, axis=0)
    y_cal = np.concatenate(y_cal_list).astype(int)
    Xc = X_cal[:, :, 0] + 1j * X_cal[:, :, 1]
    freqs = np.fft.fftfreq(Xc.shape[1], d=1 / SAMPLING_RATE)
    params = []
    for i in range(NUM_QUBITS):
        mask0 = ((y_cal >> i) & 1) == 0
        mask1 = ((y_cal >> i) & 1) == 1
        fft0 = np.mean(np.abs(np.fft.fft(Xc[mask0], axis=1)), axis=0)
        fft1 = np.mean(np.abs(np.fft.fft(Xc[mask1], axis=1)), axis=0)
        best_f = freqs[np.argmax(np.abs(fft1 - fft0))]
        m0 = np.mean(Xc[mask0]); m1 = np.mean(Xc[mask1])
        theta = -np.angle(m1 - m0)
        params.append({"f": best_f, "theta": theta})
        print(f"  Q{i}: {best_f/1e6:.2f} MHz | theta={theta:.2f}", flush=True)
    return params


def preprocess_dataset_efficient(cfg, params, ds_factor):
    print(f"\nPreprocessing dataset: {cfg['file_path']}...", flush=True)
    numtaps = 101; cutoff_freq = 5e6
    taps = firwin(numtaps, cutoff_freq, fs=SAMPLING_RATE)
    half_tap = numtaps // 2
    with h5py.File(cfg["file_path"], "r") as hf:
        N = hf[cfg["dataset_X"]].shape[0]
        s0, s1 = RAW_TIME_SLICE
        actual_s1 = s1 if s1 is not None else hf[cfg["dataset_X"]].shape[1]
        target_indices = np.arange(0, (actual_s1 - s0), ds_factor)
        T_ds = len(target_indices)
        X_out = np.zeros((N, T_ds, 1, NUM_QUBITS * 2), dtype=np.float32)
        y_out = hf[cfg["dataset_y"]][:]
        for start in range(0, N, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, N)
            X_chunk = hf[cfg["dataset_X"]][start:end, s0:actual_s1, :]
            Xc = X_chunk[:, :, 0] + 1j * X_chunk[:, :, 1]
            t = np.arange(Xc.shape[1]) / SAMPLING_RATE
            channels = []
            for p in params:
                mixed = Xc * np.exp(-1j * 2 * np.pi * p["f"] * t)
                filtered_rot = np.zeros((mixed.shape[0], T_ds), dtype=complex)
                for i, idx in enumerate(target_indices):
                    w_start = idx - half_tap; w_end = idx + half_tap + 1
                    if w_start < 0 or w_end > mixed.shape[1]:
                        chunk_segment = np.pad(
                            mixed[:, max(0, w_start): min(mixed.shape[1], w_end)],
                            ((0, 0), (max(0, -w_start), max(0, w_end - mixed.shape[1]))))
                    else:
                        chunk_segment = mixed[:, w_start:w_end]
                    filtered_rot[:, i] = np.dot(chunk_segment, taps[::-1])
                rot = filtered_rot * np.exp(1j * p["theta"])
                channels.append(rot)
            X_stack = np.stack(channels, axis=2)
            X_out[start:end, :, 0, :] = np.concatenate([X_stack.real, X_stack.imag], axis=2)
            print(f"  Progress: {end/N:6.1%}", end="\r", flush=True)
    print("\n  Preprocessing done.", flush=True)
    return X_out, y_out


def prepare_labels(y):
    return np.array([[(label >> i) & 1 for i in range(NUM_QUBITS)] for label in y])


def format_labels(y):
    return {f"q{i}": y[:, i] for i in range(NUM_QUBITS)}


def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1))(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)


def build_model(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(M_PARAM, (3, 1), strides=(2, 1), padding="same")(inp)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(M_PARAM, (3, 1), strides=(2, 1), padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = residual_block(x, M_PARAM)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = [layers.Dense(1, activation="sigmoid", name=f"q{i}")(x) for i in range(NUM_QUBITS)]
    model = models.Model(inp, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=START_LR),
                  loss={f"q{i}": "binary_crossentropy" for i in range(NUM_QUBITS)},
                  metrics={f"q{i}": "accuracy" for i in range(NUM_QUBITS)})
    return model


def evaluate(model, X, y, name="Dataset"):
    preds = model.predict(X, verbose=0)
    preds = np.hstack([(p > 0.5).astype(int) for p in preds])
    accs = [(preds[:, i] == y[:, i]).mean() for i in range(NUM_QUBITS)]
    # assignment fidelity per qubit: 1 - (P(pred0|true1)+P(pred1|true0))/2
    afid = []
    for i in range(NUM_QUBITS):
        m1 = y[:, i] == 1; m0 = y[:, i] == 0
        fnr = (preds[m1, i] == 0).mean(); fpr = (preds[m0, i] == 1).mean()
        afid.append(1 - (fnr + fpr) / 2)
    global_acc = np.mean(np.all(preds == y, axis=1))
    g_acc = float(np.prod(accs) ** (1 / NUM_QUBITS))
    g_afid = float(np.prod(afid) ** (1 / NUM_QUBITS))
    print(f"\n--- Evaluation Metrics: {name} ---", flush=True)
    for i in range(NUM_QUBITS):
        print(f"  Q{i}  bitacc {accs[i]*100:.2f}%  afid {afid[i]*100:.2f}%", flush=True)
    print(f"  Global (all-5) accuracy : {global_acc*100:.2f}%", flush=True)
    print(f"  F5Q geomean  bitacc {g_acc:.4f}  | afid {g_afid:.4f}", flush=True)
    return accs, afid, g_acc, g_afid, global_acc


if __name__ == "__main__":
    params = calibrate(TRAIN_CFG)
    X_raw, y_raw = preprocess_dataset_efficient(TRAIN_CFG, params, DOWNSAMPLE_FACTOR)
    y_bits = prepare_labels(y_raw)
    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_bits, test_size=0.2, random_state=42)
    model = build_model(X_train.shape[1:]); model.summary()

    def step_decay(epoch):
        return START_LR * (DECAY_RATE ** (epoch // DECAY_STEP))
    lr_cb = callbacks.LearningRateScheduler(step_decay)
    print("\nStarting model training...", flush=True)
    model.fit(X_train, format_labels(y_train),
              validation_data=(X_val, format_labels(y_val)),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=[lr_cb])

    v_accs, v_afid, v_g, v_ga, v_gl = evaluate(model, X_val, y_val, name="Validation Set")
    X_test, y_test_raw = preprocess_dataset_efficient(TEST_CFG, params, DOWNSAMPLE_FACTOR)
    y_test_bits = prepare_labels(y_test_raw)
    t_accs, t_afid, t_g, t_ga, t_gl = evaluate(model, X_test, y_test_bits, name="Test Set")

    n_params = model.count_params()
    out = str(ensure_dir(REPORTS_OUT) + "/colleague_cnn_tf.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "Q0", "Q1", "Q2", "Q3", "Q4", "F5Q_bitacc", "F5Q_afid", "global_acc", "n_params"])
        w.writerow(["val_bitacc"] + [f"{a:.4f}" for a in v_accs] + [f"{v_g:.4f}", f"{v_ga:.4f}", f"{v_gl:.4f}", n_params])
        w.writerow(["val_afid"] + [f"{a:.4f}" for a in v_afid] + [f"{v_g:.4f}", f"{v_ga:.4f}", f"{v_gl:.4f}", n_params])
        w.writerow(["test_bitacc"] + [f"{a:.4f}" for a in t_accs] + [f"{t_g:.4f}", f"{t_ga:.4f}", f"{t_gl:.4f}", n_params])
        w.writerow(["test_afid"] + [f"{a:.4f}" for a in t_afid] + [f"{t_g:.4f}", f"{t_ga:.4f}", f"{t_gl:.4f}", n_params])
    print(f"\n[colleague-cnn] total params {n_params}", flush=True)
    print(f"[colleague-cnn] wrote {out}", flush=True)
    print("[colleague-cnn] done", flush=True)
