import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import hls4ml


# ==========================================
# 1. CONFIGURATION
# ==========================================
DOWNSAMPLE_FACTOR = 20
NUM_QUBITS = 5
ORIGINAL_LENGTH = 1000  # Total points before downsampling
USE_TEST_SET = True     # Set to True to run final evaluation on test file

# --- Model & Training Params ---
M_PARAM = 8                 # Model depth/width
START_LR = 5e-4            # Starting learning rate
DECAY_RATE = 0.95           # Decay multiplier
DECAY_STEP = 3              # Decay happens every X epochs

# --- Time Windowing (Original Scale 0-1000) ---
# Example: (0, 500) = first half, (500, 1000) = second half, (0, None) = all
TIME_SLICE = (0, 1000)

# Paths
# [unify] was BASE_DIR = "/home/felix/Projects/mcmit/model/". The Qubit_5Channel_ds20_*.h5 files are
# made by discriminators/helpers/extract_data.py into $ORAQLE_WORK/cnn (see dataset/README.md).
BASE_DIR = os.environ.get("ORAQLE_CNN_DATA", os.path.join(os.environ.get("ORAQLE_WORK", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "work")), "cnn")) + "/"
TRAIN_PATH = os.path.join(BASE_DIR, f"Qubit_5Channel_ds{DOWNSAMPLE_FACTOR}_train.h5")
TEST_PATH  = os.path.join(BASE_DIR, f"Qubit_5Channel_ds{DOWNSAMPLE_FACTOR}_test.h5")

# ==========================================
# 2. DATA PROCESSING
# ==========================================
def prepare_data(path, is_test=False):
    """Loads H5, converts labels to bits, reshapes, and slices time."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    with h5py.File(path, "r") as hf:
        key_suffix = "test" if is_test else "train"
        X = hf[f"X_{key_suffix}"][:]
        y = hf[f"y_{key_suffix}"][:]

    # Labels (int) → bit matrix (Samples, 5)
    y_bits = np.array([[(label >> i) & 1 for i in range(NUM_QUBITS)] for label in y])

    # Initial reshape to downsampled steps: (Samples, DS, 1, 10)
    total_ds_steps = ORIGINAL_LENGTH // DOWNSAMPLE_FACTOR
    X_reshaped = X.reshape(-1, total_ds_steps, 1, 10)

    # Scale TIME_SLICE indices (0-1000) to Downsampled indices (0-DS)
    orig_start, orig_end = TIME_SLICE
    start_ds = orig_start // DOWNSAMPLE_FACTOR
    end_ds = (orig_end // DOWNSAMPLE_FACTOR) if orig_end is not None else None

    X_sliced = X_reshaped[:, start_ds:end_ds, :, :]

    return X_sliced, y_bits

def format_labels_as_dict(y_matrix):
    """Formats bit matrix into a dict for Keras multi-output."""
    return {f"q{i}": y_matrix[:, i] for i in range(NUM_QUBITS)}

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
def residual_block_2d(x, filters, kernel_size=(3, 1)):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)

def build_model(input_shape, m_param):
    inputs = layers.Input(shape=input_shape)

    # Downsampling in time via strides
    x = layers.Conv2D(m_param, (3, 1), strides=(2, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(m_param, (3, 1), strides=(2, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block_2d(x, m_param)
    x = layers.GlobalAveragePooling2D()(x)

    # One output branch per qubit
    outputs = [
        layers.Dense(1, activation='sigmoid', name=f"q{i}")(x)
        for i in range(NUM_QUBITS)
    ]

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=START_LR),
        loss={f"q{i}": "binary_crossentropy" for i in range(NUM_QUBITS)},
        metrics={f"q{i}": "accuracy" for i in range(NUM_QUBITS)}
    )
    return model

# ==========================================
# 4. TRAINING & EVALUATION LOGIC
# ==========================================
def run_evaluation(model, X, y_matrix, name="Dataset"):
    pred_outputs = model.predict(X, verbose=0)
    # Reassemble individual qubit outputs into a matrix (N, NUM_QUBITS)
    preds = np.hstack([(p > 0.5).astype(int) for p in pred_outputs])

    print(f"\n[Evaluation: {name}]")

    qubit_accs = []
    for i in range(NUM_QUBITS):
        acc = np.mean(preds[:, i] == y_matrix[:, i])
        qubit_accs.append(acc)
        print(f"  Qubit {i} Accuracy: {acc*100:.2f}%")

    global_acc = np.mean(np.all(preds == y_matrix, axis=1))
    fidelity = np.exp(np.mean(np.log(np.array(qubit_accs) + 1e-12)))

    print(f"  Global Accuracy:  {global_acc*100:.2f}%")
    print(f"  Fidelity (G-Mean): {fidelity*100:.2f}%")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
# Load Training Data
X_raw, y_raw = prepare_data(TRAIN_PATH, is_test=False)

print(f"--- Configuration ---")
print(f"Original Slice Indices: {TIME_SLICE}")
print(f"Downsampled Steps:      {X_raw.shape[1]}")
print(f"M_Param (Width):        {M_PARAM}")
print(f"LR Schedule:            Every {DECAY_STEP} epochs, decay {DECAY_RATE}\n")

# Split Training/Validation
X_train, X_val, y_train_mat, y_val_mat = train_test_split(
    X_raw, y_raw, test_size=0.35, random_state=42
)

y_train_dict = format_labels_as_dict(y_train_mat)
y_val_dict = format_labels_as_dict(y_val_mat)

# Build Model
model = build_model(X_train.shape[1:], M_PARAM)
print(model.summary())

# LR Scheduler Callback
def step_decay(epoch):
    return START_LR * (DECAY_RATE ** (epoch // DECAY_STEP))

lr_callback = callbacks.LearningRateScheduler(step_decay, verbose=0)

# Train
print("Starting Training...")
model.fit(
    X_train, y_train_dict,
    validation_data=(X_val, y_val_dict),
    epochs=30,
    batch_size=64,
    callbacks=[lr_callback],
    verbose=1
)

# Final Evaluations
run_evaluation(model, X_val, y_val_mat, name="Validation")

if USE_TEST_SET:
    print("\nLoading Test Set...")
    X_test, y_test_mat = prepare_data(TEST_PATH, is_test=True)
    run_evaluation(model, X_test, y_test_mat, name="Test Set")

model.save('cnn-model.keras')
