# Quick Reference Guide - Refactored Oraqle/Discriminators

## 🎯 Find What You Need

### Neural Networks (5 Architectures)

| Network | File | Main Class | Purpose |
|---------|------|-----------|---------|
| **Qubic** (ArXiv) | `networks/Qubic.py` | `Arxiv240618807FNN` | Single-qubit binary discrimination |
| **HERQULES** | `networks/HERQULES.py` | `Net`, `Net_rmf` | 5-qubit MF-based classification |
| **Transformer (ViT)** | `networks/Transfomer.py` | `QubitClassifierTransformer` | Vision-transformer for IQ traces |
| **SingleQubitFNN** | `networks/SingleQubitFNN.py` | `SingleQubitFNN`, `SingleQubitFNN_Baseline` | Parametric + fixed baseline FNN |
| **KLiNQ** (Teacher) | `networks/KLiNQ_TeacherModel.py` | `KLiNQTeacherModel` | Knowledge distillation teacher |
| **KLiNQ** (Student) | `networks/KLiNQ_StudentModel.py` | `KLiNQStudentModel` | Tiny FPGA-deployable student |
| **SingleQubitFNN** (Student) | `networks/SingleQubitFNN_StudentModel.py` | `SingleQubitFNN_StudentModel` | Alternative student model |

---

### Helper Functions by Category

#### 🔧 HERQULES-Specific (`helpers/herqules_helpers.py`)
```python
from helpers.herqules_helpers import (
    # Data loading & splitting
    get_train_val_and_test_set(),
    get_data(),
    
    # Demodulation
    demodulate_multiplexed_traces(),
    
    # Trace processing
    get_traces(),
    distance(),
    
    # Matched filters
    get_mf(),
)
```

#### 🏋️ Training Utilities (`helpers/training_utils.py`)
```python
from helpers.training_utils import (
    adjust_learning_rate(),  # Step-decay schedule
    inference(),             # Run inference on dataloader
    accuracy(),              # Overall + per-qubit accuracy
)
```

#### 📊 Data Utilities (`helpers/data_utils.py`)
```python
from helpers.data_utils import (
    QubitTraceDataset,           # PyTorch Dataset wrapper
    normalize_data(),            # Z-score normalization
    normalize_data_forb(),       # Frobenius norm
    flatten_iq_dimensions(),     # Reshape (N,T,2) → (N,2T)
    stratified_split(),          # Class-balanced split
    apply_mf_rmf(),              # Apply MF/RMF to traces
)
```

#### 📂 Data Loading (`helpers/data_loader.py`)
```python
from helpers.data_loader import (
    custom_hdf5_data_loader(),  # Efficient HDF5 loading
    hdf5_data_load(),
)
```

#### ⚙️ Configuration (`helpers/config.py`)
```python
from helpers.config import (
    # Hyperparameters, thresholds, etc.
)
```

---

### Training & Execution

#### 📚 Trainers (`trainers/`)
- `SingleQubitFNNTrainer.py` — Standard supervised training
- `KnowledgeDistillationTrainer_SingleQubitFNN.py` — Distillation for SingleQubitFNN
- `KnowledgeDistillationTrainer_KLiNQ.py` — KLiNQ distillation
- *Optional*: `HERQULESTrainer.py` (full pipeline)
- *Optional*: `BaselineTrainer.py` (baseline model)

#### 🚀 Runners (`runners/`)
- `train_SingleQubitFNN.py` — Entry point for SingleQubitFNN training
- `train_KD_with_SingleQubitFNN.py` — Knowledge distillation
- `train_KD_with_KLiNQ_TeacherStudent.py` — KLiNQ distillation

---

## 💡 Common Tasks

### Task 1: Train a New Model with HERQULES Features
```python
import torch as T
from torch.utils.data import DataLoader

from networks.HERQULES import Net_rmf
from helpers.training_utils import accuracy, adjust_learning_rate
from helpers.herqules_helpers import get_mf, get_traces
from helpers.data_utils import QubitTraceDataset

# 1. Get purified traces
qubit_traces, filtered_indices = get_traces(num_qubits=5)

# 2. Compute matched filters
mf, threshold = get_mf(qubit_traces[1]['traces_0'], qubit_traces[1]['traces_1'])

# 3. Create dataset & loader
dataset = QubitTraceDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32)

# 4. Initialize model & optimizer
model = Net_rmf()
optimizer = T.optim.Adam(model.parameters(), lr=1e-4)

# 5. Training loop
for epoch in range(100):
    lr = adjust_learning_rate(1e-4, optimizer, epoch)
    for batch in loader:
        # ... training step ...
    
    # Evaluate
    acc, acc_per_qubit = accuracy(model, val_loader)
    print(f"Epoch {epoch}: Acc={acc:.4f}, Per-qubit={acc_per_qubit}")
```

### Task 2: Use a Pre-trained Baseline Model
```python
from networks.SingleQubitFNN import SingleQubitFNN_Baseline
from helpers.data_utils import QubitTraceDataset

# Load model
model = SingleQubitFNN_Baseline()
model.load_state_dict(T.load('checkpoint.pth'))

# Prepare data
dataset = QubitTraceDataset(X_test, y_test)
loader = DataLoader(dataset, batch_size=64)

# Evaluate
acc, acc_per_qubit = accuracy(model, loader)
```

### Task 3: Run Knowledge Distillation with KLiNQ
```python
from networks.KLiNQ_TeacherModel import KLiNQTeacherModel
from networks.KLiNQ_StudentModel import KLiNQStudentModel
from trainers.KnowledgeDistillationTrainer_KLiNQ import KnowledgeDistillationTrainer

# Initialize models
teacher = KLiNQTeacherModel(input_size=1000, output_size=32)
student = KLiNQStudentModel(input_size=31)

# Create trainer
trainer = KnowledgeDistillationTrainer(teacher, student, device='cuda')

# Train
trainer.train(train_loader, val_loader, epochs=50, temperature=4.0, alpha=0.7)
```

### Task 4: Preprocess & Demodulate Raw Traces
```python
from helpers.herqules_helpers import demodulate_multiplexed_traces
from helpers.data_utils import normalize_data

# Raw multiplexed traces
raw_traces = np.load('raw_iq.npy')  # Shape: (N, T, 2)

# Demodulate
freq_readout = np.array([-64.729e6, -25.366e6, 24.79e6, 70.269e6, 127.282e6])
demod = demodulate_multiplexed_traces(
    raw_traces,
    freq_readout,
    sampling_rate=500e6,
    filter_cutoff=10e6
)

# Normalize
X_norm, stats = normalize_data(X_train)
```

---

## 📦 Import Cheat Sheet

```python
# ============ NETWORKS ============
from networks.Qubic import Arxiv240618807FNN
from networks.HERQULES import Net, Net_rmf
from networks.Transformer import QubitClassifierTransformer, PatchEmbedding
from networks.SingleQubitFNN import SingleQubitFNN, SingleQubitFNN_Baseline
from networks.KLiNQ_TeacherModel import KLiNQTeacherModel
from networks.KLiNQ_StudentModel import KLiNQStudentModel
from networks.SingleQubitFNN_StudentModel import SingleQubitFNN_StudentModel

# ============ HELPERS ============
from helpers.herqules_helpers import (
    get_mf, get_traces, demodulate_multiplexed_traces,
    get_train_val_and_test_set, get_data, distance
)
from helpers.training_utils import accuracy, adjust_learning_rate, inference
from helpers.data_utils import QubitTraceDataset, normalize_data, flatten_iq_dimensions
from helpers.data_loader import custom_hdf5_data_loader
from helpers.config import *

# ============ TRAINERS ============
from trainers.SingleQubitFNNTrainer import SingleQubitFNNTrainer
from trainers.KnowledgeDistillationTrainer_SingleQubitFNN import KnowledgeDistillationTrainer
from trainers.KnowledgeDistillationTrainer_KLiNQ import KnowledgeDistillationTrainer
```

---

## 🔍 Architecture Comparison

| Aspect | HERQULES | Qubic | Transformer | KLiNQ |
|--------|----------|-------|-------------|-------|
| **Input dim** | 5 or 10 | 2 | 500×2 | 31–1000 |
| **Purpose** | Multi-qubit MF | Single-qubit | Raw trace | Distilled |
| **FPGA-friendly** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **Requires MF prep** | ✅ Yes | ✅ Yes | ❌ No | ⚠️ Mixed |
| **Layers** | 3 | 3 | 4+ | 2 |
| **Qubit scope** | 5 qubits | 1 qubit | 5 qubits | 1 qubit |

---

## 📝 File Organization Map

```
helpers/
  ├─ config.py                      # Config & hyperparameters
  ├─ data_loader.py                 # HDF5 I/O
  ├─ data_utils.py                  # Preprocessing, normalisation
  ├─ nn_utils.py                    # NN utilities
  ├─ herqules_helpers.py      ✨ NEW    # HERQULES functions
  └─ training_utils.py        ✨ NEW    # Common training utils

networks/
  ├─ Qubic.py                       # Qubic single-qubit FNN
  ├─ HERQULES.py              ✨ NEW    # HERQULES Net & Net_rmf
  ├─ Transformer.py                 # ViT-inspired transformer
  ├─ SingleQubitFNN.py        ✨ ENHANCED  # Parametric + Baseline
  ├─ SingleQubitFNN_StudentModel.py
  ├─ KLiNQ_TeacherModel.py
  └─ KLiNQ_StudentModel.py

trainers/
  ├─ SingleQubitFNNTrainer.py
  ├─ KnowledgeDistillationTrainer_SingleQubitFNN.py
  └─ KnowledgeDistillationTrainer_KLiNQ.py

runners/
  ├─ train_SingleQubitFNN.py
  ├─ train_KD_with_SingleQubitFNN.py
  └─ train_KD_with_KLiNQ_TeacherStudent.py

Data/
  └─ (Notebooks, datasets, etc.)

[ROOT FILES]
├─ HERQULES.py                 (Now uses refactored imports)
├─ matched_filter.py           (Matched filter computations)
├─ test.py
├─ REFACTORING_SUMMARY.md      ✨ NEW
└─ QUICK_REFERENCE.md          ✨ NEW (This file)
```

---

## 🚀 Quick Start Examples

### Load and Evaluate HERQULES Model
```bash
# In Python
python3 -c "
from networks.HERQULES import Net_rmf
import torch as T

model = Net_rmf()
model.load_state_dict(T.load('herqules_rmf.pth'))
model.eval()

# Dummy input: 10-dim (5 MF + 5 RMF)
x = T.randn(1, 10)
logits = model(x)
print('Output shape:', logits.shape)  # (1, 32)
"
```

### Train Baseline on Raw Traces
```bash
python3 runners/train_SingleQubitFNN.py
```

### Run KLiNQ Distillation
```bash
python3 runners/train_KD_with_KLiNQ_TeacherStudent.py
```

---

## 📞 Support

**For questions about:**
- **Network architectures** → See `networks/` and docstrings
- **HERQULES pipeline** → See `helpers/herqules_helpers.py` + `REFACTORING_SUMMARY.md`
- **Data handling** → See `helpers/data_utils.py` + `helpers/data_loader.py`
- **Training loops** → See `trainers/` and `helpers/training_utils.py`

---

**Last Updated**: 2025  
**Status**: ✅ Refactoring Complete
