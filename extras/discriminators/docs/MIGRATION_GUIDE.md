# Migration Guide: Updating Code to Use Refactored Modules

## Overview

This guide helps you migrate existing code from the monolithic `HERQULES.py` structure to the new refactored architecture.

## General Migration Pattern

### Before (Old Pattern)
```python
from HERQULES import SomeFunction, SomeClass, accuracy, get_mf
```

### After (New Pattern)
```python
from networks.HERQULES import SomeNetworkClass
from helpers.herqules_helpers import SomeFunction, get_mf
from helpers.training_utils import accuracy
```

---

## Detailed Migration Examples

### Migration 1: Loading a Network Model

#### Before
```python
from HERQULES import Net_baseline, Net_rmf
model_baseline = Net_baseline()
model_rmf = Net_rmf()
```

#### After
```python
from networks.SingleQubitFNN import SingleQubitFNN_Baseline
from networks.HERQULES import Net_rmf

model_baseline = SingleQubitFNN_Baseline()
model_rmf = Net_rmf()
```

---

### Migration 2: Training Utilities

#### Before
```python
from HERQULES import accuracy, adjust_learning_rate, inference
import torch as T

optimizer = T.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(100):
    lr = adjust_learning_rate(1e-4, optimizer, epoch)
    # ... training ...
    acc, acc_per_qubit = accuracy(model, val_loader)
```

#### After
```python
from helpers.training_utils import accuracy, adjust_learning_rate
import torch as T

optimizer = T.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(100):
    lr = adjust_learning_rate(1e-4, optimizer, epoch)
    # ... training ...
    acc, acc_per_qubit = accuracy(model, val_loader)
```

**Note**: Code is identical! Only imports change.

---

### Migration 3: HERQULES Pipeline Functions

#### Before
```python
from HERQULES import get_mf, get_traces, demodulate_multiplexed_traces
from HERQULES import get_train_val_and_test_set, get_data

# Pre-filtering & characterization
qubit_traces, filtered_indices = get_traces()

# Matched filter
mf_0 = qubit_traces[1]['traces_0']
mf_1 = qubit_traces[1]['traces_1']
mf, threshold = get_mf(mf_0, mf_1)

# Demodulation
demod_traces = demodulate_multiplexed_traces(raw_data, frequencies)
```

#### After
```python
from helpers.herqules_helpers import (
    get_mf, get_traces, demodulate_multiplexed_traces,
    get_train_val_and_test_set, get_data
)

# Everything else stays the same!
qubit_traces, filtered_indices = get_traces()
mf_0 = qubit_traces[1]['traces_0']
mf_1 = qubit_traces[1]['traces_1']
mf, threshold = get_mf(mf_0, mf_1)
demod_traces = demodulate_multiplexed_traces(raw_data, frequencies)
```

---

### Migration 4: Data Utilities

#### Before
```python
from HERQULES import QubitTraceDataset, normalize_data
```

#### After
```python
from helpers.data_utils import QubitTraceDataset, normalize_data
```

---

### Migration 5: Using the New Package-Level Imports (Cleaner!)

#### Option A: Direct imports (most explicit)
```python
from networks.HERQULES import Net_rmf
from helpers.herqules_helpers import get_mf
from helpers.training_utils import accuracy
```

#### Option B: Package-level imports (cleaner)
```python
from networks import Net_rmf
from helpers import get_mf, accuracy
```

Both work! Option B requires the `__init__.py` files.

---

## Checklist for Migration

- [ ] Replace all `from HERQULES import` statements
- [ ] Route imports to appropriate modules:
  - Network classes → `networks/`
  - HERQULES functions → `helpers/herqules_helpers.py`
  - Training utilities → `helpers/training_utils.py`
  - Data utilities → `helpers/data_utils.py`
  - Data loading → `helpers/data_loader.py`
- [ ] Update `trainers/` scripts if they import from `HERQULES`
- [ ] Update `runners/` scripts if they import from `HERQULES`
- [ ] Test imports: `python3 -c "from networks import *; from helpers import *"`
- [ ] Run unit tests to verify functionality

---

## Common Import Updates

### Network Imports
```python
# OLD (remove these)
# from HERQULES import Net_baseline, Net, Net_rmf

# NEW (use these)
from networks.SingleQubitFNN import SingleQubitFNN_Baseline
from networks.HERQULES import Net, Net_rmf
from networks.Qubic import Arxiv240618807FNN
from networks.Transformer import QubitClassifierTransformer
from networks.KLiNQ_TeacherModel import KLiNQTeacherModel
```

### Helper Imports
```python
# OLD
# from HERQULES import (
#     get_mf, get_traces, demodulate_multiplexed_traces,
#     accuracy, adjust_learning_rate, inference,
#     get_train_val_and_test_set
# )

# NEW
from helpers.herqules_helpers import (
    get_mf, get_traces, demodulate_multiplexed_traces,
    get_train_val_and_test_set
)
from helpers.training_utils import (
    accuracy, adjust_learning_rate, inference
)
```

### Data Utilities
```python
# OLD
# from HERQULES import QubitTraceDataset, normalize_data, flatten_iq_dimensions

# NEW
from helpers.data_utils import (
    QubitTraceDataset, normalize_data, flatten_iq_dimensions
)
```

---

## Troubleshooting

### Import Error: "No module named 'networks'"
**Solution**: Make sure you're running from the `Discriminators/` directory or add it to Python path:
```bash
cd Discriminators
python3 your_script.py
# OR
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 your_script.py
```

### Import Error: "Cannot import name 'X' from 'helpers.herqules_helpers'"
**Solution**: Check the function exists and is exported:
```bash
python3 -c "from helpers.herqules_helpers import get_mf; print(get_mf)"
```

### Module circular imports
**Solution**: This shouldn't happen with the new structure. If it does, check:
1. Are helper modules importing each other unnecessarily?
2. Are network modules importing from helpers? (This is OK)
3. Are helpers importing networks? (This should be avoided)

---

## Before & After: Full Example

### Before (HERQULES.py monolith)
```python
#!/usr/bin/env python3
"""train_herqules.py - Old monolithic approach"""

import numpy as np
import torch as T
from torch.utils.data import DataLoader

from HERQULES import (
    Net_rmf,
    get_mf,
    get_traces,
    accuracy,
    adjust_learning_rate,
    QubitTraceDataset,
)

def main():
    # Get purified traces
    qubit_traces, filtered_indices = get_traces(num_qubits=5)
    
    # Compute MF for each qubit
    mf_envelopes = []
    for q in range(1, 6):
        mf, threshold = get_mf(
            qubit_traces[q]['traces_0'],
            qubit_traces[q]['traces_1']
        )
        mf_envelopes.append(mf)
    
    # Setup training
    model = Net_rmf()
    optimizer = T.optim.Adam(model.parameters(), lr=1e-4)
    criterion = T.nn.CrossEntropyLoss()
    
    # Training loop
    dataset = QubitTraceDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32)
    
    for epoch in range(100):
        lr = adjust_learning_rate(1e-4, optimizer, epoch)
        for batch in loader:
            # ... training step ...
            pass
        
        acc, acc_per_qubit = accuracy(model, val_loader)
        print(f"Epoch {epoch}: {acc:.4f}")

if __name__ == '__main__':
    main()
```

### After (Using refactored modules)
```python
#!/usr/bin/env python3
"""train_herqules.py - New refactored approach"""

import numpy as np
import torch as T
from torch.utils.data import DataLoader

# All imports clearly show where functions come from
from networks.HERQULES import Net_rmf
from helpers.herqules_helpers import get_mf, get_traces
from helpers.training_utils import accuracy, adjust_learning_rate
from helpers.data_utils import QubitTraceDataset

def main():
    # Get purified traces
    qubit_traces, filtered_indices = get_traces(num_qubits=5)
    
    # Compute MF for each qubit
    mf_envelopes = []
    for q in range(1, 6):
        mf, threshold = get_mf(
            qubit_traces[q]['traces_0'],
            qubit_traces[q]['traces_1']
        )
        mf_envelopes.append(mf)
    
    # Setup training (unchanged logic)
    model = Net_rmf()
    optimizer = T.optim.Adam(model.parameters(), lr=1e-4)
    criterion = T.nn.CrossEntropyLoss()
    
    # Training loop (unchanged logic)
    dataset = QubitTraceDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32)
    
    for epoch in range(100):
        lr = adjust_learning_rate(1e-4, optimizer, epoch)
        for batch in loader:
            # ... training step ...
            pass
        
        acc, acc_per_qubit = accuracy(model, val_loader)
        print(f"Epoch {epoch}: {acc:.4f}")

if __name__ == '__main__':
    main()
```

**Difference**: Only imports changed! The functional logic is identical.

---

## Testing Your Migration

### Quick Test Script
```python
#!/usr/bin/env python3
"""test_imports.py - Verify all refactored imports work"""

# Test network imports
print("Testing network imports...")
from networks import (
    Net, Net_rmf,
    SingleQubitFNN, SingleQubitFNN_Baseline,
    Arxiv240618807FNN,
    QubitClassifierTransformer,
    KLiNQTeacherModel, KLiNQStudentModel,
)
print("✅ All network imports OK")

# Test helper imports
print("Testing helper imports...")
from helpers import (
    get_mf, get_traces, demodulate_multiplexed_traces,
    accuracy, adjust_learning_rate, inference,
    QubitTraceDataset, normalize_data,
    custom_hdf5_data_loader,
)
print("✅ All helper imports OK")

# Test direct imports
print("Testing direct module imports...")
from helpers.herqules_helpers import distance, get_data, get_train_val_and_test_set
from helpers.training_utils import inference
from helpers.data_utils import flatten_iq_dimensions
print("✅ All direct imports OK")

print("\n✅✅✅ MIGRATION COMPLETE! All imports working.")
```

Run it:
```bash
cd Discriminators
python3 test_imports.py
```

---

## Summary

| Aspect | Old | New | Benefit |
|--------|-----|-----|---------|
| **Organization** | Monolithic | Modular | Easier to maintain |
| **Imports** | `from HERQULES import *` | `from helpers import ...` | Clear ownership |
| **Networks** | Buried in HERQULES.py | `networks/` | Discoverable |
| **Reusability** | Mixed with training | Separated | Easy to reuse |
| **Testing** | Interdependent | Independent | Easier to test |

**Key Principle**: Code logic stays the same; only imports change!

---

**Status**: ✅ Ready for Migration  
**Date**: 2025
