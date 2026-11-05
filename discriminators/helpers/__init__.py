"""
helpers/__init__.py
===================
Convenient package-level imports for helper functions and utilities.

Usage:
    from helpers import get_mf, accuracy, QubitTraceDataset
    from helpers.herqules_helpers import demodulate_multiplexed_traces
"""

# HERQULES-specific helpers
from helpers.herqules_helpers import (
    get_train_val_and_test_set,
    get_data,
    demodulate_multiplexed_traces,
    distance,
    get_traces,
    get_mf,
)

# Training utilities
from helpers.training_utils import (
    adjust_learning_rate,
    inference,
    accuracy,
)

# Data utilities
from helpers.data_utils import (
    QubitTraceDataset,
    normalize_data,
    normalize_data_inplace,
    normalize_data_forb,
    flatten_iq_dimensions,
    stratified_split,
    apply_mf_rmf,
    custom_hdf5_data_loader,
)

# Data loader
from helpers.data_loader import (
    hdf5_data_load,
)

# CNN helpers
from helpers.cnn_helpers import (
    prepare_cnn_data,
    compute_per_qubit_accuracy,
    evaluate_cnn_predictions,
)

__all__ = [
    # HERQULES
    'get_train_val_and_test_set',
    'get_data',
    'demodulate_multiplexed_traces',
    'distance',
    'get_traces',
    'get_mf',
    # Training
    'adjust_learning_rate',
    'inference',
    'accuracy',
    # Data utils
    'QubitTraceDataset',
    'normalize_data',
    'normalize_data_inplace',
    'normalize_data_forb',
    'flatten_iq_dimensions',
    'stratified_split',
    'apply_mf_rmf',
    # Data loader
    'custom_hdf5_data_loader',
    'hdf5_data_load',
    # CNN helpers
    'prepare_cnn_data',
    'format_labels_for_multitask',
    'compute_per_qubit_accuracy',
    'evaluate_cnn_predictions',
]
