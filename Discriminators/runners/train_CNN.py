"""
train_CNN.py
============
Training runner for multi-task CNN on qubit state classification.

This script demonstrates the full training pipeline for the CNN model:
  1. Load and preprocess downsampled IQ traces
  2. Split into train/validation sets
  3. Train the multi-task CNN (one output per qubit)
  4. Evaluate on validation and test sets

Configuration
-------------
Hyperparameters can be modified via the configuration dict at the top.
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import callbacks

from networks import CNN, build_cnn
from helpers.cnn_helpers import (
    prepare_cnn_data,
    format_labels_for_multitask,
    evaluate_cnn_predictions,
)


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Data Parameters
    'downsample_factor': 20,           # DS ratio: 1000 → 50 steps
    'original_length': 1000,            # Original trace length
    'num_qubits': 5,                    # Number of qubits
    'time_slice': (0, 1000),            # Use all time indices
    'train_path': '/home/sandra/Qubit_5Channel_ds20_train.h5',
    'test_path': '/home/sandra/Qubit_5Channel_ds20_test.h5',

    # Model Parameters
    'm_param': 8,                       # Model width/depth
    'learning_rate': 5e-4,              # Initial learning rate
    'decay_rate': 0.95,                 # LR decay multiplier per step
    'decay_step': 3,                    # Decay every N epochs

    # Training Parameters
    'validation_split': 0.35,           # Train/val ratio
    'epochs': 30,
    'batch_size': 64,
    'random_state': 42,
    'use_test_set': True,               # Evaluate on test set
}


# ============================================================================
# Learning Rate Scheduler
# ============================================================================

def step_decay(epoch, initial_lr=CONFIG['learning_rate']):
    """Step decay learning rate scheduler.

    Reduces learning rate by a factor every N epochs.

    Args:
        epoch (int): Current epoch number.
        initial_lr (float): Starting learning rate.

    Returns:
        float: Learning rate for this epoch.
    """
    return initial_lr * (
        CONFIG['decay_rate'] ** (epoch // CONFIG['decay_step'])
    )


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training pipeline."""
    print("=" * 70)
    print("CNN Multi-Task Training Pipeline")
    print("=" * 70)

    # --------
    # Load Data
    # --------
    print("\n[1/5] Loading training data...")
    X_train_raw, y_train_raw = prepare_cnn_data(
        CONFIG['train_path'],
        downsample_factor=CONFIG['downsample_factor'],
        original_length=CONFIG['original_length'],
        num_qubits=CONFIG['num_qubits'],
        time_slice=CONFIG['time_slice'],
        is_test=False,
    )
    print(f"  Loaded: {X_train_raw.shape}, Labels: {y_train_raw.shape}")

    # --------
    # Train/Val Split
    # --------
    print("\n[2/5] Splitting train/validation...")
    X_train, X_val, y_train_bits, y_val_bits = train_test_split(
        X_train_raw,
        y_train_raw,
        test_size=CONFIG['validation_split'],
        random_state=CONFIG['random_state'],
    )

    # Format labels for multi-task learning
    y_train_dict = format_labels_for_multitask(y_train_bits, CONFIG['num_qubits'])
    y_val_dict = format_labels_for_multitask(y_val_bits, CONFIG['num_qubits'])

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")

    # --------
    # Build Model
    # --------
    print("\n[3/5] Building CNN model...")
    model = build_cnn(
        input_shape=X_train.shape[1:],
        m_param=CONFIG['m_param'],
        num_qubits=CONFIG['num_qubits'],
        learning_rate=CONFIG['learning_rate'],
    )
    print(model.summary())

    # --------
    # Setup Callbacks
    # --------
    lr_scheduler = callbacks.LearningRateScheduler(step_decay, verbose=0)
    callbacks_list = [lr_scheduler]

    # --------
    # Train
    # --------
    print("\n[4/5] Training model...")
    print(f"  Epochs:     {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  LR decay:   {CONFIG['decay_rate']}× every {CONFIG['decay_step']} epochs\n")

    history = model.fit(
        X_train,
        y_train_dict,
        validation_data=(X_val, y_val_dict),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks_list,
        verbose=1,
    )

    # --------
    # Evaluate
    # --------
    print("\n[5/5] Evaluating model...")

    # Validation set
    val_predictions = model.predict(X_val, verbose=0)
    # Stack outputs: [q0_out, q1_out, ...] → (N, 5)
    val_preds_stacked = np.hstack(val_predictions)
    evaluate_cnn_predictions(
        val_preds_stacked,
        y_val_bits,
        dataset_name="Validation",
        threshold=0.5,
    )

    # Test set (if enabled)
    if CONFIG['use_test_set']:
        print("\nLoading test data...")
        X_test, y_test_bits = prepare_cnn_data(
            CONFIG['test_path'],
            downsample_factor=CONFIG['downsample_factor'],
            original_length=CONFIG['original_length'],
            num_qubits=CONFIG['num_qubits'],
            time_slice=CONFIG['time_slice'],
            is_test=True,
        )
        print(f"  Loaded: {X_test.shape}, Labels: {y_test_bits.shape}")

        test_predictions = model.predict(X_test, verbose=0)
        test_preds_stacked = np.hstack(test_predictions)
        evaluate_cnn_predictions(
            test_preds_stacked,
            y_test_bits,
            dataset_name="Test Set",
            threshold=0.5,
        )

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
