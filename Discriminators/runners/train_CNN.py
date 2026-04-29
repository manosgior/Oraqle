"""
train_CNN.py
============
Training runner for multi-task CNN on qubit state classification (PyTorch).

This script demonstrates the full training pipeline for the PyTorch CNN model:
  1. Load and preprocess downsampled IQ traces
  2. Split into train/validation sets
  3. Train the multi-task CNN using BCEWithLogitsLoss
  4. Evaluate on validation and test sets

Configuration
-------------
Hyperparameters can be modified via the configuration dict at the top.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from networks import CNN
from helpers.cnn_helpers import (
    prepare_cnn_data,
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
    'in_channels': 10,
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
# Main Training Script
# ============================================================================

def main():
    """Main training pipeline."""
    print("=" * 70)
    print("CNN Multi-Task Training Pipeline (PyTorch)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------
    # Load Data
    # --------
    print("\n[1/5] Loading training data...")
    try:
        X_train_raw, y_train_raw = prepare_cnn_data(
            CONFIG['train_path'],
            downsample_factor=CONFIG['downsample_factor'],
            original_length=CONFIG['original_length'],
            num_qubits=CONFIG['num_qubits'],
            time_slice=CONFIG['time_slice'],
            is_test=False,
        )
        print(f"  Loaded: {X_train_raw.shape}, Labels: {y_train_raw.shape}")
    except FileNotFoundError:
        print(f"Data file not found: {CONFIG['train_path']}")
        print("Please ensure the dataset is downloaded or update the path in CONFIG.")
        return

    # --------
    # Train/Val Split
    # --------
    print("\n[2/5] Splitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw,
        y_train_raw,
        test_size=CONFIG['validation_split'],
        random_state=CONFIG['random_state'],
    )

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # --------
    # Build Model
    # --------
    print("\n[3/5] Building CNN model...")
    model = CNN(
        in_channels=CONFIG['in_channels'],
        m_param=CONFIG['m_param'],
        num_qubits=CONFIG['num_qubits']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=CONFIG['decay_step'], gamma=CONFIG['decay_rate']
    )

    # --------
    # Train
    # --------
    print("\n[4/5] Training model...")
    print(f"  Epochs:     {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  LR decay:   {CONFIG['decay_rate']}× every {CONFIG['decay_step']} epochs\n")

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # --------
    # Evaluate
    # --------
    print("\n[5/5] Evaluating model...")
    model.eval()
    
    # Validation set
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            # Apply sigmoid to convert logits to probabilities
            outputs = torch.sigmoid(model(inputs))
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(labels.numpy())
            
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    evaluate_cnn_predictions(
        val_preds, 
        val_targets, 
        dataset_name="Validation", 
        threshold=0.5
    )

    # Test set (if enabled)
    if CONFIG['use_test_set']:
        print("\nLoading test data...")
        try:
            X_test, y_test = prepare_cnn_data(
                CONFIG['test_path'],
                downsample_factor=CONFIG['downsample_factor'],
                original_length=CONFIG['original_length'],
                num_qubits=CONFIG['num_qubits'],
                time_slice=CONFIG['time_slice'],
                is_test=True,
            )
            print(f"  Loaded: {X_test.shape}, Labels: {y_test.shape}")
            
            test_loader = DataLoader(
                TensorDataset(X_test, y_test), 
                batch_size=CONFIG['batch_size'], 
                shuffle=False
            )
            
            test_preds = []
            test_targets = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    outputs = torch.sigmoid(model(inputs))
                    test_preds.append(outputs.cpu().numpy())
                    test_targets.append(labels.numpy())
            
            test_preds = np.vstack(test_preds)
            test_targets = np.vstack(test_targets)
            evaluate_cnn_predictions(
                test_preds, 
                test_targets, 
                dataset_name="Test Set", 
                threshold=0.5
            )
            
        except FileNotFoundError:
             print(f"Test data file not found: {CONFIG['test_path']}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
