"""
training_utils.py
=================
Common utility functions for neural network training and evaluation.

This module provides reusable helpers for:
  - Learning rate scheduling
  - Model inference
  - Accuracy computation
  - Data loading and preparation
"""

import numpy as np
import torch as T


def adjust_learning_rate(
    initial_lr: float,
    optimizer: T.optim.Optimizer,
    epoch: int,
    lr_schedule: list = None
) -> float:
    """Apply step-decay learning-rate schedule.

    Reduces the learning rate by a factor of 10 at each epoch specified in
    ``lr_schedule``. Useful for multi-stage training where early stages benefit
    from high learning rates and later stages need finer convergence.

    Args:
        initial_lr (float): Initial learning rate (for epoch 0).
        optimizer (torch.optim.Optimizer): PyTorch optimizer instance
            whose learning rate will be updated.
        epoch (int): Current training epoch (0-indexed).
        lr_schedule (list): List of epoch thresholds at which to reduce lr by 10×.
            Default: [30, 60, 90].

    Returns:
        float: The learning rate used for the current epoch.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> for epoch in range(100):
        ...     lr = adjust_learning_rate(1e-4, optimizer, epoch)
        ...     # ... training loop ...
    """
    if lr_schedule is None:
        lr_schedule = [30, 60, 90]

    lr = initial_lr
    for threshold in lr_schedule:
        if epoch >= threshold:
            lr *= 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def inference(model: T.nn.Module, dataloader: T.utils.data.DataLoader) -> tuple:
    """Run inference on a batch-iterable dataloader and collect predictions.

    Computes model predictions (after Softmax) and ground-truth labels for all
    batches. Useful for computing accuracy, ROC curves, confusion matrices, etc.

    Args:
        model (torch.nn.Module): Trained neural network in eval mode (or will
            be switched to eval internally).
        dataloader (torch.utils.data.DataLoader): Data loader yielding batches
            of (X, Y) tuples where X has shape (batch_size, *) and Y has shape
            (batch_size,).

    Returns:
        tuple: ``(all_scores, all_labels)`` where

        - **all_scores** (np.ndarray): Shape ``(N_total, num_classes)`` containing
          softmax-normalized predictions for each sample.
        - **all_labels** (np.ndarray): Shape ``(N_total,)`` containing ground-truth
          class labels.
    """
    model.eval()
    all_scores = []
    all_labels = []
    s = T.nn.Softmax(dim=-1)

    with T.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            X = batch[0]
            Y = batch[1]
            oupt = model(X)
            oupt = s(oupt)

            all_scores.extend(oupt.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    model.train()
    return np.array(all_scores), np.array(all_labels)


def accuracy(model: T.nn.Module, dataloader: T.utils.data.DataLoader) -> tuple:
    """Compute overall and per-qubit accuracy for a 5-qubit classification task.

    For a 5-qubit multiplexed readout, the classification task involves
    predicting one of 32 possible basis states (2^5). This function computes:

    1. **Overall accuracy** — fraction of states predicted correctly.
    2. **Per-qubit accuracy** — for each qubit, the fraction of individual qubit
       predictions (obtained by bit-extraction from the state label) that are
       correct.

    Args:
        model (torch.nn.Module): Trained 32-way classifier in eval mode.
        dataloader (torch.utils.data.DataLoader): Data loader for evaluation.

    Returns:
        tuple: ``(overall_acc, per_qubit_acc)`` where

        - **overall_acc** (float): Fraction of correct state predictions ∈ [0, 1].
        - **per_qubit_acc** (list): 5-element list of per-qubit accuracies.
    """
    all_preds, all_labels = inference(model, dataloader)

    pred_indices = np.argmax(all_preds, axis=-1)
    cumulative_acc = np.sum(pred_indices == all_labels) / len(all_labels)

    acc_per_qubit = []
    for _ in range(5):
        pred_qubit = pred_indices % 2
        label_qubit = all_labels % 2
        acc_per_qubit.append(np.sum(pred_qubit == label_qubit) / len(label_qubit))
        pred_indices = pred_indices >> 1
        all_labels = all_labels >> 1

    return cumulative_acc, acc_per_qubit
