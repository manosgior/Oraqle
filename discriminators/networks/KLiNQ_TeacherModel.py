"""
KLiNQ_TeacherModel.py
=====================
Teacher network used in the KLiNQ knowledge-distillation pipeline for
single-qubit dispersive readout.

KLiNQ paper context
--------------------
KLiNQ (Knowledge-Light Neural-network Qubit-readout) is a two-stage
knowledge-distillation framework designed to produce tiny student networks
that fit on an FPGA while retaining near-teacher accuracy:

  Stage 1 — Regular training of a *large* teacher model on full IQ traces.
             The teacher in this project is actually the best-performing
             ``SingleQubitFNN`` model (e.g. layers [1000, 500, 250])
             re-used as a teacher for the KLiNQ student.

  Stage 2 — Knowledge distillation from the teacher to a *small* student.
             The student (KLiNQStudentModel) takes a compact, hand-crafted
             feature vector rather than the full flattened trace.

``KLiNQTeacherModel`` is an intermediate teacher that was explored during the
development of KLiNQ.  It uses a 2-hidden-layer FNN with BatchNorm and
Dropout.  Input comes from a flattened / averaged IQ trace; output is raw
logits over state classes (apply Softmax or use CrossEntropyLoss externally).

Architecture
------------
  Input  : (batch_size, input_size)
            Flattened IQ trace (e.g. input_size = 1000 for a 500-sample trace
            with 2 channels: I and Q).

  Layer 1: Linear(input_size, 64) -> BatchNorm1d(64) -> ReLU -> Dropout(0.3)
  Layer 2: Linear(64, 32)        -> BatchNorm1d(32) -> ReLU -> Dropout(0.3)
  Output : Linear(32, output_size)
            Raw logits.  For 2-class (binary) readout: output_size = 2 (or 1
            with BCELoss).  For multi-state: output_size = num_states.

Design choices
--------------
  - BatchNorm1d is applied *before* ReLU (BN -> ReLU) which helps stabilise
    training with large and varied input magnitudes seen in raw IQ data.
  - Dropout(0.3) provides regularisation for the relatively compact hidden
    layers.
  - Hidden sizes 64 and 32 are fixed constants (chosen empirically).
"""

import torch.nn as nn


class KLiNQTeacherModel(nn.Module):
    """
    2-hidden-layer FNN teacher for the KLiNQ distillation framework.

    Parameters
    ----------
    input_size : int
        Dimensionality of the flattened input feature vector.
        Typically 2 × trace_length (I and Q channels concatenated).
    output_size : int
        Number of output classes (logits).
        Binary readout: 2 (or 1 with BCELoss). Multi-state: 2^n_qubits.
    """

    def __init__(self, input_size, output_size):
        # Fixed hidden layer widths (empirically chosen for this dataset)
        hidden_s_1 = 64
        hidden_s_2 = 32

        hidden_size = [hidden_s_1, hidden_s_2]
        print("Hidden Layer Size:", hidden_s_1, hidden_s_2)

        super(KLiNQTeacherModel, self).__init__()

        # --- Layer 1: input_size -> 64 ---
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])  # Normalises activations per feature

        # --- Layer 2: 64 -> 32 ---
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        # --- Output layer: 32 -> output_size ---
        self.l3 = nn.Linear(hidden_size[1], output_size)

        # Shared activation and regularisation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Applied after each hidden activation

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_size)
            Normalised, flattened IQ trace features.

        Returns
        -------
        torch.Tensor, shape (batch_size, output_size)
            Raw class logits (apply softmax externally or use CrossEntropyLoss).
        """
        # Layer 1: Linear -> BN -> ReLU -> Dropout
        x = self.relu(self.bn1(self.l1(x)))
        x = self.dropout(x)

        # Layer 2: Linear -> BN -> ReLU -> Dropout
        x = self.relu(self.bn2(self.l2(x)))
        x = self.dropout(x)

        # Output layer: no activation (raw logits)
        x = self.l3(x)
        return x
