"""
KLiNQ_StudentModel.py
=====================
Student network used in the KLiNQ knowledge-distillation pipeline for
single-qubit dispersive readout.

KLiNQ paper context
--------------------
KLiNQ (Knowledge-Light Neural-network Qubit-readout) trains this small
student model via knowledge distillation from a large teacher (e.g.
``SingleQubitFNN`` with layers [1000, 500, 250] or [1000, 64, 32]).

The *key innovation* of KLiNQ is the input representation of the student:
instead of taking the full flattened raw IQ trace (1000 features for a
500-sample trace), the student receives a compact, hand-crafted feature vector
that consists of:

  1. **Full-trace flattened IQ**  (2 × trace_length values, normalised)
     — Preserves the rich temporal structure seen by the teacher.

  2. **Time-averaged IQ**  (2 × target_length values, normalised)
     — Mimics the averaging that a hardware integrator would perform,
       producing a much lower-dimensional summary of the trace.

  3. **Matched-Filter (MF) output**  (1 scalar per qubit)
     — Applies the optimal linear filter (pulse envelope template) to the
       trace and sums I + Q projections.  This feature is known to be
       nearly-optimal for dispersive readout under Gaussian noise, and is
       cheap to compute in hardware or FPGA logic.

The combined input is column-stacked: [flat_trace | avg_trace | mf_output].
For example, with trace_length=500 and target_length=5:
    input_size = 2×500 + 2×5 + 1 = 1011  (varies by configuration)

Architecture
------------
  Input  : (batch_size, input_size)  e.g. (batch, 31) or (batch, 201)

  Layer 1: Linear(input_size, 16) -> BatchNorm1d(16) -> ReLU
  Layer 2: Linear(16, 8)          -> BatchNorm1d(8)  -> ReLU
  Output : Linear(8, 1)
            Raw score (logit).  Apply Sigmoid externally for probability or
            use BCEWithLogitsLoss during distillation.

Note: unlike the teacher, there is no Dropout.  The model is intentionally
very small (< 300 parameters) for FPGA deployment.

Knowledge distillation training
--------------------------------
During distillation the student minimises a composite loss:
    L = α × L_soft + (1 - α) × L_hard

  L_soft : KL-divergence between the student's soft predictions and the
           temperature-softened teacher logits.  Temperature T > 1 spreads the
           teacher's distribution, revealing inter-class similarity.
  L_hard : Binary Cross-Entropy with the true hard labels.

Multiple (T, α) configurations are explored (see helpers/config.py).
"""

import torch.nn as nn
import torch.nn.functional as F


class KLiNQStudentModel(nn.Module):
    """
    Tiny 2-hidden-layer FNN student for FPGA-deployable qubit readout.

    Parameters
    ----------
    input_size : int
        Dimensionality of the combined input feature vector:
        flattened trace + averaged trace + matched-filter scalar.
        Example architectures from the paper: [31, 16, 8, 1] or [201, 16, 8, 1].
        The first number is input_size; the rest are hidden/output layer widths.
    """

    def __init__(self, input_size):
        super(KLiNQStudentModel, self).__init__()

        # --- Layer 1: input_size -> 16 ---
        # BatchNorm before ReLU normalises the very heterogeneous input features
        # (full trace, averaged trace, and MF scalar have different scales).
        self.fc1 = nn.Linear(input_size, 16)
        self.bn1 = nn.BatchNorm1d(16)

        # --- Layer 2: 16 -> 8 ---
        self.fc2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)

        # --- Output layer: 8 -> 1 ---
        # Single logit output.  Apply Sigmoid for probability or use
        # BCEWithLogitsLoss (numerically more stable) during training.
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_size)
            Column-stacked and normalised feature vector:
            [flat_trace | averaged_trace | mf_output].

        Returns
        -------
        torch.Tensor, shape (batch_size, 1)
            Raw logit (no activation).  Apply Sigmoid or threshold at 0 to
            obtain a binary qubit state prediction.
        """
        # Layer 1: Linear -> BN -> ReLU
        x = F.relu(self.bn1(self.fc1(x)))  # (batch, input_size) -> (batch, 16)

        # Layer 2: Linear -> BN -> ReLU
        x = F.relu(self.bn2(self.fc2(x)))  # (batch, 16) -> (batch, 8)

        # Output layer: no activation (raw logit)
        x = self.fc3(x)                    # (batch, 8) -> (batch, 1)
        return x