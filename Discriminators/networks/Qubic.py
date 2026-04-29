"""
Arxiv240618807FNN.py
====================
Feedforward Neural Network (FNN) for single-qubit state discrimination,
faithful to the architecture published in arXiv:2406.18807.

Paper context
-------------
arXiv:2406.18807 describes a lightweight FNN classifier for dispersive qubit
readout.  The state discrimination task for a *single* qubit reduces to binary
classification of the integrated IQ point (I, Q) — the time-averaged in-phase
and quadrature components of the demodulated readout signal.

Architecture (from §2 / §6.2 of the paper)
-------------------------------------------
  Input  layer : 2 nodes  — (I, Q) averaged IQ point, min-max normalised to [0, 1]
  Hidden layer 1 : 8 nodes, ReLU activation
  Hidden layer 2 : 4 nodes, ReLU activation
  Output layer : 1 node,  Sigmoid activation  -> P(qubit = |1⟩)

The network is trained with Binary Cross-Entropy loss and the Adam optimiser
(learning rate default 1e-3, as recommended in the paper).

Multiplexed-readout workflow (used in train_arxiv_model.py)
-----------------------------------------------------------
For a 5-qubit multiplexed system, the raw trace (shape [N, trace_len, 2]) is
*demodulated* to isolate each qubit's resonator frequency, then *averaged*
over the full readout window to produce a single (I, Q) point per qubit per
shot.  A separate instance of this network is trained for each of the 5 qubits.

The per-qubit binary label is extracted by bit-shifting the integer state:
    y_q = (y_combined >> q) & 1   for qubit q in {0, 1, 2, 3, 4}

Training details (paper §6.2)
------------------------------
  - Dataset      : 100,000 total; 60,000 train / 40,000 test
  - Normalisation: min-max scaling per column to [0, 1]
  - Batch size   : 64
  - Optimiser    : Adam (lr = 1e-3)
  - Loss         : Binary Cross-Entropy (nn.BCELoss)
  - Epochs       : 40
"""

import torch.nn as nn


class Arxiv240618807FNN(nn.Module):
    """
    Minimal 2-hidden-layer FNN for single-qubit binary readout.

    Input  : (batch_size, 2)  -- (I, Q) min-max normalised to [0, 1]
    Output : (batch_size, 1)  -- probability of qubit being in |1⟩ (Sigmoid)

    The Sigmoid output is thresholded at 0.5 during inference:
        prediction = 1  if output >= 0.5  else  0
    """

    def __init__(self):
        super(Arxiv240618807FNN, self).__init__()

        # --- Layer 1: 2 -> 8, ReLU ---
        # Projects the 2-D IQ point into an 8-dimensional hidden representation.
        self.fc1 = nn.Linear(2, 8)
        self.relu1 = nn.ReLU()

        # --- Layer 2: 8 -> 4, ReLU ---
        # Further compresses the representation.
        self.fc2 = nn.Linear(8, 4)
        self.relu2 = nn.ReLU()

        # --- Output layer: 4 -> 1, Sigmoid ---
        # Outputs P(qubit = |1⟩); use with nn.BCELoss during training.
        self.fc3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, 2)
            Min-max normalised (I, Q) IQ point.

        Returns
        -------
        torch.Tensor, shape (batch_size, 1)
            Probability that the qubit is in |1⟩.
        """
        x = self.relu1(self.fc1(x))   # (batch, 2) -> (batch, 8)
        x = self.relu2(self.fc2(x))   # (batch, 8) -> (batch, 4)
        x = self.sigmoid(self.fc3(x)) # (batch, 4) -> (batch, 1), range [0, 1]
        return x
