"""
HERQULES.py
===========
Primary neural-network classifiers for the HERQULES matched-filter pipeline.

The HERQULES pipeline (Hierarchical Efficient Readout with QUbit Learning via
Ensemble Stages) combines geometric pre-filtering and matched filters to extract
low-dimensional features, which are then classified by compact MLPs.

Two variants are provided:

1. **Net** — Operates on standard matched-filter (MF) outputs only.
   Input: 5 scalars (one MF per qubit)  →  Output: 32-state logits

2. **Net_rmf** — The primary production classifier using both MF and relaxation
   matched-filter (RMF) features.
   Input: 10 scalars (5 MF + 5 RMF)  →  Output: 32-state logits

Both use compact 3-layer MLPs designed to fit efficiently on FPGA hardware
while maintaining high accuracy on 5-qubit readout.
"""

import torch as T
import torch.nn as nn


class Net(T.nn.Module):
    """Compact MLP classifier that operates on standard matched-filter (MF) outputs.

    Takes a 5-dimensional input vector (one MF scalar per qubit) and produces
    logits over the 32 possible 5-qubit basis states.  This architecture is used
    when only the standard MF features are available (i.e. when the RMF is
    disabled via ``run_rmf=False`` in the training pipeline).

    Architecture::

        Input (5) → Linear(5→10) → ReLU
                  → Linear(10→20) → ReLU
                  → Linear(20−32)

    Weights are initialised with Xavier uniform and biases with zeros.

    Returns
    -------
    torch.Tensor
        Logits of shape (batch_size, 32) over all possible 5-qubit states.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(5, 10)
        self.hid2 = T.nn.Linear(10, 20)
        self.oupt = T.nn.Linear(20, 32)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)
        self.relu = T.nn.ReLU()

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 5) containing the matched-filter
            outputs for each of the 5 qubits.

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, 32).
        """
        z = self.hid1(x)
        z = self.relu(z)
        z = self.hid2(z)
        z = self.relu(z)
        z = self.oupt(z)
        return z


class Net_rmf(T.nn.Module):
    """Primary HERQULES neural-network classifier (MF + RMF inputs).

    Accepts a 10-dimensional feature vector (5 standard MF scalars + 5 RMF
    scalars, one per qubit each) and maps them to logits over the 32 possible
    5-qubit states.  This is the main production classifier in the HERQULES
    pipeline when both MF and RMF features are available.

    The relaxation matched-filter (RMF) is specialised for detecting qubit
    relaxation events (|1⟩→|0⟩ during readout), providing complementary
    information to the standard MF that discriminates |0⟩ vs |1⟩ at the end
    of readout.

    Architecture::

        Input (10) → Linear(10→10) → ReLU
                   → Linear(10→20) → ReLU
                   → Linear(20−32)

    The compact 10→ 10→20→32 structure is highly FPGA-friendly: the entire
    forward pass involves only a few hundred multiply-accumulates.

    Weights are initialised with Xavier uniform and biases with zeros.

    Returns
    -------
    torch.Tensor
        Logits of shape (batch_size, 32) over all possible 5-qubit states.
    """
    def __init__(self):
        super(Net_rmf, self).__init__()
        self.hid1 = T.nn.Linear(10, 10)
        self.hid2 = T.nn.Linear(10, 20)
        self.oupt = T.nn.Linear(20, 32)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)
        self.relu = T.nn.ReLU()

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 10) containing the concatenated
            matched-filter and relaxation matched-filter outputs for each of
            the 5 qubits ([MF_1, ..., MF_5, RMF_1, ..., RMF_5]).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, 32).
        """
        z = self.hid1(x)
        z = self.relu(z)
        z = self.hid2(z)
        z = self.relu(z)
        z = self.oupt(z)
        return z
