"""
SingleQubitFNN.py
=================
Fully-connected neural network architectures for single-qubit and multi-qubit
state discrimination from raw or processed IQ traces.

Two variants are provided:

1. **SingleQubitFNN** — Parametric multi-layer FNN with dynamic hidden layer sizing.
   Accepts variable input and output sizes; hidden layers are automatically scaled.
   Includes BatchNorm1d and Dropout for regularisation.

2. **SingleQubitFNN_Baseline** — Fixed 3-layer MLP (1000→500→250→32).
   Serves as the upper-bound accuracy reference for the HERQULES pipeline by
   training directly on full raw IQ traces (no feature extraction).
"""

import torch as T
import torch.nn as nn


class SingleQubitFNN(nn.Module):
    """Parametric multi-layer FNN with adaptive hidden layer sizing.

    Takes variable input and output dimensions and automatically configures
    two hidden layers with sizes based on the input dimension.
    Includes BatchNorm1d for normalisation and Dropout(0.5) for regularisation.

    Architecture::

        Input → Linear(in → h1) → BatchNorm1d → ReLU → Dropout(0.5)
              → Linear(h1 → h2) → BatchNorm1d → ReLU → Dropout(0.5)
              → Linear(h2 → out)

    where hidden layer sizes are computed as:
        h1 = max(input_size // 2, 64) if input_size > 50 else input_size
        h2 = h1 // 2

    Parameters
    ----------
    input_size : int
        Dimensionality of the input feature vector.
    output_size : int
        Number of output classes (logits).
    """
    def __init__(self, input_size, output_size):
        hidden_s_1 = max(input_size // 2, 64) if input_size > 50 else input_size
        hidden_s_2 = hidden_s_1 // 2

        hidden_size = [hidden_s_1, hidden_s_2]
        print("Hidden Layer Size:", hidden_s_1, hidden_s_2)

        super(SingleQubitFNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], output_size)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, output_size).
        """
        x = self.relu(self.bn1(self.l1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.l2(x)))
        x = self.dropout(x)
        x = self.l3(x)
        return x


class SingleQubitFNN_Baseline(T.nn.Module):
    """Baseline fully-connected neural network for raw-trace 5-qubit classification.

    A simple 3-layer MLP that takes the first 1000 raw IQ samples (500 I values
    concatenated with 500 Q values) as its input and maps them to logits over
    the 32 possible 5-qubit states.

    This model serves as the upper-bound reference point; it requires no
    feature engineering but is impractical for FPGA deployment due to its
    1 000-dimensional input. It demonstrates the maximum accuracy achievable
    when training directly on raw traces without pre-processing.

    Architecture::

        Input (1000) → Linear(1000→500) → ReLU
                     → Linear(500→250) → ReLU
                     → Linear(250−32)

    Weights are initialised with Xavier uniform and biases are zeroed.

    Returns
    -------
    torch.Tensor
        Logits of shape (batch_size, 32) over all possible 5-qubit states.
    """
    def __init__(self):
        super(SingleQubitFNN_Baseline, self).__init__()
        self.hid1 = T.nn.Linear(1000, 500)
        self.hid2 = T.nn.Linear(500, 250)
        self.oupt = T.nn.Linear(250, 32)

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
            Input tensor of shape (batch_size, 1000) containing the first 1000
            raw IQ samples (500 I values and 500 Q values).

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
