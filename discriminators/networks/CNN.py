"""
CNN.py
======
Convolutional Neural Network architecture for multi-qubit IQ trace classification.

This module implements a residual CNN designed for efficient feature extraction
from downsampled multi-qubit IQ traces. The architecture uses:
  - Temporal downsampling via strided convolutions
  - Residual blocks for gradient flow
  - Per-qubit binary classifiers
  - Multi-task learning (one output logit per qubit)

Model input: (batch_size, in_channels, time_steps)
Model output: (batch_size, num_qubits) raw logits (apply sigmoid externally)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    """1-D Residual block with batch normalization and ReLU activation.

    Architecture::

        Input → Conv1d(kernel) → BatchNorm → ReLU
              ↓                                    ↓
              +────────────────────────────────────+
                                                   ↓
                          Conv1d(kernel) → BatchNorm → ReLU
                          (Shortcut adapted if needed)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """Initialize residual block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Kernel dimension. Default: 3.
        """
        super(ResidualBlock1D, self).__init__()
        
        # 'same' padding for kernel_size=3 is padding=1
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut projection if channel mismatch
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """Forward pass through residual block."""
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


class CNN(nn.Module):
    """Multi-task CNN for per-qubit binary classification from IQ traces.

    The model performs multi-task learning with one output per qubit.
    Each output produces a raw logit, which can be trained using
    BCEWithLogitsLoss.

    Architecture overview:
      1. Strided Conv1D (m_param filters, stride 2)
      2. Strided Conv1D (m_param filters, stride 2)
      3. Residual block (skip connection)
      4. Global average pooling
      5. num_qubits × Dense(1) outputs

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default: 10.
    m_param : int
        Model depth parameter controlling filter width. Default: 8.
    num_qubits : int
        Number of output qubits (one binary classifier per qubit). Default: 5.
    """

    def __init__(self, in_channels=10, m_param=8, num_qubits=5):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.m_param = m_param
        self.num_qubits = num_qubits

        # Strided convolutions for temporal downsampling
        self.conv1 = nn.Conv1d(in_channels, m_param, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(m_param)

        self.conv2 = nn.Conv1d(m_param, m_param, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(m_param)

        # Residual block
        self.res_block = ResidualBlock1D(m_param, m_param)

        # Per-qubit output layers
        self.output_layers = nn.ModuleList([
            nn.Linear(m_param, 1) for _ in range(num_qubits)
        ])

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, time_steps).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, num_qubits) containing raw logits.
        """
        # Temporal downsampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Residual block
        x = self.res_block(x)

        # Global average pooling (over the time_steps dimension)
        x = x.mean(dim=2)

        # Per-qubit outputs
        outputs = [layer(x) for layer in self.output_layers]
        
        # Concatenate outputs along the features dimension
        # outputs shape will be [batch_size, num_qubits]
        return torch.cat(outputs, dim=1)
