"""
HybridCNN.py
============
Hybrid signal-processing + deep-learning readout discriminator. The "hybrid"
part means most of the work happens BEFORE the network: each raw trace is
already complex-mixed to baseband per qubit and low-pass-filtered with a
windowed-sinc sparse FIR (see helpers/sparse_fir_demod.py). What feeds this
model is therefore a heavily compressed, physically aligned multi-channel
representation; the network only learns small nonlinear corrections on top.

Architecture (matches the published "hybrid CNN" description):

  Input  : (batch, n_channels=10, T_ds)
           - n_channels = 5 qubits * (I, Q) = 10, with channels 2q and 2q+1
             carrying qubit q's I and Q from sparse-FIR demod
           - T_ds = downsampled time length (~12-25 for typical settings)
  Internal reshape to (batch, 1, T_ds, n_channels) so Conv2d with kernel (3,1)
  acts ALONG TIME ONLY, independently for each channel:

     Conv2d(1, m, (3,1), stride=(s,1), pad=(1,0)) -> BN -> ReLU
     Conv2d(m, m, (3,1), stride=(s,1), pad=(1,0)) -> BN -> ReLU
     ResBlock2D : Conv2d(m,m,(3,1)) -> BN -> ReLU -> Conv2d(m,m,(3,1)) -> BN
                  + 1x1 skip if channel mismatch -> ReLU
     GAP over time (dim=2)              -> (batch, m, n_channels)
     flatten                            -> (batch, m * n_channels)
     [per-qubit] 5 x Linear(m * n_channels, n_levels)

The (3,1) asymmetric kernel is the key: it convolves over time but with width
1 along the channel axis, so the conv body NEVER mixes qubit channels. Per-qubit
information stays disentangled until the classification heads.

By default ``n_levels=2`` produces a single sigmoid logit per qubit (output
shape (batch, num_qubits)), matched to ``BCEWithLogitsLoss`` for binary readout.
For ternary (|0>/|1>/|2>) readout, pass ``n_levels=3``: output shape becomes
(batch, num_qubits, n_levels), one CrossEntropyLoss per qubit head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock2D(nn.Module):
    """(3,1)-kernel residual block, time-only conv per channel."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out_ch)
        # 1x1 projection only when channels mismatch (per description).
        self.skip = (nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1))
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        s = self.skip(x)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + s)


class HybridCNN(nn.Module):
    """Hybrid signal-processing + DL multi-qubit readout discriminator.

    Parameters
    ----------
    n_channels : int
        Number of input channels (typ. 5 qubits * 2 IQ = 10). Channel layout
        is (q0_I, q0_Q, q1_I, q1_Q, ..., q4_I, q4_Q).
    m_param : int
        Conv filter width. The whole architecture is "shallow + lightweight"
        per the description, so a small value (e.g. 8) is the intended regime.
    num_qubits : int
        Number of output qubits (one classification head each). Default 5.
    n_levels : int
        Classes per qubit head. 2 -> 1 sigmoid logit per qubit (binary readout,
        the published configuration). 3 -> 3 softmax logits per qubit head
        (ternary / qutrit readout: |0>, |1>, |2>).
    strides : tuple[int, int]
        Time-axis strides for the two initial convs. Default (2, 2) -> 4x
        temporal compression on top of the sparse-FIR downsampling. The
        channel-axis stride is fixed at 1 to preserve channel independence.
    """

    def __init__(self, n_channels: int = 10, m_param: int = 8,
                 num_qubits: int = 5, n_levels: int = 2,
                 strides: tuple = (2, 2)):
        super().__init__()
        self.n_channels = n_channels
        self.m_param = m_param
        self.num_qubits = num_qubits
        self.n_levels = n_levels

        self.conv1 = nn.Conv2d(1, m_param, kernel_size=(3, 1),
                                stride=(strides[0], 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(m_param)
        self.conv2 = nn.Conv2d(m_param, m_param, kernel_size=(3, 1),
                                stride=(strides[1], 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(m_param)

        self.res = _ResBlock2D(m_param, m_param)

        # GAP over time -> (batch, m, n_channels); flatten -> (batch, m*n_channels).
        # The description's "compact latent embedding vector" + "five independent
        # ... classifiers, each corresponding to a distinct qubit": one shared
        # flat embedding feeds 5 independent per-qubit heads.
        head_in = m_param * n_channels
        head_out = 1 if n_levels == 2 else n_levels
        self.heads = nn.ModuleList(
            [nn.Linear(head_in, head_out) for _ in range(num_qubits)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_channels, T_ds), matching the classic CNN's layout
            so this model is a drop-in replacement at the trainer level.

        Returns
        -------
        torch.Tensor
            (batch, num_qubits) raw sigmoid logits when ``n_levels=2``;
            (batch, num_qubits, n_levels) softmax logits when ``n_levels>=3``.
        """
        # (batch, n_channels, T_ds) -> (batch, 1, T_ds, n_channels): Conv2d
        # with kernel (3,1) then acts ALONG TIME (the H axis) and looks at
        # only one channel at a time (W axis).
        x = x.transpose(1, 2).unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res(x)

        # GAP over the temporal (H) axis only — channels are preserved.
        x = x.mean(dim=2)                      # (batch, m, n_channels)
        x = x.reshape(x.shape[0], -1)          # (batch, m * n_channels)

        outs = [h(x) for h in self.heads]
        if self.n_levels == 2:
            return torch.cat(outs, dim=1)      # (batch, num_qubits)
        return torch.stack(outs, dim=1)        # (batch, num_qubits, n_levels)
