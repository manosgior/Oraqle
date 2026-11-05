"""
HERQULESPlus.py
===============
HERQULES Net_rmf extended with multi-class LDA features and a zero-init
residual MLP.

Architecture
------------
Two parallel paths to the final 32-class logits:

  1. **Backbone** (vanilla HERQULES Net_rmf):
        mf_features (10-D)  ->  Linear(10->10) -> ReLU
                            ->  Linear(10->20) -> ReLU
                            ->  Linear(20->32)

  2. **Residual MLP** (zero-init last layer):
        concat[mf_features, lda_features] (mf_dim + lda_dim)
                            ->  LayerNorm
                            ->  [Linear -> ReLU -> Dropout] x num_hidden_layers
                            ->  Linear(hidden_dim -> num_classes)   [zero-init]

Output:
        logits = backbone(mf_features) + residual_mlp(concat[mf, lda])

At initialization the residual MLP outputs exactly zero (last Linear has both
weight and bias zero-initialised), so HERQULESPlus behaves identically to the
standard HERQULES Net_rmf. The MLP only contributes if training finds the
residual useful -- guaranteeing the model is no worse than HERQULES on the
same MF features at the start of optimisation.

Rationale
---------
HERQULES uses 5 per-qubit binary LDA directions (the matched filters) plus 5
relaxation MF directions -- 10 linear projections of the demodulated trace.
A 32-class problem has up to (num_classes - 1) = 31 informative linear
directions. Joint multi-class LDA on the concatenated per-qubit demodulated
trace recovers these and captures the cross-qubit discriminative structure
that per-qubit marginal MFs cannot see. The LDA scalars are precomputed before
training, so the network only learns the *combination*, not the projection.
"""

import torch
import torch.nn as nn

from networks.HERQULES import Net_rmf


class HERQULESPlus(nn.Module):
    """HERQULES Net_rmf backbone + zero-init residual MLP over LDA features.

    Parameters
    ----------
    mf_dim : int
        Dimensionality of the HERQULES MF + RMF feature vector. Default 10.
    lda_dim : int
        Dimensionality of the precomputed multi-class LDA feature vector. For
        the standard 5-qubit / 32-class problem this is 31 (= num_classes - 1).
    num_classes : int
        Number of output classes. Default 32 (= 2 ** 5).
    hidden_dim : int
        Width of each hidden layer in the residual MLP. Default 64.
    num_hidden_layers : int
        Number of hidden layers in the residual MLP. Must be >= 1. Default 2.
    dropout : float
        Dropout probability between MLP layers. Default 0.1.
    """

    def __init__(self, mf_dim: int = 10, lda_dim: int = 31, num_classes: int = 32,
                 hidden_dim: int = 64, num_hidden_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        self.mf_dim = mf_dim
        self.lda_dim = lda_dim
        self.num_classes = num_classes

        # --- Backbone: vanilla HERQULES Net_rmf (10 -> 10 -> 20 -> 32) ---
        # Kept as the standalone Net_rmf class so a HERQULES checkpoint can be
        # loaded directly via load_backbone(state_dict).
        self.backbone = Net_rmf()

        # --- Residual MLP over [mf_features ; lda_features] ---
        in_dim = mf_dim + lda_dim
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        # Small init gain so intermediate activations start near zero. Combined
        # with the zero-init output layer this makes "residual ~= 0" the path
        # of least resistance: the MLP only drifts away from zero when the
        # extra features actually carry signal beyond what the backbone already
        # explains. Avoids the noise-drift degradation seen on uninformative
        # feature sets.
        for _ in range(num_hidden_layers):
            lin = nn.Linear(prev, hidden_dim)
            nn.init.xavier_uniform_(lin.weight, gain=0.1)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_dim

        # Output projection: zero-init weight AND bias so the residual path
        # contributes exactly zero at step 0. First forward equals backbone(mf).
        out = nn.Linear(prev, num_classes)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)

        self.residual = nn.Sequential(*layers)

    def forward(self, mf_features: torch.Tensor,
                lda_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        mf_features : torch.Tensor, shape (B, mf_dim)
            Precomputed HERQULES MF + RMF scalars (5 + 5 for the standard case).
        lda_features : torch.Tensor, shape (B, lda_dim)
            Precomputed multi-class LDA projection of the concatenated
            per-qubit demodulated traces.

        Returns
        -------
        torch.Tensor, shape (B, num_classes)
            Raw class logits.
        """
        backbone_logits = self.backbone(mf_features)
        residual_logits = self.residual(
            torch.cat([mf_features, lda_features], dim=-1)
        )
        return backbone_logits + residual_logits

    def load_backbone(self, net_rmf_state_dict: dict) -> None:
        """Warm-start the backbone from a pretrained Net_rmf checkpoint."""
        self.backbone.load_state_dict(net_rmf_state_dict)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters so only the residual MLP trains."""
        for p in self.backbone.parameters():
            p.requires_grad = False
