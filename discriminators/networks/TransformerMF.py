"""
TransformerMF.py
================
HERQULES-aided variant of the QubitClassifierTransformer.

Architecture
------------
Identical to QubitClassifierTransformer, except the [CLS]-token representation
is concatenated with a precomputed F-dim HERQULES MF+RMF feature vector
(default F=10: 5 MF scalars + 5 RMF scalars, one per qubit) before the
classification head.

Rationale
---------
The matched filter is the Wiener-optimal linear projection for the |0> vs |1>
binary decision under additive Gaussian noise -- its scalar output is a
sufficient statistic for that single-qubit hypothesis test. Concatenating the
10-D MF+RMF vector to the [CLS] token gives the classifier head a direct
"easy answer" channel while the transformer encoder is free to attend to the
raw trace for non-MF structure (|2> leakage, drift, cross-talk, partial
relaxation at non-canonical times, ...).

Forward signature
-----------------
    model(x, mf_features) -> logits
    x            : (B, T, 2)               raw IQ trace, same as base Transformer
    mf_features  : (B, mf_feature_dim)     precomputed MF + RMF scalars
    logits       : (B, num_classes)

Building mf_features
--------------------
Use trainers.train_HERQULES.build_features on demodulated train/test data:

    from trainers.train_HERQULES import (
        demodulate_all_qubits, compute_all_envelopes, build_features,
    )
    demod_train = demodulate_all_qubits(X_train_raw)
    demod_test  = demodulate_all_qubits(X_test_raw)
    mf_envs, rmf_envs = compute_all_envelopes(demod_train, y_train)
    mf_features_train = build_features(demod_train, mf_envs, rmf_envs, T_full)
    mf_features_test  = build_features(demod_test,  mf_envs, rmf_envs, T_full)

The envelopes must be the same ones the model was trained against. For honest
short-readout evaluation use a truncated length L < T_full when calling
build_features (it truncates both trace and envelope correctly).
"""

import torch
import torch.nn as nn

from networks.Transformer import PatchEmbedding, PositionalEncoding


class QubitClassifierTransformerMF(nn.Module):
    """Transformer encoder with MF+RMF features concatenated at the CLS head.

    Parameters
    ----------
    num_classes : int
        Number of output classes. 32 for 5-qubit multiplexed readout. Default: 32.
    patch_size : int
        Temporal patch size in samples. trace_length must be divisible by this.
        Default: 10.
    embedding_dim : int
        Token embedding dimensionality (= d_model). Default: 128.
    num_heads : int
        Number of attention heads. Must evenly divide embedding_dim. Default: 8.
    num_layers : int
        Number of stacked TransformerEncoderLayers. Default: 4.
    dropout : float
        Dropout probability inside the transformer. Default: 0.1.
    mf_feature_dim : int
        Number of MF+RMF scalar inputs concatenated to the CLS embedding.
        Default: 10 (5 MF + 5 RMF for the standard 5-qubit setup).
    mf_hidden_dim : int or None
        If set, MF features pass through a small MLP
        (LayerNorm -> Linear -> ReLU -> Linear -> LayerNorm) of width
        `mf_hidden_dim` before concatenation. Lets the model learn per-feature
        scale/weighting and balance the MF subspace against the (typically
        larger) trace embedding. If None, MF features are LayerNormed and
        concatenated raw. Default: None.
    """

    def __init__(self, num_classes=32, patch_size=10, embedding_dim=128,
                 num_heads=8, num_layers=4, dropout=0.1,
                 mf_feature_dim=10, mf_hidden_dim=None):
        super().__init__()
        self.mf_feature_dim = mf_feature_dim

        # --- Trace path: identical to QubitClassifierTransformer ---
        self.patch_embed = PatchEmbedding(patch_size=patch_size,
                                          embedding_dim=embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim=embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # --- MF feature path: keep scales sane before concatenation ---
        # MF/RMF scalars are dot products of raw IQ with full-length envelopes;
        # their magnitudes can be orders of magnitude larger than the post-LN
        # transformer embeddings, so a normalization step is essential.
        if mf_hidden_dim is not None:
            self.mf_proj = nn.Sequential(
                nn.LayerNorm(mf_feature_dim),
                nn.Linear(mf_feature_dim, mf_hidden_dim),
                nn.ReLU(),
                nn.Linear(mf_hidden_dim, mf_hidden_dim),
                nn.LayerNorm(mf_hidden_dim),
            )
            mf_out_dim = mf_hidden_dim
        else:
            self.mf_proj = nn.LayerNorm(mf_feature_dim)
            mf_out_dim = mf_feature_dim

        # --- Joint classifier head ---
        head_in = embedding_dim + mf_out_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, num_classes),
        )

    def forward(self, x: torch.Tensor, mf_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, 2)
            Raw IQ trace. T must be divisible by patch_size.
        mf_features : torch.Tensor, shape (B, mf_feature_dim)
            Precomputed HERQULES MF+RMF scalars for this batch.

        Returns
        -------
        torch.Tensor, shape (B, num_classes)
            Raw class logits.
        """
        # 1. Trace path: patches -> positional encoding -> transformer -> CLS
        x = self.patch_embed(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        cls_repr = x[:, 0]                                    # (B, embedding_dim)

        # 2. MF path: normalize (and optionally project) the 10-D feature vector
        mf_repr = self.mf_proj(mf_features)                   # (B, mf_out_dim)

        # 3. Concatenate and classify from the joint representation
        joint = torch.cat([cls_repr, mf_repr], dim=-1)        # (B, head_in)
        return self.classifier(joint)
