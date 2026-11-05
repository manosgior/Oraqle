"""
Transfomer.py
=============
Vision-Transformer-inspired architecture for qubit state classification from
raw IQ traces.

Architecture overview
---------------------
This file implements a compact ViT-style encoder. The core idea (borrowed from
the original ViT paper, Dosovitskiy et al. 2020) is to split a 1-D time-series
into fixed-size *patches*, project each patch into a *d*-dimensional embedding
space, prepend a learnable [CLS] token, add sinusoidal positional encodings,
and feed the resulting sequence through a stack of standard Multi-Head
Self-Attention (MHSA) + Feed-Forward blocks.

Data format
-----------
  Input  : (batch_size, trace_length, 2)
            - trace_length : number of time samples (e.g. 500 @ 2 ns per sample)
            - 2            : IQ quadrature channels (I = in-phase, Q = quadrature)
  Output : (batch_size, num_classes)
            Raw logits over all possible qubit states
            (e.g. 32 states for 5-qubit multiplexed readout)

Component breakdown
-------------------
  PatchEmbedding
    Divides the trace into non-overlapping windows of ``patch_size`` samples
    and linearly projects each window to ``embedding_dim`` dimensions.
    A single learnable [CLS] token is prepended; its final representation is
    used for classification (analogous to BERT's [CLS] token).

  PositionalEncoding
    Injects fixed sinusoidal position information so the transformer can
    distinguish patch order.  Uses the classic Vaswani et al. (2017) formula:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  QubitClassifierTransformer
    Full encoder stack:
      1. PatchEmbedding  -> (batch, num_patches+1, embedding_dim)
      2. PositionalEncoding (adds position info, same shape)
      3. N × TransformerEncoderLayer  (MHSA + FFN + LayerNorm + Dropout)
      4. Classification head at position 0 (the [CLS] token)

Default hyper-parameters
------------------------
  patch_size    = 10   -> 500 // 10 = 50 patches
  embedding_dim = 128
  num_heads     = 8    -> head dim = 128 // 8 = 16
  num_layers    = 4    -> stacked encoder layers
  dropout       = 0.1
  FFN hidden    = 4 × embedding_dim = 512  (standard ViT convention)
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Converts a 1-D IQ time-series trace into a sequence of patch embeddings.

    Given a trace of shape (batch_size, trace_length, in_channels):
      1. The trace is split into non-overlapping patches of size ``patch_size``
         along the time dimension (unfold).
      2. Each patch (in_channels × patch_size values) is flattened and linearly
         projected to ``embedding_dim`` dimensions.
      3. A learnable [CLS] token is prepended to the sequence; the final
         sequence length becomes num_patches + 1.

    Parameters
    ----------
    patch_size : int
        Number of time samples per patch.  The trace_length must be divisible
        by patch_size.  Default: 10.
    in_channels : int
        Number of input channels (2 for I and Q).  Default: 2.
    embedding_dim : int
        Output dimensionality of every token (both patches and the CLS token).
        Default: 128.
    """

    def __init__(self, patch_size=10, in_channels=2, embedding_dim=128):
        super().__init__()
        self.patch_size = patch_size
        # Each patch contains patch_size time steps × in_channels values
        self.patch_dim = in_channels * patch_size

        # Linear projection: patch_dim -> embedding_dim
        self.projection = nn.Linear(self.patch_dim, embedding_dim)

        # Learnable [CLS] token (randomly initialised).
        # Shape (1, 1, embedding_dim) -> expanded to (batch_size, 1, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, trace_length, in_channels)

        Returns
        -------
        torch.Tensor, shape (batch_size, num_patches + 1, embedding_dim)
            Token sequence with the [CLS] token at position 0.
        """
        # x.unfold(dim, size, step): slides a window of `size` along `dim`
        # Result shape: (batch_size, num_patches, in_channels, patch_size)
        # Then .flatten(2) merges in_channels and patch_size into patch_dim:
        # -> (batch_size, num_patches, patch_dim)
        x_patched = x.unfold(1, self.patch_size, self.patch_size).flatten(2)

        # Linear projection of each patch to embedding space
        # -> (batch_size, num_patches, embedding_dim)
        x_projected = self.projection(x_patched)

        # Expand CLS token from (1, 1, embedding_dim) to (batch_size, 1, embedding_dim)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # Concatenate CLS token at the front of the patch sequence
        # Final shape: (batch_size, num_patches + 1, embedding_dim)
        x_final = torch.cat((cls_tokens, x_projected), dim=1)

        return x_final


class PositionalEncoding(nn.Module):
    """
    Adds fixed sinusoidal positional encodings to the input token sequence.

    The encoding is computed once and stored as a non-trainable buffer.
    It is added element-wise to the token embeddings so that the transformer
    can distinguish token order (patch position along the trace).

    Formula (Vaswani et al., 2017 "Attention Is All You Need"):
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the token embeddings (= d_model).  Default: 128.
    max_len : int
        Maximum sequence length supported.  Default: 500 (well above the
        expected num_patches + 1).
    """

    def __init__(self, embedding_dim=128, max_len=500):
        super().__init__()

        # position indices: shape (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # div_term: shape (embedding_dim // 2,)
        # log-space computation for numerical stability
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )

        # pe initialised to zeros: shape (max_len, 1, embedding_dim)
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # even dims -> sin
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # odd  dims -> cos

        # Reshape to (1, max_len, embedding_dim) for easy addition (batch first)
        # register_buffer: not a learnable parameter but part of the model state
        self.register_buffer('pe', pe.permute(1, 0, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, seq_len, embedding_dim)

        Returns
        -------
        torch.Tensor, same shape as x (in-place addition of positional encoding)
        """
        # Slice the precomputed encoding to match the actual sequence length
        x = x + self.pe[:, :x.size(1), :]
        return x


class QubitClassifierTransformer(nn.Module):
    """
    Transformer encoder-based classifier for multi-qubit state discrimination.

    This is the top-level model that wires PatchEmbedding + PositionalEncoding
    + Transformer encoder stack + a classification head into a single forward
    pass.

    Architecture
    ------------
    Input  : (batch_size, trace_length, 2)   -- raw IQ trace

    Step 1 - Patch embedding:
        trace_length / patch_size = 50 patches (for trace_length=500, patch_size=10)
        With the [CLS] token: seq_len = 51

    Step 2 - Positional encoding:
        Standard sinusoidal PE added to all 51 tokens.

    Step 3 - Transformer encoder (N layers):
        Each layer applies:
          a) Multi-Head Self-Attention (MHSA) with ``num_heads`` heads
          b) Layer normalisation
          c) 2-layer Feed-Forward Network (FFN) with hidden dim = 4 × embedding_dim
          d) Another layer norm + residual connections (Pre-LN or Post-LN per PyTorch default)
          e) Dropout

    Step 4 - Classification head (on [CLS] token):
        LayerNorm(embedding_dim) -> Linear(embedding_dim, num_classes)
        Returns raw logits (no softmax); use nn.CrossEntropyLoss during training.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  For 5 multiplexed qubits: 2^5 = 32.
    patch_size : int
        Temporal patch size in samples.  Default: 10.
    embedding_dim : int
        Token embedding dimensionality (= d_model).  Default: 128.
    num_heads : int
        Number of attention heads.  Must evenly divide embedding_dim.  Default: 8.
    num_layers : int
        Number of stacked TransformerEncoderLayers.  Default: 4.
    dropout : float
        Dropout probability applied inside the transformer.  Default: 0.1.
    """

    def __init__(self, num_classes=32, patch_size=10, embedding_dim=128,
                 num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        # --- Patch embedding (also adds [CLS] token) ---
        self.patch_embed = PatchEmbedding(patch_size=patch_size,
                                          embedding_dim=embedding_dim)

        # --- Positional encoding ---
        self.pos_encoder = PositionalEncoding(embedding_dim=embedding_dim)

        # --- Transformer encoder ---
        # A single TransformerEncoderLayer bundles MHSA + FFN + LayerNorm + Dropout.
        # batch_first=True keeps the (batch, seq, feature) convention throughout.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,  # Standard ViT: 4× expansion
            dropout=dropout,
            batch_first=True  # Required: our tensors are (batch, seq, feature)
        )
        # Stack num_layers of the above layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                          num_layers=num_layers)

        # --- Classification head ---
        # Applied only to the [CLS] token representation (position 0).
        # LayerNorm stabilises the final representation before projection.
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, trace_length, 2)
            Raw IQ traces.  trace_length must be divisible by patch_size.

        Returns
        -------
        torch.Tensor, shape (batch_size, num_classes)
            Raw class logits (pre-softmax).
        """
        # 1. Patch and embed: (batch, trace_length, 2) -> (batch, num_patches+1, embedding_dim)
        x = self.patch_embed(x)

        # 2. Add positional encoding (same shape)
        x = self.pos_encoder(x)

        # 3. Pass through stacked transformer encoder
        x = self.transformer_encoder(x)

        # 4. Extract the [CLS] token (first position in the sequence)
        #    Its representation aggregates information from all patches via attention.
        cls_token_output = x[:, 0]  # shape: (batch_size, embedding_dim)

        # 5. Classify via linear projection
        return self.classifier(cls_token_output)  # shape: (batch_size, num_classes)


# ---------------------------------------------------------------------------
# Training and evaluation utilities
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Run one full training epoch over a dataloader.

    Performs the standard supervised learning loop:
        zero_grad -> forward -> loss -> backward -> step

    Parameters
    ----------
    model : nn.Module
        The QubitClassifierTransformer (or any compatible model).
    dataloader : DataLoader
        Yields (inputs, labels) batches.
    criterion : nn.Module
        Loss function.  Typically ``nn.CrossEntropyLoss`` for multi-class
        classification.
    optimizer : torch.optim.Optimizer
        Parameter optimiser (e.g. Adam).
    device : torch.device
        Target device ('cpu' or 'cuda').

    Returns
    -------
    float
        Average training loss over all batches in the epoch.
    """
    model.train()  # Enable training mode (activates dropout, batch-norm updates)
    running_loss = 0.0

    for inputs, labels in dataloader:
        # Move data to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. Reset accumulated gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(inputs)

        # 3. Compute loss
        loss = criterion(outputs, labels)

        # 4. Backpropagation: compute gradients
        loss.backward()

        # 5. Gradient descent step: update model weights
        optimizer.step()

        running_loss += loss.item()

    # Return the mean loss across all batches
    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataloader (validation or test set).

    Disables gradient computation for efficiency.

    Parameters
    ----------
    model : nn.Module
        The trained model to evaluate.
    dataloader : DataLoader
        Yields (inputs, labels) batches.
    criterion : nn.Module
        Loss function (same as used during training).
    device : torch.device
        Target device.

    Returns
    -------
    avg_loss : float
        Average loss per batch.
    accuracy : float
        Classification accuracy as a percentage (0–100).
    """
    model.eval()  # Disable dropout; use running stats for batch norm
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # No gradient tracking -> lower memory, faster inference
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Predicted class = argmax of raw logits
            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item()

    accuracy = 100 * correct_predictions / total_samples
    avg_loss = running_loss / len(dataloader)

    return avg_loss, accuracy
