"""
CNN.py
======
Convolutional Neural Network architecture for multi-qubit IQ trace classification.

This module implements a residual CNN designed for efficient feature extraction
from downsampled multi-qubit IQ traces. The architecture uses:
  - Temporal downsampling via strided convolutions
  - Residual blocks for gradient flow
  - Per-qubit binary classifiers with sigmoid outputs
  - Multi-task learning (one output per qubit)

Model input: (batch_size, time_steps, 1, channels)
Model output: 5 sigmoid-activated scalars [0, 1] (one per qubit)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np


class ResidualBlock2D(layers.Layer):
    """2-D Residual block with batch normalization and ReLU activation.

    Architecture::

        Input → Conv2D(kernel) → BatchNorm → ReLU
              ↓                                    ↓
              +────────────────────────────────────+
                                                   ↓
                          Conv2D(kernel) → BatchNorm → ReLU
                          (Shortcut adapted if needed)
    """

    def __init__(self, filters, kernel_size=(3, 1), **kwargs):
        """Initialize residual block.

        Parameters
        ----------
        filters : int
            Number of output filters/channels.
        kernel_size : tuple
            Kernel dimensions (height, width). Default: (3, 1) for temporal kernels.
        """
        super(ResidualBlock2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        """Build layer based on input shape."""
        self.conv1 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()

        # Shortcut projection if channel mismatch
        if input_shape[-1] != self.filters:
            self.shortcut_conv = layers.Conv2D(self.filters, (1, 1), padding='same')
        else:
            self.shortcut_conv = None

        self.add = layers.Add()
        self.relu_out = layers.Activation('relu')

    def call(self, x):
        """Forward pass through residual block."""
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        x = self.add([x, shortcut])
        x = self.relu_out(x)

        return x

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
        })
        return config


class CNN(models.Model):
    """Multi-task CNN for per-qubit binary classification from IQ traces.

    The model performs multi-task learning with one output per qubit (5 total).
    Each output is trained with binary cross-entropy loss and uses sigmoid
    activation to produce probabilities in [0, 1].

    Architecture overview:
      1. Strided Conv2D (m_param filters, stride 2)
      2. Strided Conv2D (m_param filters, stride 2)
      3. Residual block (skip connection)
      4. Global average pooling
      5. 5 × Dense(1, sigmoid) outputs

    Parameters
    ----------
    input_shape : tuple
        Input tensor shape (time_steps, 1, channels). Example: (50, 1, 10)
    m_param : int
        Model depth parameter controlling filter width. Default: 8.
    num_qubits : int
        Number of output qubits (one binary classifier per qubit). Default: 5.

    Returns
    -------
    Model output: List of 5 tensors, each shape (batch_size, 1) with sigmoid activation.
    """

    def __init__(self, input_shape, m_param=8, num_qubits=5, **kwargs):
        """Initialize CNN model.

        Parameters
        ----------
        input_shape : tuple
            Shape of input (time_steps, 1, channels).
        m_param : int
            Model width/depth parameter. Default: 8.
        num_qubits : int
            Number of qubits. Default: 5.
        """
        super(CNN, self).__init__(**kwargs)
        self.input_shape_spec = input_shape
        self.m_param = m_param
        self.num_qubits = num_qubits

        # Strided convolutions for temporal downsampling
        self.conv1 = layers.Conv2D(m_param, (3, 1), strides=(2, 1), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(m_param, (3, 1), strides=(2, 1), padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')

        # Residual block
        self.res_block = ResidualBlock2D(m_param)

        # Global pooling
        self.global_pool = layers.GlobalAveragePooling2D()

        # Per-qubit output layers
        self.output_layers = [
            layers.Dense(1, activation='sigmoid', name=f'q{i}')
            for i in range(num_qubits)
        ]

    def call(self, inputs, training=None):
        """Forward pass through the network.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, time_steps, 1, channels).
        training : bool, optional
            Whether in training mode (affects batch norm). Default: None.

        Returns
        -------
        List[tf.Tensor]
            List of 5 output tensors, each (batch_size, 1) with sigmoid values.
        """
        # Temporal downsampling
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        # Residual block
        x = self.res_block(x)

        # Global pooling
        x = self.global_pool(x)

        # Per-qubit outputs
        outputs = [layer(x) for layer in self.output_layers]

        return outputs

    def get_config(self):
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_shape_spec': self.input_shape_spec,
            'm_param': self.m_param,
            'num_qubits': self.num_qubits,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)


def build_cnn(input_shape, m_param=8, num_qubits=5, learning_rate=5e-4):
    """Build and compile a multi-task CNN for qubit classification.

    Parameters
    ----------
    input_shape : tuple
        Shape of input (time_steps, 1, channels). Example: (50, 1, 10).
    m_param : int
        Model width/depth parameter. Default: 8.
    num_qubits : int
        Number of qubits. Default: 5.
    learning_rate : float
        Initial learning rate for Adam optimizer. Default: 5e-4.

    Returns
    -------
    model : CNN
        Compiled Keras model ready for training.

    Example
    -------
    >>> model = build_cnn((50, 1, 10), m_param=8)
    >>> model.fit(X_train, y_train_dict, epochs=30)
    """
    model = CNN(input_shape, m_param=m_param, num_qubits=num_qubits)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={f'q{i}': 'binary_crossentropy' for i in range(num_qubits)},
        metrics={f'q{i}': 'accuracy' for i in range(num_qubits)},
    )

    return model
