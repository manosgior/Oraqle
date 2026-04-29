"""
networks/__init__.py
====================
Convenient package-level imports for all neural network architectures.

Usage:
    from networks import Net_rmf, SingleQubitFNN_Baseline, Transformer
"""

# HERQULES networks
from networks.HERQULES import Net, Net_rmf

# Qubic (ArXiv) network
from networks.Qubic import Arxiv240618807FNN

# SingleQubitFNN variants
from networks.SingleQubitFNN import SingleQubitFNN, SingleQubitFNN_Baseline
from networks.SingleQubitFNN_StudentModel import SingleQubitFNN_StudentModel

# KLiNQ models
from networks.KLiNQ_TeacherModel import KLiNQTeacherModel
from networks.KLiNQ_StudentModel import KLiNQStudentModel

# Transformer
from networks.Transfomer import (
    QubitClassifierTransformer,
    PatchEmbedding,
    PositionalEncoding,
)

# CNN
from networks.CNN import CNN, build_cnn

__all__ = [
    # HERQULES
    'Net',
    'Net_rmf',
    # Qubic
    'Arxiv240618807FNN',
    # SingleQubitFNN
    'SingleQubitFNN',
    'SingleQubitFNN_Baseline',
    'SingleQubitFNN_StudentModel',
    # KLiNQ
    'KLiNQTeacherModel',
    'KLiNQStudentModel',
    # Transformer
    'QubitClassifierTransformer',
    'PatchEmbedding',
    'PositionalEncoding',
    # CNN
    'CNN',
    'build_cnn',
]
