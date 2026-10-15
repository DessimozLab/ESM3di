"""Inference-release model API using the shared training architecture."""

from .ESM3di_model import ESM3DiModel
from .inference.model import CNNClassificationHead, ESMWithCNNHead
