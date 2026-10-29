"""Mask-aware numerical branch: 11 descriptors + 5 flags → 128 → 64."""
from torch import nn

class NumericalEncoder(nn.Sequential):
    """Sequential indices intentionally retain the existing checkpoint keys."""
    def __init__(self, dropout=0.1, input_dim=16):
        super().__init__(nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU())
