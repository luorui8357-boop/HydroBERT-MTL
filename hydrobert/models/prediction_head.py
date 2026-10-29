"""Hydrogen-uptake head: 192 fused features → 128 → one scaled prediction."""
from torch import nn

class UptakePredictionHead(nn.Sequential):
    def __init__(self, dropout=0.1):
        super().__init__(nn.Linear(192, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
