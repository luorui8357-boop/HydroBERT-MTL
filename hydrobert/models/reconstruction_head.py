"""Auxiliary head: 192 fused features → 64 → five standardized descriptors."""
from torch import nn

class StructuralReconstructionHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 5))
