"""Assemble independently defined components with the original parameter names."""
import torch
from torch import nn
from .models import (build_text_encoder, encode_text, NumericalEncoder,
                     UptakePredictionHead, StructuralReconstructionHead, fuse_embeddings)
from .losses import joint_loss  # Backward-compatible import for existing callers.

class HydroBERTMTL(nn.Module):
    def __init__(self, cfg, backbone=None):
        super().__init__()
        self.variant = cfg.get("variant", "full")
        self.bert, self.bert_fc = build_text_encoder(cfg, backbone)
        for key, expected in {"text_embedding_dim":128,"numerical_hidden_dim":128,"numerical_embedding_dim":64,
                              "prediction_hidden_dim":128,"reconstruction_hidden_dim":64}.items():
            if cfg.get(key, expected) != expected:
                raise ValueError(f"This manuscript architecture requires {key}={expected}")
        self.num_mlp = NumericalEncoder(cfg["dropout"], 11 if self.variant == "numerical_only" else 16)
        self.fusion = UptakePredictionHead(cfg["dropout"])
        self.aux_head = StructuralReconstructionHead()

    def forward(self, input_ids, attention_mask, num_feats, input_mask_flags):
        if self.variant == "numerical_only":
            text = num_feats.new_zeros((len(num_feats), 128))
        else:
            text = encode_text(self.bert, self.bert_fc, input_ids, attention_mask)
        if self.variant == "text_only":
            numeric = num_feats.new_zeros((len(num_feats), 64))
        else:
            numeric_input = num_feats if self.variant == "numerical_only" else torch.cat((num_feats, input_mask_flags), dim=1)
            numeric = self.num_mlp(numeric_input)
        shared = fuse_embeddings(text, numeric)
        return self.fusion(shared), self.aux_head(shared)
