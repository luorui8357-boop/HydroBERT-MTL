"""SciBERT loading and CLS projection into the 128-dimensional text embedding.

The factory returns the encoder and projection separately so the parent model
retains the original `bert.*` and `bert_fc.*` checkpoint names.
"""
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

def build_text_encoder(cfg, backbone=None):
    bert = backbone if backbone is not None else AutoModel.from_pretrained(
        cfg["backbone"], attn_implementation=cfg.get("attention_implementation", "eager"))
    architecture = (bert.config.hidden_size, bert.config.num_hidden_layers, bert.config.num_attention_heads)
    if architecture != (768, 12, 12):
        raise ValueError("The manuscript uses SciBERT with hidden size 768, 12 layers and 12 attention heads")
    return bert, nn.Linear(768, 128)

def encode_text(bert, projection, input_ids, attention_mask):
    hidden = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return F.gelu(projection(hidden[:, 0, :]))
