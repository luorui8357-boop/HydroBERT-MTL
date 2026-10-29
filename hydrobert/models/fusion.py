"""Parameter-free concatenation of 128 text features and 64 numerical features."""
import torch

def fuse_embeddings(text, numerical):
    return torch.cat((text, numerical), dim=1)
