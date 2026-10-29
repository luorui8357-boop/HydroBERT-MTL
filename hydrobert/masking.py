"""Natural/artificial masks and standardized numeric input construction."""
import random
import numpy as np
from .prompts import make_prompt

def masks(values, probability=0., rng=None):
    if not 0 <= probability <= 1:
        raise ValueError("Mask probability must be between zero and one")
    natural = ~np.isfinite(np.asarray(values, float)[4:9])
    rng = random if rng is None else rng
    artificial = np.array([rng.random() < probability for _ in range(5)]) & ~natural
    return natural | artificial, artificial

def encode_row(values, scaler_x, probability=0., rng=None):
    values = np.asarray(values, dtype=np.float32)
    flags, loss_flags = masks(values, probability, rng)
    norm = scaler_x.transform(values.reshape(1, -1)).reshape(-1).astype(np.float32)
    targets = np.nan_to_num(norm[4:9], nan=0.).copy()
    model_input = norm.copy()
    model_input[4:9] = np.where(flags, 0., model_input[4:9])
    if not np.isfinite(model_input).all():
        raise ValueError("Non-structural missing values are unsupported")
    return make_prompt(values, flags), model_input, flags.astype(np.float32), targets, loss_flags.astype(np.float32)
