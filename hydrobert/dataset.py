"""PyTorch dataset combining tokenization, numeric features and task targets."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import FEATURES, TARGET
from .masking import encode_row

class HydroDataset(Dataset):
    def __init__(self, df, tokenizer, scaler_x, scaler_y, max_length=128,
                 probability=0., fixed_seed=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer, self.sx, self.sy = tokenizer, scaler_x, scaler_y
        self.max_length, self.probability, self.fixed_seed = max_length, probability, fixed_seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        rng = None if self.fixed_seed is None else random.Random(self.fixed_seed + i)
        text, numeric, flags, structures, loss_flags = encode_row(row[FEATURES].to_numpy(float), self.sx, self.probability, rng)
        tokens = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        out = {k: tokens[k].flatten() for k in ("input_ids", "attention_mask")}
        out.update(num_feats_input=torch.tensor(numeric), input_mask_flags=torch.tensor(flags),
                   targets_struct=torch.tensor(structures), loss_mask_flags=torch.tensor(loss_flags))
        if TARGET in row and np.isfinite(row[TARGET]):
            out["targets_uptake_scaled"] = torch.tensor(float(self.sy.transform([[row[TARGET]]])[0, 0]), dtype=torch.float32)
        return out
