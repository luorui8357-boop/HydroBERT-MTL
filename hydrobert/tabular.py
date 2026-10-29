"""FT-Transformer and TabR on the manuscript's 11 standardized numerical inputs."""
import torch
from .vendor import ft_transformer as rtdl
# The author Model class is extracted at packaging time without its training script.
from .vendor.tabr_model import Model as AuthorTabR

def build(kind, cfg):
    if kind == "FT-Transformer":
        options = rtdl.FTTransformer.get_default_kwargs(n_blocks=cfg["blocks"])
        return rtdl.FTTransformer(n_cont_features=11, cat_cardinalities=[], d_out=1, **options)
    return AuthorTabR(n_num_features=11, n_bin_features=0, cat_cardinalities=[], n_classes=None,
                      num_embeddings=None, d_main=cfg["width"], d_multiplier=2.,
                      encoder_n_blocks=cfg["encoder_blocks"], predictor_n_blocks=1,
                      mixer_normalization="auto", context_dropout=.2, dropout0=.1, dropout1=0.,
                      normalization="LayerNorm", activation="ReLU", memory_efficient=False)

def forward(model, kind, x, bank_x=None, bank_y=None, index=None, context=32):
    if kind == "FT-Transformer":
        return model(x, None).reshape(-1)
    # Training queries can use other training labels, never outer-test labels.
    y = None
    if index is not None:
        y = bank_y[index]
        keep = torch.ones(len(bank_x), dtype=torch.bool, device=x.device)
        keep[index] = False
        candidates, labels = bank_x[keep], bank_y[keep]
        maximum_context = len(bank_x) - 1
    else:
        candidates, labels = bank_x, bank_y
        maximum_context = len(bank_x)
    if maximum_context < 1:
        raise ValueError("TabR requires at least two training records")
    return model(x_={"num": x}, y=y, candidate_x_={"num": candidates}, candidate_y=labels,
                 context_size=min(context, maximum_context), is_train=index is not None).reshape(-1)
