"""Primary uptake loss and masked structural-descriptor reconstruction loss."""
from torch.nn import functional as F

def joint_loss(uptake, structures, batch, auxiliary_weight):
    primary = F.mse_loss(uptake.reshape(-1), batch["targets_uptake_scaled"])
    # Natural missingness has no reconstruction target and a zero loss flag.
    flags = batch["loss_mask_flags"]
    auxiliary = ((structures - batch["targets_struct"]).square() * flags).sum() / (flags.sum() + 1e-8)
    return primary + auxiliary_weight * auxiliary, primary, auxiliary
