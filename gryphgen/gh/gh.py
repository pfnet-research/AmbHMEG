from typing import Tuple

import torch


def gh_channelwise_global_spatial(
    diff1: torch.Tensor,
    diff2: torch.Tensor,
    eps: float = 1e-6,
    only_adjust: str = "diff2",
    return_mask: bool = False,
) -> (
    Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Channel-wise global spatial GH:
      - Reduce over (H,W) only, keep channel dimension.
      - dot/den shapes: [B, C, 1, 1]
      - Apply projection removal per (B,C) when conflict in that channel.

    Args:
        diff1, diff2: [B,C,H,W]
        only_adjust: "diff2" or "both"
        return_mask: returns mask [B,C,1,1]
    """
    assert diff1.shape == diff2.shape, f"shape mismatch: {diff1.shape} vs {diff2.shape}"
    assert diff1.dim() == 4, "expected [B,C,H,W] tensors"
    if only_adjust not in ("diff2", "both"):
        raise ValueError("only_adjust must be 'diff2' or 'both'")

    # reduce over spatial dims only
    dot12 = (diff1 * diff2).sum(dim=(2, 3), keepdim=True)  # [B,C,1,1]
    den11 = (diff1 * diff1).sum(dim=(2, 3), keepdim=True).clamp_min(eps)  # [B,C,1,1]
    den22 = (diff2 * diff2).sum(dim=(2, 3), keepdim=True).clamp_min(eps)  # [B,C,1,1]

    mask = (dot12 < 0).to(diff1.dtype)  # [B,C,1,1]

    # Adjust diff2 per-channel if conflict
    proj21 = dot12 / den11  # [B,C,1,1]
    diff2_h = diff2 - (mask * proj21) * diff1  # broadcast to [B,C,H,W]

    diff1_h = diff1
    if only_adjust == "both":
        proj12 = dot12 / den22
        diff1_h = diff1 - (mask * proj12) * diff2

    if return_mask:
        return diff1_h, diff2_h, mask
    return diff1_h, diff2_h


def gh_global_spatial(
    diff1: torch.Tensor,
    diff2: torch.Tensor,
    eps: float = 1e-6,
    only_adjust: str = "diff2",
    return_mask: bool = False,
) -> (
    Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Global spatial GH: treat each sample's [C,H,W] tensor as a single vector.
    URL: https://arxiv.org/pdf/2408.00288

    Args:
        diff1, diff2: [B, C, H, W] tensors.
        eps: numerical stability for denominator.
        only_adjust: "diff2" (default) or "both".
        return_mask: if True, return conflict mask [B,1,1,1] (1 where dot<0 per sample).

    Returns:
        diff1_h, diff2_h (, mask)
    """
    assert diff1.shape == diff2.shape, f"shape mismatch: {diff1.shape} vs {diff2.shape}"
    assert diff1.dim() == 4, "expected [B,C,H,W] tensors"

    # Global dot/den per sample over (C,H,W)
    dot12 = (diff1 * diff2).sum(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
    den11 = (diff1 * diff1).sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)  # [B,1,1,1]
    den22 = (diff2 * diff2).sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)  # [B,1,1,1]

    mask = (dot12 < 0).to(diff1.dtype)  # [B,1,1,1]

    if only_adjust not in ("diff2", "both"):
        raise ValueError("only_adjust must be 'diff2' or 'both'")

    # Adjust diff2: diff2 <- diff2 - proj(diff2 on diff1) (only when conflict)
    proj21 = dot12 / den11  # [B,1,1,1]
    diff2_h = diff2 - (mask * proj21) * diff1  # broadcast to [B,C,H,W]

    diff1_h = diff1
    if only_adjust == "both":
        proj12 = dot12 / den22
        diff1_h = diff1 - (mask * proj12) * diff2

    if return_mask:
        return diff1_h, diff2_h, mask
    return diff1_h, diff2_h


def gh_pixelwise_channel(
    diff1: torch.Tensor,
    diff2: torch.Tensor,
    eps: float = 1e-6,
    only_adjust: str = "diff2",
    return_mask: bool = False,
) -> (
    Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Pixel-wise channel-vector GH (projection removal on conflicts).
    URL: https://arxiv.org/pdf/2408.00288

    Args:
        diff1, diff2: [B, C, H, W] tensors.
        eps: numerical stability for denominator.
        only_adjust: "diff2" (default) or "both".
            - "diff2": adjust diff2 to remove its component along diff1 when conflict.
            - "both": also adjust diff1 symmetrically to remove its component along diff2.
        return_mask: if True, also return conflict mask [B,1,H,W] (1 where dot<0).

    Returns:
        diff1_h, diff2_h (, mask)
    """
    assert diff1.shape == diff2.shape, f"shape mismatch: {diff1.shape} vs {diff2.shape}"
    assert diff1.dim() == 4, "expected [B,C,H,W] tensors"

    # dot/den are computed per-pixel over channel dimension
    dot12 = (diff1 * diff2).sum(dim=1, keepdim=True)  # [B,1,H,W]
    den11 = (diff1 * diff1).sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1,H,W]
    den22 = (diff2 * diff2).sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1,H,W]

    mask = (dot12 < 0).to(diff1.dtype)  # [B,1,H,W]

    if only_adjust not in ("diff2", "both"):
        raise ValueError("only_adjust must be 'diff2' or 'both'")

    # Adjust diff2: diff2 <- diff2 - proj(diff2 on diff1)  (only where conflict)
    proj21 = dot12 / den11  # [B,1,H,W]
    diff2_h = diff2 - (mask * proj21) * diff1  # broadcast to [B,C,H,W]

    diff1_h = diff1
    if only_adjust == "both":
        # Symmetric adjustment of diff1: diff1 <- diff1 - proj(diff1 on diff2)
        # Use original dot12, but projection coeff for diff1 on diff2 is dot12/||diff2||^2
        proj12 = dot12 / den22
        diff1_h = diff1 - (mask * proj12) * diff2

    if return_mask:
        return diff1_h, diff2_h, mask
    print(mask.sum())
    return diff1_h, diff2_h


"""
# proj = (diff2*diff1).sum((1,2,3), keepdim=True) / (diff1*diff1).sum((1,2,3), keepdim=True).clamp(min=1e-6)
# diff2 = diff2 - proj*diff1
dot = (diff1*diff2).sum((1,2,3), keepdim=True)
# if (dot < 0).any():
proj = dot / (diff1*diff1).sum((1,2,3), keepdim=True).clamp(min=1e-6)
diff2 = diff2 - proj*diff1

dot = (diff1 * diff2).sum((1,2,3), keepdim=True)                  # [B,1,1,1]
den = (diff1 * diff1).sum((1,2,3), keepdim=True).clamp_min(1e-6)  # [B,1,1,1]
proj = dot / den
mask = (dot < 0).to(diff1.dtype)                                  # [B,1,1,1]
diff2 = diff2 - mask * proj * diff1


noise_c  = p1*diff1 + p2*diff2


target_norm = 0.5 * (l2_norm(diff1) + l2_norm(diff2))
c_norm = l2_norm(noise_c)

# noise_c = noise_c / c_norm * target_norm
# ratio = (target_norm / c_norm).clamp(max=1.3)
ratio = (target_norm / c_norm)
noise_c = noise_c * ratio
"""
