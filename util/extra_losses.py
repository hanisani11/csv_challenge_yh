# util/extra_losses.py
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


# -------------------------
# Distance-to-GT Plaque penalty (L_far-plaque)
# -------------------------
def _distance_to_mask(mask_bool_hw: np.ndarray) -> np.ndarray:
    """
    mask_bool_hw: True where GT plaque exists
    returns dist map: distance to nearest plaque pixel (0 inside plaque)
    """
    # edt computes distance to nearest False. So invert: plaque=True -> invert -> False on plaque
    dist = distance_transform_edt(~mask_bool_hw)
    return dist.astype(np.float32)

def build_plaque_distance_map(gt_mask_bhw: torch.Tensor, plaque_idx: int) -> torch.Tensor:
    """
    gt_mask_bhw: [B,H,W] int64
    returns dist: [B,H,W] float32 on CPU
    """
    gt_np = gt_mask_bhw.detach().cpu().numpy()
    B, H, W = gt_np.shape
    out = np.zeros((B, H, W), dtype=np.float32)
    for b in range(B):
        plaque_bool = (gt_np[b] == plaque_idx)
        out[b] = _distance_to_mask(plaque_bool)
    return torch.from_numpy(out)  # CPU tensor


def far_plaque_loss_from_logits(
    seg_logits_bchw: torch.Tensor,
    gt_mask_bhw: torch.Tensor = None,
    plaque_idx: int = 1,
    dmax: float = 30.0,
    sigma: float = 10.0,
    mode: str = "exp",
    dist_map_bhw: torch.Tensor = None,   # NEW
) -> torch.Tensor:
    """
    dist_map_bhw: [B,H,W] float32 (GT plaque까지 거리). 주어지면 SciPy EDT 계산 안 함.
    gt_mask_bhw는 dist_map이 없을 때만 필요.
    """
    prob = torch.softmax(seg_logits_bchw, dim=1)
    p_plq = prob[:, plaque_idx]  # [B,H,W]

    if dist_map_bhw is None:
        assert gt_mask_bhw is not None, "Need gt_mask_bhw when dist_map_bhw is None"
        dist = build_plaque_distance_map(gt_mask_bhw, plaque_idx=plaque_idx).to(seg_logits_bchw.device)
    else:
        dist = dist_map_bhw.to(seg_logits_bchw.device)

    if mode == "linear":
        w = torch.clamp(dist / (dmax + 1e-8), 0.0, 1.0)
    else:
        w = 1.0 - torch.exp(-dist / (sigma + 1e-8))

    return (p_plq * w).mean()



# -------------------------
# (B) Soft clDice for vessel (L_clDice_vessel)
#    differentiable soft-skeletonization
# -------------------------
def _soft_erode(img):
    # img: [B,1,H,W]
    p1 = -F.max_pool2d(-img, kernel_size=(3,1), stride=1, padding=(1,0))
    p2 = -F.max_pool2d(-img, kernel_size=(1,3), stride=1, padding=(0,1))
    return torch.min(p1, p2)

def _soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def _soft_open(img):
    return _soft_dilate(_soft_erode(img))

def soft_skel(img, iters=10):
    """
    img: [B,1,H,W] in [0,1]
    returns soft skeleton
    """
    img1 = img
    skel = torch.zeros_like(img)
    for _ in range(iters):
        opened = _soft_open(img1)
        delta = F.relu(img1 - opened)
        skel = skel + F.relu(delta - skel * delta)  # accumulate
        img1 = _soft_erode(img1)
    return skel

def cldice_loss_from_logits(
    seg_logits_bchw: torch.Tensor,
    gt_mask_bhw: torch.Tensor,
    vessel_idx: int = 2,
    skel_iters: int = 10,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    seg_logits_bchw: [B,C,H,W]
    gt_mask_bhw:     [B,H,W]
    clDice loss for vessel class only.
    """
    prob = torch.softmax(seg_logits_bchw, dim=1)
    p = prob[:, vessel_idx:vessel_idx+1]  # [B,1,H,W] soft vessel

    g = (gt_mask_bhw == vessel_idx).float().unsqueeze(1)  # [B,1,H,W] hard GT vessel

    # soft skeletons
    skel_p = soft_skel(p, iters=skel_iters)
    skel_g = soft_skel(g, iters=skel_iters)

    # topology precision/recall
    tprec = (torch.sum(skel_p * g) + eps) / (torch.sum(skel_p) + eps)
    tsens = (torch.sum(skel_g * p) + eps) / (torch.sum(skel_g) + eps)

    cl_dice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return 1.0 - cl_dice
