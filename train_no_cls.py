import argparse
import logging
import os
import sys
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.csv import CSVSemiDataset 
from util.utils import AverageMeter, count_params, DiceLoss, compute_nsd

# 모델 임포트 (기존 경로 유지)
from model.Echocare import Echocare_UniMatch
from model.Echocare_sep_dec import Echocare_sep_dec_UniMatch
from model.unet import UNetTwoView
from model.unet2 import UNetPlusPlusTwoView 


def main():
    parser = argparse.ArgumentParser("UniMatch Two-View Training - Segmentation Only Mode")
    parser.add_argument("--train-labeled-json", type=str, default="./data/train_labeled.json")
    parser.add_argument("--train-unlabeled-json", type=str, default="./data/train_unlabeled.json")
    parser.add_argument("--valid-labeled-json", type=str, default="./data/valid.json")

    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=0.0001)
    parser.add_argument("--conf_thresh", type=float, default=0.9)
    parser.add_argument("--seg_num_classes", type=int, default=3)
    parser.add_argument("--cls_num_classes", type=int, default=1)
    parser.add_argument("--resize_target", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20) # Early stopping 기준

    parser.add_argument("--echo_care_ckpt", type=str, default="./pretrain/echocare_encoder.pth")
    parser.add_argument('--amp', type=bool, default=True, help='enable torch.cuda.amp')
    parser.add_argument('--amp-dtype', type=str, default='fp16', choices=['fp16', 'bf16'])

    parser.add_argument("--model", type=str, default="Echocare_sep_dec", choices=["Echocare", "Echocare_sep_dec","UNet", "UNet++"])
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--gpu", type=str, default="3")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = build_logger(args.save_path)
    logger.info("Starting Segmentation-Only Training mode.")

    cudnn.enabled = True
    cudnn.benchmark = True

    tb_logdir = os.path.join(args.save_path, "tensorboard")
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    model = get_model(args)
    
    # [수정] Classification Head 파라미터 고정 (Grad 차단)
    if hasattr(model, 'cls_decoder'):
        for param in model.cls_decoder.parameters():
            param.requires_grad = False
        logger.info("Classification head frozen. Training segmentation only.")

    logger.info("Total params: {:.1f}M".format(count_params(model)))
    model = model.to(device)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.base_lr)

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=use_amp and (amp_dtype == torch.float16))
    
    db_train_u = CSVSemiDataset(args.train_unlabeled_json, "train_u", size=args.resize_target)
    db_train_l = CSVSemiDataset(args.train_labeled_json, "train_l", size=args.resize_target, n_sample=len(db_train_u.case_list))
    db_valid_l = CSVSemiDataset(args.valid_labeled_json, "valid")

    train_loader_l = DataLoader(db_train_l, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_loader_u = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_loader_u_mix = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(db_valid_l, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    total_iters = len(train_loader_u) * args.train_epochs

    previous_best = 0.0
    start_epoch = 0
    patience_counter = 0
    latest_ckpt = os.path.join(args.save_path, "latest.pth")
    
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        previous_best = ckpt.get("previous_best", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        logger.info(f"Resume from epoch {start_epoch}, best_seg={previous_best:.4f}")

    for epoch in range(start_epoch, args.train_epochs):
        logger.info(f"===========> Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}, Previous best seg: {previous_best:.4f}")
        
        stats = train_one_epoch(
            args=args, model=model, optimizer=optimizer, loader_l=train_loader_l,
            loader_u=train_loader_u, loader_u_mix=train_loader_u_mix, device=device,
            total_iters=total_iters, epoch=epoch, logger=logger,
            use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler
        )

        writer.add_scalar("Train/Total_Loss", stats["loss"], epoch)
        writer.add_scalar("Train/Loss_Seg_X", stats["loss_x"], epoch)
        writer.add_scalar("Train/Loss_Seg_S", stats["loss_s"], epoch)

        output_dict = validate(args, model, valid_loader, device, logger, writer=writer, epoch=epoch)

        total_score = output_dict["total_score"] # 오직 Segmentation 점수

        is_best = total_score > previous_best
        if is_best:
            previous_best = total_score
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_path, "best.pth"))
            logger.info(f"New Best Segmentation Score: {total_score:.4f}!")
        else:
            patience_counter += 1

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "previous_best": previous_best,
            "patience_counter": patience_counter
        }
        torch.save(ckpt, latest_ckpt)

        if patience_counter >= args.patience:
            logger.info("Early stopping triggered.")
            break
    
    writer.close()
    logger.info("Training finished.")

# ---------------------------------------------------------
# Helper Functions (Pseudo-label, CutMix 등 기존과 동일)
# ---------------------------------------------------------

def pseudo_from_logits(seg_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    prob = torch.softmax(seg_logits, dim=1)
    conf, mask = prob.max(dim=1)
    return conf, mask

def cutmix_apply_image(img_s: torch.Tensor, img_mix: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    box_ = box.unsqueeze(1).expand_as(img_s)
    out = img_s.clone()
    out[box_ == 1] = img_mix[box_ == 1]
    return out

def cutmix_apply_pseudo(mask: torch.Tensor, conf: torch.Tensor, mask_mix: torch.Tensor, conf_mix: torch.Tensor, box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_cm = mask.clone()
    conf_cm = conf.clone()
    mask_cm[box == 1] = mask_mix[box == 1]
    conf_cm[box == 1] = conf_mix[box == 1]
    return mask_cm, conf_cm

def ensure_cls_shape(y_cls: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(y_cls): y_cls = torch.as_tensor(y_cls)
    if y_cls.ndim == 1: y_cls = y_cls.unsqueeze(1)
    return y_cls

# ---------------------------------------------------------
# Modified Train One Epoch (Segmentation Only)
# ---------------------------------------------------------

def train_one_epoch(args, model, optimizer, loader_l, loader_u, loader_u_mix, device, total_iters, epoch, logger, use_amp, amp_dtype, scaler):
    model.train()
    criterion_seg_ce = nn.CrossEntropyLoss()
    criterion_seg_dice = DiceLoss(n_classes=args.seg_num_classes)

    total_loss = AverageMeter()
    total_loss_x = AverageMeter()
    total_loss_s = AverageMeter()
    total_mask_ratio = AverageMeter()

    loader = zip(loader_l, loader_u, loader_u_mix)

    for i, (
        (x_long, x_trans, m_long, m_trans, _), # y_cls 무시
        (uL_w, uL_s1, uL_s2, boxL1, boxL2, uT_w, uT_s1, uT_s2, boxT1, boxT2),
        (uL_wm, uL_s1m, uL_s2m, _, _, uT_wm, uT_s1m, uT_s2m, _, _),
    ) in enumerate(loader):

        x_long, x_trans = x_long.to(device), x_trans.to(device)
        m_long, m_trans = m_long.to(device), m_trans.to(device)
        uL_w, uL_s1, uL_s2 = uL_w.to(device), uL_s1.to(device), uL_s2.to(device)
        uT_w, uT_s1, uT_s2 = uT_w.to(device), uT_s1.to(device), uT_s2.to(device)
        boxL1, boxL2, boxT1, boxT2 = boxL1.to(device), boxL2.to(device), boxT1.to(device), boxT2.to(device)
        uL_wm, uL_s1m, uT_wm, uT_s1m = uL_wm.to(device), uL_s1m.to(device), uT_wm.to(device), uT_s1m.to(device)

        with torch.no_grad():
            model.eval()
            segL_wm, segT_wm, _ = model(uL_wm, uT_wm)
            confL_wm, maskL_wm = pseudo_from_logits(segL_wm.detach())
            confT_wm, maskT_wm = pseudo_from_logits(segT_wm.detach())

        uL_s1 = cutmix_apply_image(uL_s1, uL_s1m, boxL1)
        uT_s1 = cutmix_apply_image(uT_s1, uT_s1m, boxT1)

        model.train()
        num_l_bs, num_u_bs = x_long.size(0), uL_w.size(0)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            # need_fp=True 유지하여 Consistency 학습 강화
            (segL_all, segL_fp_all), (segT_all, segT_fp_all), _ = model(
                torch.cat([x_long, uL_w], 0), torch.cat([x_trans, uT_w], 0), need_fp=True
            )

            segL_x, segL_u_w = segL_all.split([num_l_bs, num_u_bs], 0)
            segT_x, segT_u_w = segT_all.split([num_l_bs, num_u_bs], 0)
            segL_u_w_fp, segT_u_w_fp = segL_fp_all[num_l_bs:], segT_fp_all[num_l_bs:]

            # Strong forward
            segL_s1, segT_s1, _ = model(uL_s1, uT_s1)

            # Labeled Seg Loss
            loss_x = (criterion_seg_ce(segL_x, m_long) + criterion_seg_dice(segL_x, m_long, softmax=True) +
                      criterion_seg_ce(segT_x, m_trans) + criterion_seg_dice(segT_x, m_trans, softmax=True)) / 4.0

            # Unlabeled Seg Loss (Strong)
            confL_w, maskL_w = pseudo_from_logits(segL_u_w.detach())
            confT_w, maskT_w = pseudo_from_logits(segT_u_w.detach())
            
            maskL_cm1, confL_cm1 = cutmix_apply_pseudo(maskL_w, confL_w, maskL_wm, confL_wm, boxL1)
            maskT_cm1, confT_cm1 = cutmix_apply_pseudo(maskT_w, confT_w, maskT_wm, confT_wm, boxT1)

            loss_u_s = (criterion_seg_dice(segL_s1, maskL_cm1, softmax=True, ignore=(confL_cm1 < args.conf_thresh).float()) +
                        criterion_seg_dice(segT_s1, maskT_cm1, softmax=True, ignore=(confT_cm1 < args.conf_thresh).float())) / 2.0

            # Feature Perturbation Loss
            loss_fp = (criterion_seg_dice(segL_u_w_fp, maskL_w, softmax=True, ignore=(confL_w < args.conf_thresh).float()) +
                       criterion_seg_dice(segT_u_w_fp, maskT_w, softmax=True, ignore=(confT_w < args.conf_thresh).float())) / 2.0

            # [수정] 최종 Loss에서 Classification 제외
            loss = loss_x + (loss_u_s * 0.5) + (loss_fp * 0.5)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Poly LR
        iters = epoch * len(loader_u) + i
        optimizer.param_groups[0]["lr"] = args.base_lr * (1 - iters / total_iters) ** 0.9

        total_loss.update(loss.item())
        total_loss_x.update(loss_x.item())
        total_loss_s.update(loss_u_s.item())
        mask_ratio = (confL_w >= args.conf_thresh).float().mean()
        total_mask_ratio.update(mask_ratio.item())

    return {"loss": total_loss.avg, "loss_x": total_loss_x.avg, "loss_s": total_loss_s.avg}

# ---------------------------------------------------------
# Modified Validation (Segmentation Focus)
# ---------------------------------------------------------

@torch.no_grad()
def validate(args, model, valid_loader, device, logger, writer=None, epoch=None):
    model.eval()
    dice_long, dice_trans = {1:0.0, 2:0.0}, {1:0.0, 2:0.0}
    nsd_long, nsd_trans = {1:0.0, 2:0.0}, {1:0.0, 2:0.0}
    
    num_batches = len(valid_loader)
    for i, (x_long, x_trans, m_long, m_trans, _) in enumerate(valid_loader):
        x_long_r = F.interpolate(x_long.to(device), (args.resize_target, args.resize_target), mode="bilinear")
        x_trans_r = F.interpolate(x_trans.to(device), (args.resize_target, args.resize_target), mode="bilinear")

        segL, segT, _ = model(x_long_r, x_trans_r)
        
        segL = F.interpolate(segL, x_long.shape[-2:], mode="bilinear")
        segT = F.interpolate(segT, x_trans.shape[-2:], mode="bilinear")
        
        predL, predT = torch.argmax(segL, 1), torch.argmax(segT, 1)
        gtL, gtT = m_long.to(device), m_trans.to(device)

        for cls in [1, 2]:
            # Dice
            def get_dice(p, g, c):
                inter = ((p==c)&(g==c)).sum().item()
                return 2.0*inter / ((p==c).sum()+(g==c).sum()+1e-8)
            
            dice_long[cls] += get_dice(predL, gtL, cls)
            dice_trans[cls] += get_dice(predT, gtT, cls)
            
            # NSD
            nsd_long[cls] += compute_nsd((predL[0]==cls).cpu().numpy(), (gtL[0]==cls).cpu().numpy(), 3.0)
            nsd_trans[cls] += compute_nsd((predT[0]==cls).cpu().numpy(), (gtT[0]==cls).cpu().numpy(), 3.0)

    # Average metrics
    for c in [1, 2]:
        dice_long[c] /= num_batches; dice_trans[c] /= num_batches
        nsd_long[c] /= num_batches; nsd_trans[c] /= num_batches

    # [수정] Total Score는 오직 Seg 점수 기반 (Plaque 0.6, Vessel 0.4 가중치)
    score_L = (dice_long[1]+nsd_long[1])/2 * 0.6 + (dice_long[2]+nsd_long[2])/2 * 0.4
    score_T = (dice_trans[1]+nsd_trans[1])/2 * 0.6 + (dice_trans[2]+nsd_trans[2])/2 * 0.4
    total_seg_score = (score_L + score_T) / 2

    logger.info(f"Val Mean Dice: {(sum(dice_long.values())+sum(dice_trans.values()))/4:.4f} | Total Seg Score: {total_seg_score:.4f}")

    return {
        "total_score": total_seg_score,
        "dice_long_vessel": dice_long[2], "dice_long_plaque": dice_long[1],
        "dice_trans_vessel": dice_trans[2], "dice_trans_plaque": dice_trans[1]
    }

def build_logger(save_path: str):
    logger = logging.getLogger("UniMatch_SegOnly")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(save_path, exist_ok=True)
        fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
        fh = logging.FileHandler(os.path.join(save_path, "log.txt")); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

def get_model(args):
    """
    Return model instance based on args.model.
      - 'Echocare' -> Echocare_UniMatch(...) (uses encoder checkpoint)
      - 'UNet'     -> UNetTwoView(...)
    """
    if args.model == "Echocare":
        model = Echocare_UniMatch(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            encoder_pth=args.echo_care_ckpt,
        )
    elif args.model == "Echocare_sep_dec":
        model = Echocare_sep_dec_UniMatch(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            encoder_pth=args.echo_care_ckpt,
        )

    elif args.model == "UNet":
        model = UNetTwoView(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
        )
    elif args.model == "UNet++":
        model = UNetPlusPlusTwoView(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
        )
    else:
        raise ValueError(f"Unknown model choice: {args.model}")
    return model

if __name__ == "__main__":
    main()