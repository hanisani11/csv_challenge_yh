# CSV 2026 Challenge – Semi-supervised Ultrasound Segmentation

This repository contains a semi-supervised training pipeline for the **CSV 2026 Challenge**,  
focusing on **ultrasound image segmentation**.  
The current setup supports both **classification and segmentation training** and **segmentation-only training (without classification head)** using a separated decoder architecture.

---

## Environment Setup

We recommend using **conda** to manage the environment.

Use
    pip install -r requirements.txt


---

## Extra Setup

    root/
    ├─ data/
    |  └─ train/
    |     ├─ images/        # .h5 image files (long_img & trans_img)
    |     └─ labels/        # _label.h5 files (long_mask, trans_mask, cls)
    └─ pretrain/ 
        └─ echocare_encoder.pth    # Pre-trained Echocare encoder weights


---

## Create Local Train / Validation Split

Generate a balanced validation set and JSON splits:


    python split_train_valid_fold.py --root ./data --seed 2026 --val_size 50


This creates:

    train_labeled.json
    train_unlabeled.json
    valid.json

under the data/ directory.

---

## Training (Segmentation Only, No Classification)

To train the model without classification head, use train_no_cls.py.


    python train_no_cls.py \
        --train-labeled-json ./data/train_labeled.json \
        --train-unlabeled-json ./data/train_unlabeled.json \
        --valid-labeled-json ./data/valid.json \
        --model Echocare_sep_dec \
        --echo_care_ckpt ./pretrain/echocare_encoder.pth \
        --save_path ./checkpoints_sep_nocls \
        --gpu 0 \
        --train_epochs 100 \
        --batch_size 2

---

## Training (Segmentation & classification with sep dec *plaque weighted, cldice_ves loss added*)

To train the model descreibed above, use train_new_loss.py.


    python train_new_loss.py \
        --train-labeled-json ./data/train_labeled.json \
        --train-unlabeled-json ./data/train_unlabeled.json \
        --valid-labeled-json ./data/valid.json \
        --model Echocare_sep_dec \
        --echo_care_ckpt ./pretrain/echocare_encoder.pth \
        --save_path ./checkpoints_s_plqloss_cld \
        --gpu 0 \
        --train_epochs 100 \
        --batch_size 2 \