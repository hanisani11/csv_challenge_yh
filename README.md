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

place echocare_encoder.pth 
-->
pretrain/echocare_encoder.pth

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

