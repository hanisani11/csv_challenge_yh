from copy import deepcopy
import h5py
import math
import json
import numpy as np
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box
from scipy.ndimage import distance_transform_edt
from scipy import ndimage




class CSVSemiDataset(Dataset):
    def __init__(self, json_file_path, mode, size=None, n_sample=None):
        self.json_file_path = json_file_path
        self.mode = mode
        self.size = size
        self.n_sample = n_sample

        if mode == 'train_l' or mode == 'train_u':
            with open(self.json_file_path, mode='r') as f:
                self.case_list = json.load(f)
            if mode == 'train_l' and n_sample is not None:
                self.case_list *= math.ceil(n_sample / len(self.case_list))
                self.case_list = self.case_list[:n_sample]
        else:
            with open(self.json_file_path, mode='r') as f:
                self.case_list = json.load(f)
    
    def _read_pair(self, image_h5_file):
        with h5py.File(image_h5_file, "r") as f:
            long_img = f["long_img"][:]
            trans_img = f["trans_img"][:]

        # ensure float32 and normalize to [0,1] (many h5 stores 0-255)
        long_img = long_img.astype(np.float32)
        trans_img = trans_img.astype(np.float32)

        # only scale if values appear to be in 0-255 range
        try:
            if long_img.max() > 1.0:
                long_img = long_img / 255.0
        except ValueError:
            # empty arrays or unexpected shapes: skip scaling
            pass

        try:
            if trans_img.max() > 1.0:
                trans_img = trans_img / 255.0
        except ValueError:
            pass

        return long_img, trans_img

    def _read_label(self, label_h5_file):
        with h5py.File(label_h5_file, 'r') as f:
            long_mask = f['long_mask'][:]
            trans_mask = f['trans_mask'][:]
            cls = f['cls'][()]   # shape: [] or [1] or [C]
        # map possible mask values {0,128,255} -> {0,1,2}
        long_mask = long_mask.astype(np.int64)
        trans_mask = trans_mask.astype(np.int64)
        # map 128 -> 1, 255 -> 2, leave 0 as 0 (robust if masks already 0/1/2)
        long_mask = np.where(long_mask == 128, 1, long_mask)
        long_mask = np.where(long_mask == 255, 2, long_mask)
        trans_mask = np.where(trans_mask == 128, 1, trans_mask)
        trans_mask = np.where(trans_mask == 255, 2, trans_mask)
        return long_mask, trans_mask, cls
    

    def __getitem__(self, item):
        case = self.case_list[item]

        if self.mode == 'valid':
            image_h5_file, label_h5_file = case['image'], case['label']
            long_img, trans_img = self._read_pair(image_h5_file)
            long_mask, trans_mask, cls = self._read_label(label_h5_file)

            dist_long = self._plaque_dist_map(long_mask, plaque_idx=1)
            dist_trans = self._plaque_dist_map(trans_mask, plaque_idx=1)

            return (
                torch.from_numpy(long_img).unsqueeze(0).float(),
                torch.from_numpy(trans_img).unsqueeze(0).float(),
                torch.from_numpy(long_mask).long(),
                torch.from_numpy(trans_mask).long(),
                torch.tensor(cls).long(),
                torch.from_numpy(dist_long).float(),
                torch.from_numpy(dist_trans).float(),
            )

        elif self.mode == 'train_l':
            image_h5_file, label_h5_file = case['image'], case['label']
            long_img, trans_img = self._read_pair(image_h5_file)
            long_mask, trans_mask, cls = self._read_label(label_h5_file)

            # --- NEW: plaque distance maps (GT 기반, 한번만 계산) ---
            dist_long = self._plaque_dist_map(long_mask, plaque_idx=1)
            dist_trans = self._plaque_dist_map(trans_mask, plaque_idx=1)

            # Apply same-type augmentation to img/mask/dist
            long_img, long_mask, dist_long = self._apply_aug_pair(long_img, long_mask, dist_long)
            trans_img, trans_mask, dist_trans = self._apply_aug_pair(trans_img, trans_mask, dist_trans)

            # Resize to target size
            x, y = long_img.shape
            long_img = zoom(long_img, (self.size / x, self.size / y), order=0)
            long_mask = zoom(long_mask, (self.size / x, self.size / y), order=0)
            dist_long = zoom(dist_long, (self.size / x, self.size / y), order=1)  # NEW

            x2, y2 = trans_img.shape
            trans_img = zoom(trans_img, (self.size / x2, self.size / y2), order=0)
            trans_mask = zoom(trans_mask, (self.size / x2, self.size / y2), order=0)
            dist_trans = zoom(dist_trans, (self.size / x2, self.size / y2), order=1)  # NEW

            return (
                torch.from_numpy(long_img).unsqueeze(0).float(),
                torch.from_numpy(trans_img).unsqueeze(0).float(),
                torch.from_numpy(long_mask).long(),
                torch.from_numpy(trans_mask).long(),
                torch.tensor(cls).long(),
                torch.from_numpy(dist_long).float(),   # NEW: [H,W]
                torch.from_numpy(dist_trans).float(),  # NEW: [H,W]
            )

        elif self.mode == 'train_u':
            image_h5_file = case['image']
            long_img, trans_img = self._read_pair(image_h5_file)

            def _make_u(img):
                # Matches previous logic; helper for producing weak/strong variants
                if random.random() > 0.5:
                    img = random_rot_flip(img)
                elif random.random() > 0.5:
                    img = random_rotate(img)

                x, y = img.shape
                img = zoom(img, (self.size / x, self.size / y), order=0)

                img = Image.fromarray((img * 255).astype(np.uint8))
                img_s1, img_s2 = deepcopy(img), deepcopy(img)
                img_w = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

                if random.random() < 0.8:
                    img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
                img_s1 = blur(img_s1, p=0.5)
                box1 = obtain_cutmix_box(self.size, p=0.5)
                img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

                if random.random() < 0.8:
                    img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
                img_s2 = blur(img_s2, p=0.5)
                box2 = obtain_cutmix_box(self.size, p=0.5)
                img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

                return img_w, img_s1, img_s2, box1, box2

            long_w, long_s1, long_s2, box_l1, box_l2 = _make_u(long_img)
            trans_w, trans_s1, trans_s2, box_t1, box_t2 = _make_u(trans_img)

            return long_w, long_s1, long_s2, box_l1, box_l2, trans_w, trans_s1, trans_s2, box_t1, box_t2

    def __len__(self):
        return len(self.case_list)
    
    def _plaque_dist_map(self, mask: np.ndarray, plaque_idx: int = 1) -> np.ndarray:
        """
        mask: [H,W] int64 (0=bg, 1=plaque, 2=vessel)
        return dist: [H,W] float32, distance to nearest plaque pixel (0 inside plaque)
        """
        plaque = (mask == plaque_idx)
        # distance_transform_edt는 "True" 위치까지의 거리 계산을 위해 보통 반전 사용
        # plaque=True인 곳 dist=0이 되게 하려면 ~plaque에 edt 적용
        dist = distance_transform_edt(~plaque).astype(np.float32)
        return dist

    def _apply_aug_pair(self, img, mask=None, dist=None):
        """
        random_rot_flip / random_rotate 를 img, mask, dist에 일관되게 적용
        dist는 연속값이므로 rotate는 order=1이 더 적절.
        """
        if random.random() > 0.5:
            # rot90 + flip
            k = np.random.randint(0, 4)
            axis = np.random.randint(0, 2)
            img = np.rot90(img, k)
            img = np.flip(img, axis=axis).copy()
            if mask is not None:
                mask = np.rot90(mask, k)
                mask = np.flip(mask, axis=axis).copy()
            if dist is not None:
                dist = np.rot90(dist, k)
                dist = np.flip(dist, axis=axis).copy()
            return img, mask, dist

        elif random.random() > 0.5:
            # rotate
            angle = np.random.randint(-20, 20)
            img = ndimage.rotate(img, angle, order=0, reshape=False)
            if mask is not None:
                mask = ndimage.rotate(mask, angle, order=0, reshape=False)
            if dist is not None:
                dist = ndimage.rotate(dist, angle, order=1, reshape=False)  # dist는 연속값
            return img, mask, dist

        return img, mask, dist




