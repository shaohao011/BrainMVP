import json
import os
import torch
import numpy as np
import nibabel as nib
# SỬA LỖI Ở ĐÂY: Dùng Dataset của Torch thay vì MONAI để tránh lỗi Attribute 'data'
from torch.utils.data import Dataset 
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    ScaleIntensityd, ResizeWithPadOrCropd
)

class BrainMVPAdvancedDataset(Dataset):
    def __init__(self, json_file, section, template_dir, patch_shape=(96, 96, 96), is_train=True):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.sample_list = data[section]
        self.template_dir = template_dir
        self.is_train = is_train
        self.patch_shape = patch_shape
        self.modalities = ["flair", "dwi", "t1", "t2"]
        
        # Load Templates vào RAM
        self.templates = {}
        print(f"Loading templates from: {template_dir}")
        for mod in self.modalities:
            # Lưu ý tên file template phải khớp (FLAIR.nii.gz)
            path = os.path.join(template_dir, f"{mod.upper()}.nii.gz")
            if os.path.exists(path):
                try:
                    img = nib.load(path).get_fdata()
                    # Chuyển về Tensor (1, 96, 96, 96)
                    self.templates[mod] = torch.tensor(img).float().unsqueeze(0)
                except Exception as e:
                    print(f"❌ Error loading template {mod}: {e}")
                    self.templates[mod] = torch.zeros((1, *patch_shape))
            else:
                print(f"⚠ Warning: Missing template for {mod} at {path}")
                self.templates[mod] = torch.zeros((1, *patch_shape))

        # Base Transforms
        self.transforms = Compose([
            LoadImaged(keys=self.modalities, allow_missing_keys=True, image_only=True),
            EnsureChannelFirstd(keys=self.modalities, allow_missing_keys=True),
            Orientationd(keys=self.modalities, axcodes="RAS", allow_missing_keys=True),
            ResizeWithPadOrCropd(keys=self.modalities, spatial_size=patch_shape, mode="constant", value=0, allow_missing_keys=True),
            ScaleIntensityd(keys=self.modalities, allow_missing_keys=True)
        ])

    def __len__(self):
        return len(self.sample_list)

    def _apply_template_mix(self, data_dict):
        """Random thay thế kênh thật bằng kênh Template"""
        mixed_channels = []
        for mod in self.modalities:
            if mod in data_dict:
                img = data_dict[mod]
                # Sửa lỗi shape: Nếu ảnh load vào là (C, D, H, W) mà C > 1 (như DWI) -> Lấy mean
                if img.shape[0] > 1:
                    img = torch.mean(img, dim=0, keepdim=True)
                
                # Logic Mix: 30% thay bằng Template
                if self.is_train and np.random.rand() < 0.3:
                    # Resize template về cùng kích thước ảnh (đề phòng lệch)
                    temp = self.templates[mod]
                    if temp.shape != img.shape:
                         # Nếu template lệch shape, dùng interpolate (chữa cháy) hoặc dùng ảnh gốc
                         img = img # Fallback
                    else:
                         img = temp.clone()
            else:
                # Không có ảnh thật -> Dùng Template bù vào
                img = self.templates[mod].clone()
            
            mixed_channels.append(img)
        
        # Stack lại: (4, 96, 96, 96)
        return torch.cat(mixed_channels, dim=0)

    def __getitem__(self, index):
        sample = self.sample_list[index]
        paths = sample['image']
        
        input_dict = {}
        for i, mod in enumerate(self.modalities):
            if i < len(paths): input_dict[mod] = paths[i]
        
        # Load & Preprocess ảnh gốc
        data = self.transforms(input_dict)
        
        # Tạo 2 Views
        if self.is_train:
            view1 = self._apply_template_mix(data)
            view2 = self._apply_template_mix(data)
            return {
                'image': view1, 
                'image_aug': view2, 
                'label': torch.tensor(sample['label'], dtype=torch.long)
            }
        else:
            # Validation: Không mix ngẫu nhiên, chỉ điền khuyết
            channels = []
            for mod in self.modalities:
                if mod in data: 
                    img = data[mod]
                    if img.shape[0] > 1: img = torch.mean(img, dim=0, keepdim=True)
                    channels.append(img)
                else: 
                    channels.append(self.templates[mod].clone())
            img = torch.cat(channels, dim=0)
            return {'image': img, 'label': torch.tensor(sample['label'], dtype=torch.long)}