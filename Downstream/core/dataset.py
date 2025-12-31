import json
import torch
import numpy as np
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityd,
    ResizeWithPadOrCropd
)

class BrainMVPDataset(Dataset):
    def __init__(self, json_file, section, patch_shape=(96, 96, 96)):
        """
        Args:
            json_file: Đường dẫn file JSON.
            section: 'training', 'validation', hoặc 'test'.
            patch_shape: Kích thước input cho Uniformer (mặc định 96^3).
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Nếu json có key section (training/val...)
        self.sample_list = data[section]
        self.patch_shape = patch_shape
        # Định nghĩa 4 loại xung cần lấy
        self.modalities = ["flair", "dwi", "t1", "t2"]
        
        # Pipeline xử lý ảnh MONAI
        self.transforms = Compose([
            # Load 4 kênh (cho phép thiếu)
            LoadImaged(keys=self.modalities, allow_missing_keys=True, image_only=True),
            
            # Đảm bảo channel first (C, D, H, W)
            # Nếu file gốc 3D -> (1, D, H, W)
            # Nếu file gốc 4D (DWI nhiều b-value) -> (N, D, H, W)
            EnsureChannelFirstd(keys=self.modalities, allow_missing_keys=True),
            
            # Chuẩn hóa hướng không gian (RAS)
            Orientationd(keys=self.modalities, axcodes="RAS", allow_missing_keys=True),
            
            # Resize về kích thước cố định
            ResizeWithPadOrCropd(keys=self.modalities, spatial_size=patch_shape, mode="constant", value=0, allow_missing_keys=True),
            
            # Chuẩn hóa Intensity về [0, 1]
            ScaleIntensityd(keys=self.modalities, allow_missing_keys=True)
        ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample = self.sample_list[index]
        
        # 1. Map path từ JSON vào Dictionary
        input_dict = {}
        paths = sample['image']
        
        # JSON đã sort sẵn: [FLAIR, DWI, T1, T2]
        # Ta map lần lượt vào keys
        for i, mod_name in enumerate(self.modalities):
            if i < len(paths):
                input_dict[mod_name] = paths[i]
        
        # 2. Apply Transforms
        data = self.transforms(input_dict)
        
        # 3. Stack thành Tensor 4 kênh
        channels = []
        for mod in self.modalities:
            if mod in data:
                img_tensor = data[mod]
                
                # --- SỬA LỖI TẠI ĐÂY: Xử lý file nhiều channels (như DWI) ---
                if img_tensor.shape[0] > 1:
                    # Nếu có > 1 kênh (ví dụ DWI b0, b1000), lấy trung bình để ép về 1 kênh
                    # Shape: [N, D, H, W] -> [1, D, H, W]
                    img_tensor = torch.mean(img_tensor, dim=0, keepdim=True)
                
                channels.append(img_tensor)
            else:
                # Tạo ảnh đen (Zero) nếu thiếu modality
                # Shape: [1, 96, 96, 96]
                channels.append(torch.zeros((1, *self.patch_shape)))
        
        # Stack lại: [1, D, H, W] * 4 -> [4, D, H, W]
        final_img = torch.cat(channels, dim=0)
        
        # Nhãn
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)
        
        return {
            'image': final_img, 
            'label': label_tensor, 
            'id': sample.get('id', str(index))
        }