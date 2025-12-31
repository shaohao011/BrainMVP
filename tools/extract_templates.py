# BrainMVP/tools/extract_templates.py
import torch
import numpy as np
import nibabel as nib
import os
from monai.transforms import Resize

# Cấu hình
CKPT_PATH = '../pretrained_weights/BrainMVP_uniformer.pt'
OUTPUT_DIR = '../templates'
TARGET_SHAPE = (96, 96, 96) # Resize luôn về kích thước train để đỡ tốn CPU lúc train

# Mapping thứ tự trong checkpoint (Dựa trên code gốc train_script.py)
# 0:FLAIR, 1:T1, 2:T1c, 3:T2, 4:MRA, 5:PD, 6:DWI, 7:ADC
TEMPLATE_MAP = {
    0: 'FLAIR',
    1: 'T1',
    3: 'T2',
    6: 'DWI'
}

def extract():
    if not os.path.exists(CKPT_PATH):
        print("❌ Không tìm thấy checkpoint!")
        return

    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    
    # Lấy tensor template
    # Key có thể là 'module.rep_template' hoặc 'rep_template'
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    if 'module.rep_template' in state_dict:
        templates = state_dict['module.rep_template']
    elif 'rep_template' in ckpt:
        templates = ckpt['rep_template']
    else:
        print("❌ Không tìm thấy key 'rep_template'!")
        return

    print(f"Templates shape original: {templates.shape}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Resize transform
    resizer = Resize(spatial_size=TARGET_SHAPE, mode='trilinear')

    for idx, name in TEMPLATE_MAP.items():
        # Lấy template (1, D, H, W)
        temp_tensor = templates[idx].unsqueeze(0) 
        
        # Resize về 96x96x96
        resized_tensor = resizer(temp_tensor)
        
        # Save
        data = resized_tensor.squeeze().numpy()
        img = nib.Nifti1Image(data, affine=np.eye(4))
        save_path = os.path.join(OUTPUT_DIR, f"{name}.nii.gz")
        nib.save(img, save_path)
        print(f"-> Saved: {save_path} (Shape: {data.shape})")

if __name__ == "__main__":
    extract()