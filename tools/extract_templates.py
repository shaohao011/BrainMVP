# File: BrainMVP/tools/extract_templates_full.py
import torch
import numpy as np
import nibabel as nib
import os
from monai.transforms import Resize

# --- CẤU HÌNH ---
CKPT_PATH = '../pretrained_weights/BrainMVP_uniformer.pt'
OUTPUT_DIR = '../templates'
TARGET_SHAPE = (96, 96, 96) 

# MAPPING ĐẦY ĐỦ 8 LOẠI (Dựa trên code gốc BrainMVP)
TEMPLATE_MAP = {
    0: 'FLAIR',
    1: 'T1',
    2: 'T1c',  # T1 contrast
    3: 'T2',
    4: 'MRA',  # Mạch máu
    5: 'PD',   # Proton Density
    6: 'DWI',
    7: 'ADC'
}

def extract():
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Không tìm thấy checkpoint tại {CKPT_PATH}!")
        return

    print(f"Loading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    
    # Lấy tensor template
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    if 'module.rep_template' in state_dict:
        templates = state_dict['module.rep_template']
    elif 'rep_template' in ckpt:
        templates = ckpt['rep_template']
    else:
        print("❌ Không tìm thấy key 'rep_template'!")
        return

    print(f"Templates shape gốc: {templates.shape}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Resize về 96x96x96 sẵn để đỡ tốn CPU khi train
    resizer = Resize(spatial_size=TARGET_SHAPE, mode='trilinear')

    print("Đang trích xuất và lưu file...")
    for idx, name in TEMPLATE_MAP.items():
        # Lấy template (1, D, H, W)
        temp_tensor = templates[idx].unsqueeze(0) 
        
        # Resize
        resized_tensor = resizer(temp_tensor)
        data = resized_tensor.squeeze().numpy()
        
        # Save NIfTI
        img = nib.Nifti1Image(data, affine=np.eye(4))
        save_path = os.path.join(OUTPUT_DIR, f"{name}.nii.gz")
        nib.save(img, save_path)
        print(f"   -> [{idx}] Saved: {name}.nii.gz")

if __name__ == "__main__":
    extract()