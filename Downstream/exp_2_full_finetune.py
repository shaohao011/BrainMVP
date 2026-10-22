import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from core.dataset import BrainMVPDataset
from core.modeling import BrainMVPClassifier
from core.engine import run_training_loop

def run_exp_2():
    # --- CẤU HÌNH EXP 2 ---
    JSON_FILE = '../jsons/dataset_DienBien_MRI_Full.json'
    PRETRAINED = '../pretrained_weights/BrainMVP_uniformer.pt'
    SAVE_DIR = 'results/exp_2_full_finetune'
    
    # Fine-tune cần số epoch nhiều hơn Linear Probe một chút để hội tụ sâu
    EPOCHS = 40 
    BATCH_SIZE = 4 # Nếu GPU yếu thì giảm xuống 2, vì unfreeze tốn VRAM hơn
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- EXP 2: FULL FINE-TUNING on {device} ---")

    # 1. Data
    train_ds = BrainMVPDataset(JSON_FILE, "training")
    val_ds = BrainMVPDataset(JSON_FILE, "validation")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Model (FREEZE = FALSE -> Cho phép học lại toàn bộ)
    model = BrainMVPClassifier(num_classes=3, in_channels=4, pretrained_path=PRETRAINED, freeze_backbone=False)
    model.to(device)

    # 3. Loss (Weighted - Chống mất cân bằng dữ liệu)
    weights = torch.tensor([1.0, 15.0, 15.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 4. Optimizer với Layer-wise Learning Rate (KỸ THUẬT QUAN TRỌNG)
    # Tách tham số thành 2 nhóm: Backbone và Head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5}, # Backbone học chậm (giữ kiến thức cũ)
        {'params': head_params, 'lr': 1e-3}      # Head học nhanh (bắt kịp task mới)
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. RUN LOOP (Tự động Log, Save, Resume)
    run_training_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, EPOCHS, SAVE_DIR, resume=True)

if __name__ == "__main__":
    run_exp_2()