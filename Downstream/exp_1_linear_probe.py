import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from core.dataset import BrainMVPDataset
from core.modeling import BrainMVPClassifier
from core.engine import run_training_loop # Import hàm mới

def run_exp_1():
    # --- CẤU HÌNH ---
    JSON_FILE = '../jsons/dataset_DienBien_MRI_Full.json'
    PRETRAINED = '../pretrained_weights/BrainMVP_uniformer.pt'
    SAVE_DIR = 'results/exp_1_linear_probe_mlp'
    
    EPOCHS = 30
    BATCH_SIZE = 8
    LR = 1e-3
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- EXP 1: LINEAR PROBING on {device} ---")

    # 1. Data
    train_ds = BrainMVPDataset(JSON_FILE, "training")
    val_ds = BrainMVPDataset(JSON_FILE, "validation")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Model (FREEZE BACKBONE)
    model = BrainMVPClassifier(num_classes=3, in_channels=4, pretrained_path=PRETRAINED, freeze_backbone=True)
    model.to(device)

    # 3. Optimizer
    weights = torch.tensor([1.0, 15.0, 15.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. RUN LOOP (Có Resume, có Log CSV, có Save Best)
    run_training_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, EPOCHS, SAVE_DIR, resume=True)

if __name__ == "__main__":
    run_exp_1()