import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Modules
from core.dataset_advanced import BrainMVPAdvancedDataset
from core.modeling import BrainMVPClassifier
from core.utils import TrainingLogger, save_checkpoint
import numpy as np

from torch.utils.data import WeightedRandomSampler

def make_weighted_sampler(dataset):
    # 1. Lấy danh sách nhãn
    targets = [sample['label'] for sample in dataset.sample_list]
    class_counts = np.bincount(targets)
    
    # 2. Tính trọng số cho từng Class (Nghịch đảo số lượng)
    # Class ít -> Trọng số cao
    class_weights = 1.0 / class_counts
    
    # 3. Gán trọng số cho từng Mẫu (Sample)
    sample_weights = [class_weights[label] for label in targets]
    
    # 4. Tạo Sampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Tensor trọng số class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def run_exp_3():
    JSON_FILE = '../jsons/dataset_DienBien_MRI_Full.json'
    PRETRAINED = '../pretrained_weights/BrainMVP_uniformer.pt'
    TEMPLATE_DIR = '../templates'
    SAVE_DIR = 'results/exp_3_sota'
    
    EPOCHS = 50
    BATCH_SIZE = 2
    LR_BACKBONE = 1e-5
    LR_HEAD = 1e-3
    CONSISTENCY_WEIGHT = 1.0 
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- EXP 3: SOTA (Template Mix + Consistency) on {device} ---")

    train_ds = BrainMVPAdvancedDataset(JSON_FILE, "training", TEMPLATE_DIR, is_train=True)
    val_ds = BrainMVPAdvancedDataset(JSON_FILE, "validation", TEMPLATE_DIR, is_train=False)

    train_sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Model
    model = BrainMVPClassifier(num_classes=3, in_channels=4, pretrained_path=PRETRAINED, freeze_backbone=False)
    model.to(device)

    # 3. Loss & Optimizer
    cls_weights = torch.tensor([1.0, 10.0, 10.0]).to(device) 
    criterion_cls = FocalLoss(alpha=cls_weights, gamma=2.0)
    criterion_cons = nn.MSELoss()
    
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
        {'params': model.head.parameters(), 'lr': LR_HEAD}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    logger = TrainingLogger(SAVE_DIR)

    # 4. SOTA Training Loop
    best_f1 = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            # Load 2 views
            view1 = batch['image'].to(device)
            view2 = batch['image_aug'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward cả 2 views (Lấy cả logits và features)
            logits1, feats1 = model(view1, return_features=True)
            logits2, feats2 = model(view2, return_features=True)
            
            # Tính Loss
            # 1. Classification Loss (trên cả 2 view để học mạnh hơn)
            loss_cls = (criterion_cls(logits1, labels) + criterion_cls(logits2, labels)) / 2
            
            # 2. Consistency Loss (ép feature của 2 view gần nhau)
            loss_cons = criterion_cons(feats1, feats2)
            
            # Tổng Loss
            loss = loss_cls + CONSISTENCY_WEIGHT * loss_cons
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Metric (chỉ tính trên view 1)
            preds = torch.argmax(logits1, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", l_cons=f"{loss_cons.item():.4f}")
            
        scheduler.step()
        train_loss_avg = total_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation (Standard)
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                logits = model(inputs) # Không cần features lúc val
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        report = classification_report(val_targets, val_preds, target_names=["Negative", "Hemorrhage", "Infarction"], zero_division=0, output_dict=True)
        
        print(f"Ep {epoch} | Loss: {train_loss_avg:.4f} | Val Acc: {val_acc:.3f} F1: {val_f1:.3f}")
        print(f"   -> Hemorrhage F1: {report['Hemorrhage']['f1-score']:.2f} | Infarction F1: {report['Infarction']['f1-score']:.2f}")
        
        logger.log(epoch, train_loss_avg, train_acc, 0.0, val_acc, val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model.state_dict(), SAVE_DIR, is_best=True)

    logger.plot_history()

if __name__ == "__main__":
    run_exp_3()