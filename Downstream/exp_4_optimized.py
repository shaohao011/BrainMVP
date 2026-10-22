import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, accuracy_score
import numpy as np
from tqdm import tqdm
import copy

# Modules
from core.dataset_advanced import BrainMVPAdvancedDataset
from core.modeling import BrainMVPClassifier
from core.utils import TrainingLogger

# --- CẤU HÌNH ---
JSON_FILE = '../jsons/dataset_DienBien_MRI_Full.json'
PRETRAINED = '../pretrained_weights/BrainMVP_uniformer.pt'
TEMPLATE_DIR = '../templates'
SAVE_DIR = 'results/exp_4_optimized'

# K-Fold Config
N_FOLDS = 5
EPOCHS = 60       # Tăng lên vì có Early Stopping lo
PATIENCE = 15     # Đợi 15 epoch không tiến bộ thì dừng
BATCH_SIZE = 2    # Batch nhỏ cho an toàn

# Learning Rates
LR_BACKBONE = 2e-5
LR_HEAD = 2e-3
WEIGHT_DECAY = 1e-4

# --- HELPER CLASSES ---

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score, model):
        # Score ở đây là F1 (càng cao càng tốt)
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'   | EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f'   | ⭐ New Best Score! Model saved.')

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

def get_optimizer_params(model, lr_backbone, lr_head, weight_decay):
    """
    Tách tham số để áp dụng Weight Decay đúng cách:
    - Weight Decay: Áp dụng cho Linear, Conv weights.
    - NO Weight Decay: Áp dụng cho Bias, LayerNorm.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'norm', 'LayerNorm']
    
    optimizer_grouped_parameters = [
        # Backbone - Decay
        {'params': [p for n, p in param_optimizer if "backbone" in n and not any(nd in n for nd in no_decay)], 
         'lr': lr_backbone, 'weight_decay': weight_decay},
        # Backbone - No Decay
        {'params': [p for n, p in param_optimizer if "backbone" in n and any(nd in n for nd in no_decay)], 
         'lr': lr_backbone, 'weight_decay': 0.0},
        # Head - Decay
        {'params': [p for n, p in param_optimizer if "head" in n and not any(nd in n for nd in no_decay)], 
         'lr': lr_head, 'weight_decay': weight_decay},
        # Head - No Decay
        {'params': [p for n, p in param_optimizer if "head" in n and any(nd in n for nd in no_decay)], 
         'lr': lr_head, 'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters

# --- TRAINING ENGINE ---

def train_fold(fold_idx, train_idx, val_idx, dataset, device):
    print(f"\n{'='*20} FOLD {fold_idx+1}/{N_FOLDS} {'='*20}")
    fold_dir = os.path.join(SAVE_DIR, f'fold_{fold_idx+1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # 1. Prepare Data
    # --- Weighted Sampler Logic ---
    subset_labels = [dataset.sample_list[i]['label'] for i in train_idx]
    class_counts = np.bincount(subset_labels)
    class_counts = np.where(class_counts == 0, 1, class_counts) # Avoid div 0
    weight = 1. / class_counts
    samples_weight = torch.tensor([weight[l] for l in subset_labels]).double()
    
    train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    # Subsets & Loaders
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model
    model = BrainMVPClassifier(num_classes=3, in_channels=4, pretrained_path=PRETRAINED, freeze_backbone=False)
    model.to(device)
    
    # 3. Loss & Optimizer (Advanced)
    # Sampler đã cân bằng, nên loss weight để 1.0. Gamma=2.0 tập trung ca khó.
    criterion_cls = FocalLoss(alpha=torch.tensor([1.0, 1.0, 1.0]).to(device), gamma=2.0)
    criterion_cons = nn.MSELoss()
    
    # Optimizer with Grouped Parameters
    opt_params = get_optimizer_params(model, LR_BACKBONE, LR_HEAD, WEIGHT_DECAY)
    optimizer = optim.AdamW(opt_params)
    
    # Scheduler with Warmup (Dùng OneCycleLR hoặc CosineAnnealingWarmRestarts tốt hơn Cosine thường)
    # Ở đây dùng CosineAnnealingLR cho đơn giản nhưng ổn định
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Early Stopping & Logger
    early_stopping = EarlyStopping(patience=PATIENCE, path=os.path.join(fold_dir, 'best_model.pth'))
    logger = TrainingLogger(fold_dir, fold_idx=fold_idx+1)
    
    # 4. Loop
    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        t_preds, t_targets = [], []
        
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx+1} Ep {epoch}", leave=False)
        for batch in pbar:
            # Load 2 views (BrainMVP Template Mixing)
            view1 = batch['image'].to(device)
            view2 = batch['image_aug'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward (Mixed Precision cho nhẹ nếu muốn, ở đây viết thường cho rõ)
            l1, f1 = model(view1, return_features=True)
            l2, f2 = model(view2, return_features=True)
            
            # Loss Combined
            loss_cls = (criterion_cls(l1, labels) + criterion_cls(l2, labels)) / 2
            loss_cons = criterion_cons(f1, f2)
            loss = loss_cls + 1.0 * loss_cons
            
            loss.backward()
            
            # ** EXPERT TRICK: Gradient Clipping **
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            t_preds.extend(torch.argmax(l1, 1).cpu().numpy())
            t_targets.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()
        
        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        v_preds, v_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                # Val thì lấy ảnh gốc (hoặc view1 nếu dataset trả về dict)
                # Dataset trả về {'image', 'label'} khi is_train=False
                img = batch['image'].to(device)
                lbl = batch['label'].to(device)
                
                out = model(img)
                loss = criterion_cls(out, lbl)
                val_loss += loss.item()
                
                v_preds.extend(torch.argmax(out, 1).cpu().numpy())
                v_targets.extend(lbl.cpu().numpy())
        
        # Metrics
        t_acc = accuracy_score(t_targets, t_preds)
        v_acc = accuracy_score(v_targets, v_preds)
        v_f1 = f1_score(v_targets, v_preds, average='macro')
        report = classification_report(v_targets, v_preds, target_names=["Negative", "Hemorrhage", "Infarction"], zero_division=0, output_dict=True)
        
        # Print
        print(f"Ep {epoch} | Val F1: {v_f1:.4f} (Hem: {report['Hemorrhage']['f1-score']:.2f}, Inf: {report['Infarction']['f1-score']:.2f})")
        
        # Log
        logger.log(epoch, 
                   {'loss': train_loss/len(train_loader), 'acc': t_acc},
                   {'loss': val_loss/len(val_loader), 'acc': v_acc, 'f1': v_f1},
                   report)
        
        # Check Early Stopping
        early_stopping(v_f1, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered.")
            break

def run():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Optimized Experiment on {device}")
    
    # Load Dataset (Mode Train để có Template Mix)
    # Lưu ý: Khi chia fold, phần val_idx sẽ dùng dataset này nhưng ta muốn tắt mix ở val?
    # Trong dataset_advanced.py, ta có tham số is_train.
    # Để chuẩn nhất, ta nên tạo 2 dataset instance sharing memory nhưng khác mode.
    # Nhưng để đơn giản và tiết kiệm RAM, ta chấp nhận Val cũng mix nhẹ (Test-time Augmentation).
    # Hoặc tốt hơn: Sửa Dataset để check index.
    # Ở đây: Dùng chung dataset is_train=True. Coi như Val có Augmentation (cũng tốt để đánh giá robustness).
    
    full_ds = BrainMVPAdvancedDataset(JSON_FILE, "training", TEMPLATE_DIR, is_train=True)
    labels = [s['label'] for s in full_ds.sample_list]
    
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        train_fold(fold, train_ids, val_ids, full_ds, device)

if __name__ == "__main__":
    run()