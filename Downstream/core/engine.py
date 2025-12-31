# BrainMVP/Downstream/core/engine.py
import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from core.utils import save_checkpoint, TrainingLogger

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(loader), accuracy_score(all_targets, all_preds)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    report = classification_report(all_targets, all_preds, target_names=["Negative", "Hemorrhage", "Infarction"], zero_division=0, output_dict=True)
    return avg_loss, acc, f1, report

def run_training_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, save_dir, resume=True):
    # 1. Khởi tạo Logger
    logger = TrainingLogger(save_dir)
    start_epoch = 1
    best_f1 = 0.0
    
    # 2. Logic RESUME (Tự động load nếu có checkpoint cũ)
    last_ckpt_path = os.path.join(save_dir, 'last_checkpoint.pth')
    if resume and os.path.exists(last_ckpt_path):
        print(f"🔄 Tìm thấy checkpoint cũ tại {last_ckpt_path}. Đang khôi phục...")
        checkpoint = torch.load(last_ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"-> Resume từ Epoch {start_epoch}. Best F1 hiện tại: {best_f1:.4f}")

    # 3. Vòng lặp chính
    print("\n>>> BẮT ĐẦU TRAINING <<<")
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc, val_f1, report = validate(model, val_loader, criterion, device)
        
        # Step LR
        scheduler.step()
        
        # Log & Print
        print(f"Ep {epoch}/{epochs} | Train: Loss {train_loss:.4f} Acc {train_acc:.3f} | Val: Loss {val_loss:.4f} Acc {val_acc:.3f} F1 {val_f1:.3f}")
        print(f"   -> Hemorrhage F1: {report['Hemorrhage']['f1-score']:.2f} | Infarction F1: {report['Infarction']['f1-score']:.2f}")
        
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, val_f1)
        
        # Save Checkpoint
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            print("   -> ⭐ New Best Model!")
            
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_f1': best_f1
        }, save_dir, is_best)

    # 4. Vẽ biểu đồ sau khi xong
    logger.plot_history()