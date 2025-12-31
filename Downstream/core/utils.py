# BrainMVP/Downstream/core/utils.py
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch

class TrainingLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, 'training_log.csv')
        self.headers = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1']
        
        # Nếu file chưa tồn tại thì tạo header
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, val_f1):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_f1])

    def plot_history(self):
        # Vẽ biểu đồ từ file CSV
        try:
            df = pd.read_csv(self.log_file)
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss
            ax[0].plot(df['epoch'], df['train_loss'], label='Train Loss')
            ax[0].plot(df['epoch'], df['val_loss'], label='Val Loss')
            ax[0].set_title('Loss History')
            ax[0].set_xlabel('Epoch')
            ax[0].legend()
            
            # Accuracy & F1
            ax[1].plot(df['epoch'], df['val_acc'], label='Val Acc')
            ax[1].plot(df['epoch'], df['val_f1'], label='Val F1')
            ax[1].set_title('Metrics History')
            ax[1].set_xlabel('Epoch')
            ax[1].legend()
            
            plt.savefig(os.path.join(self.save_dir, 'history_plot.png'))
            plt.close()
            print(f"-> Đã lưu biểu đồ tại: {os.path.join(self.save_dir, 'history_plot.png')}")
        except Exception as e:
            print(f"Không thể vẽ biểu đồ: {e}")

def save_checkpoint(state, save_dir, is_best=False):
    # Lưu checkpoint mới nhất (để resume)
    filename = os.path.join(save_dir, 'last_checkpoint.pth')
    torch.save(state, filename)
    
    # Nếu là best thì copy ra file riêng
    if is_best:
        best_filename = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_filename)