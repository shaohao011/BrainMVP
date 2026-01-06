import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch
import shutil

class TrainingLogger:
    def __init__(self, save_dir, fold_idx=None):
        self.save_dir = save_dir
        prefix = f"fold{fold_idx}_" if fold_idx is not None else ""
        self.log_file = os.path.join(save_dir, f'{prefix}training_log.csv')
        
        # Header chi tiết hơn
        self.headers = [
            'epoch', 'train_loss', 'train_acc', 
            'val_loss', 'val_acc', 'val_macro_f1',
            'neg_prec', 'neg_rec', 'neg_f1', # Class 0
            'hem_prec', 'hem_rec', 'hem_f1', # Class 1
            'inf_prec', 'inf_rec', 'inf_f1'  # Class 2
        ]
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, epoch, train_stats, val_stats, report):

        
        def get_metrics(name):
            if name in report:
                return [
                    report[name]['precision'], 
                    report[name]['recall'], 
                    report[name]['f1-score']
                ]
            return [0.0, 0.0, 0.0]

        row = [
            epoch, 
            train_stats['loss'], train_stats['acc'],
            val_stats['loss'], val_stats['acc'], val_stats['f1']
        ]
        row.extend(get_metrics('Negative'))
        row.extend(get_metrics('Hemorrhage'))
        row.extend(get_metrics('Infarction'))

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def save_checkpoint(state, save_dir, filename='checkpoint.pth'):
    path = os.path.join(save_dir, filename)
    torch.save(state, path)