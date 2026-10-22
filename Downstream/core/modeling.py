import torch
import torch.nn as nn
import sys
import os

# Hack path để import model gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.uniformer import uniformer_small

class BrainMVPClassifier(nn.Module):
    def __init__(self, num_classes=3, in_channels=4, pretrained_path=None, freeze_backbone=False):
        super().__init__()
        
        # 1. Khởi tạo Backbone
        self.backbone = uniformer_small(in_chans=in_channels)
        
        # 2. Load Weights (nếu có)
        if pretrained_path:
            self._load_weights(pretrained_path, in_channels)
            
        # 3. Freeze/Unfreeze logic
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("-> Backbone FROZEN.")
        else:
            print("-> Backbone TRAINABLE (Full Fine-tuning).")
        
        # 4. Định nghĩa các lớp Head
        self.embed_dim = 512
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # --- SỬA LỖI TẠI ĐÂY: Khai báo Dropout riêng ---
        self.dropout = nn.Dropout(0.5) 
        
        # MLP Head
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _load_weights(self, path, target_channels):
        print(f"Loading weights from: {path}")
        try:
            ckpt = torch.load(path, map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            new_state_dict = {}
            for k, v in state_dict.items():
                # Clean Key Prefix
                clean_key = k
                if "module.encoder.uniformer." in k:
                    clean_key = k.replace("module.encoder.uniformer.", "")
                elif "encoder.uniformer." in k:
                    clean_key = k.replace("encoder.uniformer.", "")
                
                if "decoder" in k or "template" in k: continue
                
                # Weight Inflation
                if "patch_embed1.proj.weight" in clean_key:
                    if v.shape[1] == 1 and target_channels == 4:
                        print(f"   -> Inflating {clean_key}: 1 -> 4 channels.")
                        v = v.repeat(1, target_channels, 1, 1, 1) / float(target_channels)
                
                new_state_dict[clean_key] = v
            
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. Missing keys (Head): {len(msg.missing_keys)}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def forward(self, x, return_features=False):
        # 1. Forward Backbone
        out = self.backbone(x)
        x4 = out[-1] # Lấy feature map sâu nhất (Batch, 512, D, H, W)
        
        # 2. Pooling -> Features Vector (Batch, 512)
        features = self.avg_pool(x4).flatten(1)
        
        # 3. Dropout (Gọi self.dropout đã khai báo ở __init__)
        x = self.dropout(features)
        
        # 4. Head -> Logits
        logits = self.head(x)
        
        if return_features:
            return logits, features
        return logits