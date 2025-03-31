from typing import Any
import torch
from torch.nn import functional as F


class Contrast_Loss(torch.nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def __call__(self, image_feat, text_feat) :
        N,D = image_feat.shape
                 
        image_feat = torch.nn.functional.normalize(image_feat, dim=1)
        text_feat = torch.nn.functional.normalize(text_feat, dim=1)
        logits = torch.matmul(image_feat, text_feat.t())
        logits /= self.temperature
        gt = torch.arange(N, device=logits.device)
        loss1 = torch.nn.functional.cross_entropy(logits, gt)
        # print(f"loss1: {loss1}")
        loss2 = torch.nn.functional.cross_entropy(logits.t(), gt)
        # print(f"loss1: {loss2}")
        loss = (loss1 + loss2) / 2
        return loss