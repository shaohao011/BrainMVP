import torch
import torch.nn as nn


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
                
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target, is_train=False):
        dice = 0
        if is_train:
            ce = 0
            ce_loss = torch.nn.BCELoss()
            for i in range(target.size(1)):
                dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
                ce = ce + ce_loss(torch.sigmoid(inputs[:, i, ...]), target[:, i, ...])
            final_dice = (0.7 * dice + 0.3 * ce) / target.size(1)
        else:
            for i in range(target.size(1)):
                dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
            final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices



