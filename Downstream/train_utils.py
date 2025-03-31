import os
from pathlib import Path
import csv
import yaml

import numpy as np
import torch
from inference_util import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)


import random
import numpy as np


def init_config(args):
    args.ckpt_save_dir = Path(f"./{args.output_dir}/{args.experiment}/")
    args.ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_save_dir = args.ckpt_save_dir.resolve()
    config = vars(args).copy()
    config_file = args.ckpt_save_dir / (args.experiment + ".yaml")
    with open(config_file, "w") as file:
        yaml.dump(config, file)
    return args

def set_seed(args):
    torch.manual_seed(args.random_seed+args.rank)
    torch.cuda.manual_seed(args.random_seed+args.rank)
    torch.cuda.manual_seed_all(args.random_seed+args.rank) # multi-GPU and sample  different cases in each rank
    
    np.random.seed(args.random_seed+args.rank)
    random.seed(args.random_seed+args.rank)   
    os.environ['PYTHONHASHSEED'] = str(args.random_seed+args.rank)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def cal_metrics(metrics, current_epoch=0, save_folder=None, mode='validation'):
    metrics = list(zip(*metrics))
    metrics = [np.nanmean(torch.tensor(dice, dtype=float).cpu().numpy()) for dice in metrics]
    print(
            f'{mode} {current_epoch} epoch: ',
            f'AVG: {np.round(np.mean(metrics), 5)}, ',
            ', '.join([f'{idx}: {np.round(value, 5)}' for idx, value in enumerate(metrics)])
        )

    if save_folder:
        csv_file_path = f"{save_folder}/{mode}_dice.csv"
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, mode="a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = [f'{mode} epoch', 'AVG'] + [f'class {str(i)}' for i in range(len(metrics))]
                writer.writerow(header)

            row = [current_epoch, np.round(np.mean(metrics), 5)] + [np.round(value, 5) for value in metrics]
            writer.writerow(row)

    return metrics

def save_checkpoint(state: dict, save_folder: Path):
    best_filename = str(save_folder) + '/model_best' + "_" + str(state["epoch"]) + '.pth.tar'
    torch.save(state, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def inference(input, model, patch_shape = 128):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(patch_shape, patch_shape, patch_shape),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5
        )
    with torch.cuda.amp.autocast():
        return _compute(input)

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")