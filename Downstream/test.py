import argparse
import os
from tqdm import tqdm
from pathlib import Path

import json
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
from torch.cuda.amp import autocast as autocast

from train_utils import inference, cal_metrics, dice_metric, post_trans
from dataset.multimodal_dataset import format_input
from dataset.transforms import custom_transform
from model.Uni_unet import UniUnet
from loss.dice import EDiceLoss
from monai.data import DataLoader, Dataset, decollate_batch
from monai.data.utils import pad_list_data_collate

parser = argparse.ArgumentParser(description='BrainMVP downstream segmentation testing')

parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--patch_shape', default=96, type=int, help='input shape, default to 96')
parser.add_argument('--in_channels', default=4, type=int, help="channels of model inputs")
parser.add_argument('--out_classes', default=3, type=int, help="channels of model outputs")
parser.add_argument('--checkpoint', default='', type=str, help='path of checkpoint to load from')

parser.add_argument('--workers', default=8, type=int, help='number of workers, default to 8')                    
parser.add_argument('--devices', default='0', type=str, help='cuda visible devices, default to 0')
parser.add_argument("--local_rank", type=int, default=0, help="local rank")

parser.add_argument('--dataset', type=str, default='upenngbm', choices=['brats18', 'brats20', 'brats23-ped', 'brats23-met',
                                                                        'isles22','mrbrains13', 'vsseg', 'upenngbm'])
parser.add_argument('--data_root', default='', type=str, help='root path to images')
parser.add_argument('--json_file', default='dataset_upenngbm_ds.json', type=str, help='json file name')


def run_inference(data_loader, model, metric, patch_shape = 96):
    
    metrics = []

    for val_data in tqdm(data_loader):
        model.eval()
        with torch.no_grad():
            val_inputs = val_data["image"].cuda()
            val_labels = val_data["label"].cuda()
            val_outputs = inference(val_inputs, model, patch_shape = patch_shape)
            val_preds = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_preds, y=val_labels)

        metric_ = metric(val_outputs, val_labels)
        metrics.extend(metric_)
    print('Dice metric: ', dice_metric.aggregate().mean().item())
    class_avg_metrics = cal_metrics(metrics, mode='test')

    dice_metric.reset()

    return np.mean(class_avg_metrics) 

def get_test_loader(args):
    data_root = Path(args.data_root)
    with open(os.path.join(data_root, args.json_file), 'r') as fr:
        data_list = json.load(fr)
    test_list = format_input(data_root, data_list["test"])
    print(f'Test length: {len(test_list)}')
    test_transform = custom_transform(patch_shape=args.patch_shape, mode='test')
    test_dataset = Dataset(data=test_list, transform=test_transform)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn = pad_list_data_collate, pin_memory=torch.cuda.is_available())
    return val_loader

def main(args):
    
    print("Building model ...")
    model = UniUnet(input_shape = args.patch_shape, in_channels=args.in_channels, out_channels=args.out_classes, multi_scale=True)

    ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.cuda()

    loss_func = EDiceLoss().cuda()
    eval_metrics = loss_func.metric
    
    test_loader = get_test_loader(args)
    class_avg_dice = run_inference(test_loader, model, eval_metrics, patch_shape=args.patch_shape)
    print(class_avg_dice)


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
