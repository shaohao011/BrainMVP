import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
from torch.cuda.amp import autocast as autocast

from train_utils import init_config, set_seed
from trainer import run_training
from dataset.multimodal_dataset import get_datasets
from dataset.transforms import *
from model.Uni_unet import UniUnet
from loss.dice import EDiceLoss
from monai.data import DataLoader
from monai.data.utils import pad_list_data_collate
import nibabel as nib
import numpy as np
parser = argparse.ArgumentParser(description='BrainMVP downstream segmentation finetune')
parser.add_argument('--start_epoch', default=0, type=int, help='epoch where start training')
parser.add_argument('--max_epochs', default=300, type=int, help='total number of training epoch')

parser.add_argument('--eval_interval', default=10, type=int, help="epoch interval to run validation")
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--patch_shape', default=96, type=int, help='input shape, default to 96')
parser.add_argument('--in_channels', default=4, type=int, help="channels of model inputs")
parser.add_argument('--out_classes', default=3, type=int, help="channels of model outputs")
parser.add_argument('--resume', default='', type=str, help='path of checkpoint to resume from')
parser.add_argument('--pretrained', default='', type=str, help="Pretrained weight path")
parser.add_argument('--mix_template', default=False, type=bool, help='whether to apply random channel substitute with template, default to False')
parser.add_argument('--template_dir', default='', type=str, help='directory to template images (used when mix_template is true)')
parser.add_argument('--use_cl', default=False, type=bool, help='whether to apply contrastive loss, default to False')
parser.add_argument('--cl_weight', default=0.5, type=float, help="train sub")
parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--wd', default=0.00001, type=float, help='weight decay')
parser.add_argument('--eta_min', default=0, type=float, help='minimum learning rate, default to 0')

parser.add_argument('--workers', default=8, type=int, help='number of workers, default to 8')                    
parser.add_argument('--devices', default='0', type=str, help='cuda visible devices, default to 0')
parser.add_argument('--random_seed', type=int,default=42, help='random seed')
parser.add_argument("--local_rank", type=int, default=0, help="local rank")

parser.add_argument('--dataset', type=str, default='upenngbm', choices=['brats18', 'brats20', 'brats23-ped', 'brats23-met',
                                                                        'isles22','mrbrains13', 'vsseg', 'upenngbm'])
parser.add_argument('--data_root', default='', type=str, help='root path to images')
parser.add_argument('--json_file', default='dataset.json', type=str, help='json file name')
parser.add_argument('--experiment', default='baseline', type=str, help='exp name')
parser.add_argument('--output_dir', default='runs', type=str, help='output dir')
parser.add_argument('--cfg', type=str, default="configs/config.yaml", help='path to config file')


def main(args):
    
    args.rank = 0
    init_config(args)
    print(args)
    set_seed(args)
    
    # Model initialization.
    print("Building model ...")
    model = UniUnet(input_shape = args.patch_shape, in_channels=args.in_channels, out_channels=args.out_classes, multi_scale=True)

    #### Load weights.
    if args.resume:
        resume_ck = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(resume_ck['state_dict'], strict=True)
    elif args.pretrained:
        ck = torch.load(args.pretrained, map_location=torch.device('cpu'))
        pre_ckpt = ck['state_dict']
        new_ckpt = {}
        for key in pre_ckpt:
            new_key = key.replace('module.', '').replace('uniformer.', '')
            new_ckpt[new_key] = pre_ckpt[key]  
        pre_ckpt = new_ckpt  
        del pre_ckpt['encoder.patch_embed1.proj.weight']
        del pre_ckpt['encoder.patch_embed1.proj.bias']
        del pre_ckpt['decoder.proj1.proj.weight']
        del pre_ckpt['decoder.proj1.proj.bias']
        model.load_state_dict(pre_ckpt, strict=False)
        print("Pretrained weight loaded from %s" % args.pretrained)
        # seperate templates
        rep_templates = pre_ckpt['rep_template']
        tem_list = ['flair','t1','t1c', 't2', 'mra','pd','dwi','adc']
        for i in range(rep_templates.shape[0]):
            nii_data = rep_templates[i].numpy() 
            nii_img = nib.Nifti1Image(nii_data, affine=np.eye(4)) 
            tem_name = tem_list[i].upper()
            nib.save(nii_img, os.path.join(args.template_dir,f"{tem_name}.nii.gz")) 
            print(f"Saved {tem_name}.nii.gz")
        print("All NIfTI images saved successfully!")
        del pre_ckpt 
    else:
        print("Do not load pretrained weight.")
    model = model.cuda()
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Loss func
    loss_func = EDiceLoss().cuda()
    eval_metrics = loss_func.metric

    # Optim
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=args.eta_min)
    if args.resume:
        optimizer.load_state_dict(resume_ck["optimizer"])
        scheduler.load_state_dict(resume_ck["scheduler"])
        
    # Create dataloader.
    
    train_dataset, val_dataset, _ = get_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn = pad_list_data_collate, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn = pad_list_data_collate, pin_memory=torch.cuda.is_available())

    print(f'Length of train loader: {len(train_loader)}')
    print(f'Length of validation loader: {len(val_loader)}')

    start_epoch = 0 if not args.resume else resume_ck["epoch"]
    
    # Train epoch
    max_val_dice = run_training(model, 
                                start_epoch, 
                                train_loader, 
                                val_loader, 
                                optimizer, 
                                scheduler, 
                                loss_func, 
                                eval_metrics, 
                                args)
    return max_val_dice


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
