import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import random
import yaml
import pathlib
from collections import OrderedDict
import torch
from utils.data_utils import get_loader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels") #4 squence
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction") # NOTE  
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
#====================training hyperparameters=============
parser.add_argument("--accumulation_steps", default=2, type=int, help="steps of accumulation") 
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size") 
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of crop samples") 
parser.add_argument("--lrdecay", action="store_false", help="enable learning rate decay") # True
parser.add_argument("--lr_schedule", default="warmup_cosine", type=str) # warmup_cosine
parser.add_argument("--lr", default=3e-3, type=float, help="learning rate") # 3e-3
parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm") # adamW
parser.add_argument("--decay", default=1e-5, type=float, help="decay rate") # 1e-5
parser.add_argument("--momentum", default=0.9, type=float, help="momentum") # 0.9
#====================Save and train epochs=============
parser.add_argument("--max_epochs", default=300, type=int, help="number of training iterations")
parser.add_argument("--save_interval",default=10,type=int,help="save interval for checkpoint")
parser.add_argument("--warmup_steps", default=50, type=int, help="warmup steps") # lr warm_up
parser.add_argument("--resume", default=None, type=str, help="resume training") #NOTE resume
#====================build envs=============
parser.add_argument("--local-rank", type=int, default=0, help="local rank") 
parser.add_argument("--noamp", action="store_false", help="do NOT use amp for training") 
parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset") 
parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
parser.add_argument("--num_workers", type=int, default=16, metavar='N',
                help='how many training processes to use (default: 16)')
#=================== Uniformer parameters====
parser.add_argument('--initial_checkpoint', default='', type=str, metavar='PATH',
                help='Initialize model from this checkpoint (default: none)') 
#===================   MAE Part   ===========
parser.add_argument("--mask_rate", default=0.875, type=float, help="rate of masking.")#NOTE mask_rate Here
parser.add_argument("--mask_block_size", default=8, type=int, help="block size os masking.")#NOTE mask_block_size Here
#===================  Data Augmentation =======
parser.add_argument("--random_seed", default="46",type=int, help="random seed")
parser.add_argument("--dataset", default=["BraTS2018"],nargs="+",type=str, help="dataset for pretraining")
parser.add_argument("--base_dir", type=str, default="./jsons", help="base JSON configuration folder")
parser.add_argument("--debug", action="store_true", help="whether use cross_attention to cal feature for reconstruction") # 
parser.add_argument("--dst_h", default="300",type=int, help="h for template")
parser.add_argument("--dst_w", default="300",type=int, help="w for template")
parser.add_argument("--dst_d", default="300",type=int, help="d for template")
parser.add_argument("--num_modals", default=4,type=int, help="#modals for training")
parser.add_argument("--template_index", default=["flair","t1","t1c","t2"],nargs="+",type=str, help="")
parser.add_argument("--use_Unet", action="store_true", help="modal type for pretrain")
parser.add_argument("--start_epoch", default=-1,type=int, help="start epoch of training")

def save_args(args):
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()
    del config['save_folder']
    config_file = args.save_folder / ("config.yaml")
    with open(config_file, "w") as file:
        yaml.dump(config, file)

def save_ckpt(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)

def train_epoch(args, model,
           train_loader, optimizer,scheduler,epoch):
    model.train()
    psnr_train = []
    loss_train = []
    loss_contrast = []
    accumulation_steps = args.accumulation_steps
    for step, batch in enumerate(train_loader): # batch :torch.Size([2, 4, 96, 96, 96])
        # continue
        loss_case = []
        M = batch['image'].shape[1]
        x = batch["image"].to(args.device)
        series_name = batch['series_name']
        
        affine_matrix = batch['image'].meta['affine'][:, :3, 3]  # (B,3)
        rearranged_series = []
        num_mods = len(series_name[0].split(','))
        # rearranged_series: [B,num_elements]: modalities of same position
        for i in range(num_mods):
            combined_values = ','.join([elem.split(',')[i] for elem in series_name])
            rearranged_series.append(combined_values)
            
        # get the corresponding ROI of template
        with torch.no_grad():
            dim_matrix = batch['image'].meta['new_dim'] #(B,3)
            start_point = affine_matrix -batch['image'].meta['orient_affine'][:, :3, 3]
            rep_shape = [args.dst_h,args.dst_w,args.dst_d]
            center_shift = (torch.tensor(rep_shape) - dim_matrix)//2
            rep_start_point = start_point + center_shift
        
        # select a modality for future masking
        cur_modal = random.randint(0,M-1)
        loss,loss_rec_1,loss_rec_2,loss_cons = model(x,args=args,index=cur_modal,rep_start_point=rep_start_point,rearranged_series=rearranged_series,cur_epoch=epoch,series_name=series_name)

        loss_case.append(loss_rec_1)

        loss_case.append(loss_rec_2)
        loss_contrast.append(loss_cons)
        
        loss = loss/accumulation_steps
        loss.backward()
 
        loss_train.append(np.mean(loss_case))
        psnr_train.append(0.0)
        
        if step % accumulation_steps ==0:
            optimizer.step()
            optimizer.zero_grad()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()    
    return  np.mean(loss_train), np.mean(loss_contrast),np.mean(psnr_train)

def load_checkpoint(args,model, checkpoint_path):
    print ('>>>Load Video weight for Pretraining: ', checkpoint_path)
    ##### check state dict
    model_state_dict = model.state_dict()
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    checkpoint = OrderedDict([("encoder.uniformer." + k, v) if "encoder.uniformer." + k in model_state_dict.keys() else (k, v) for k,v in checkpoint.items()])
    del checkpoint['encoder.uniformer.patch_embed1.proj.weight']
    del checkpoint['encoder.uniformer.patch_embed2.proj.weight'] # 3D kernel size is not consistent with the uniformer video
    del checkpoint['encoder.uniformer.patch_embed3.proj.weight']
    del checkpoint['encoder.uniformer.patch_embed4.proj.weight']
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    return model


def set_seed(args):
    # multi-GPU and sample different cases in each rank
    torch.manual_seed(args.random_seed+args.rank)
    torch.cuda.manual_seed(args.random_seed+args.rank)
    torch.cuda.manual_seed_all(args.random_seed+args.rank) 
    
    np.random.seed(args.random_seed+args.rank)
    random.seed(args.random_seed+args.rank)   
    os.environ['PYTHONHASHSEED'] = str(args.random_seed+args.rank)

    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True # train speed is slower after enabling this opts.
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    # torch.use_deterministic_algorithms(True)

def main(args):
    args = parser.parse_args()
    logdir = "./runs/" + args.logdir +"/logs"
    args.amp = not args.noamp
    torch.autograd.set_detect_anomaly(True) 
    args.distributed = False 
    if "WORLD_SIZE" in os.environ:   
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1  
    args.rank = 0
    #======================　DDP Environment  ===============
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        # get world_size and rank according to the current device
        args.world_size = torch.distributed.get_world_size() 
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
        
        
    config_path = "./runs/" + args.logdir
    args.save_folder = pathlib.Path(f"./{config_path}/config")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    if args.rank==0:
        save_args(args) 
        print(f"[!]Training log saved to {logdir}")
    
    assert args.rank >= 0
    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None
        
    # for reproducibility
    set_seed(args)
    
    #====================  Load Data ================
    train_loader = get_loader(args)
    
    #====================  Load model [unet, uniformer] ===============
    if args.use_Unet:
        from models.Unet import RecModel
        model = RecModel(args,dim=512)
    else:
        from models.Uniformer import RecModel
        model = RecModel(args, dim=512) # Dimension of the inter feature
        # video ckpt for original uniformer
        if args.initial_checkpoint:
            load_checkpoint(args,model, args.initial_checkpoint)
    
    if args.rank==0:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total parameters count", pytorch_total_params)
    
    # optimizer
    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    else:
        raise NotImplementedError
    
    # scheduler
    scheduler = None
    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=2*len(train_loader), t_total=args.max_epochs*len(train_loader)//args.accumulation_steps) 
        elif args.lr_schedule == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs*len(train_loader)//args.accumulation_steps)
        elif args.lr_schedule == "poly":
            def lambdas(epoch):
                return (1 - float(epoch) / float(args.max_epochs)) ** 0.9
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    
    
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    current_epoch = 0
    val_best = 1e8 # NOTE actually we do not use validation as in MAE
    
    model.to(args.device)
    
    # resume model
    if args.resume and args.rank==0:
        model_pth = args.resume
        model_dict = torch.load(model_pth,map_location="cpu")
        state_dict = model_dict['state_dict']
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.") 
        del state_dict['rep_template']
        model.load_state_dict(state_dict,strict=False)
        optimizer.load_state_dict(model_dict["optimizer"])
        optimizer.to(args.device)
        current_epoch = model_dict["current_epoch"] # NOTE not include scheduler here
        if scheduler:
            scheduler.step(current_epoch*len(train_loader))
        scheduler.load_state_dict(model_dict["scheduler"])
        val_best = model_dict["val_best"]
        if args.rank == 0:
            print(f">>>Resume model from history {args.resume}")
            print(f"[!]current epoch {current_epoch}")
        del model_dict
        model_dict = None
        torch.cuda.empty_cache()
    # wrap model for DDP
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids = [args.local_rank], output_device = args.local_rank, find_unused_parameters =True)
    
    # training starts here
    if args.rank ==0:
        print("[!]Start Training...")
    while current_epoch < args.max_epochs:
        if args.distributed:
            train_loader.sampler.set_epoch(current_epoch)
        # epoch_start = time.time()
        loss_train,loss_contrast, psnr_train= train_epoch(args, model,
                                                        train_loader, optimizer,scheduler,current_epoch)
        current_epoch += 1
        # epoch_end = time.time()
       
        if args.rank==0:
            print(f"Epoch:{current_epoch}/{args.max_epochs}, Loss_total:{loss_train+loss_contrast:.4f},Loss_rec:{loss_train:.4f},"
                  f"Loss_contrast:{loss_contrast:.4f}, PSNR:{psnr_train:2.3f}dB")
        #========================Train===========================
            writer.add_scalar("train/loss_total", scalar_value=loss_train+loss_contrast, global_step=current_epoch)
            writer.add_scalar("train/loss_rec", scalar_value=loss_train, global_step=current_epoch)
            writer.add_scalar("train/loss_contrast", scalar_value=loss_contrast, global_step=current_epoch)
            writer.add_scalar("train/psnr", scalar_value = psnr_train, global_step=current_epoch)
            writer.add_scalar("train/lr", scalar_value=optimizer.param_groups[0]['lr'],global_step=current_epoch)

            # for resume
            if current_epoch % args.save_interval == 0:
                checkpoint = {
                        "current_epoch": current_epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "val_best": val_best
                    }
                save_ckpt(checkpoint, config_path + "/model_fix_save.pt")
                print(f"Model was saved override-mode {args.save_interval}")
            
            if (current_epoch % 100) == 0:
                checkpoint = {
                        "current_epoch": current_epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "val_best": val_best
                    }
                model_path = config_path + f"/model_{current_epoch}.pt"
                save_ckpt(checkpoint,model_path)
                print(f"Model saved to {model_path}")
        
        if args.distributed:
            dist.barrier()    
            
    if args.rank ==0:    
        writer.close()
    if args.distributed:
        dist.destroy_process_group()
    torch.cuda.empty_cache()    

if __name__ == "__main__":
    parser_args = parser.parse_args()
    main(parser_args)
    torch.cuda.empty_cache()    