from monai.transforms import Compose
from monai.data import (DataLoader,CacheDataset)
from monai import transforms
import random
import os
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler as Sampler
from creat_new_dst import CustomizedMonaiDataset
from utils.sample import DistributedWeightedRandomSampler
from utils.random_zoom import Randomzoomd

def get_loader(args):
    json_dir = args.json_dir
    datalist_json = os.path.join(json_dir,f"{args.dataset}"+".json")
    
    # NOTE load npy.json here
    import json
    with open(datalist_json, "r") as f:
        json_content = json.load(f)
    train_list = json_content["training"]
    val_list = json_content["validation"]
    test_list = json_content["test"]
    
    random.shuffle(train_list)
    if args.debug:
        train_list = train_list[:6]
        val_list = val_list[:6]
        test_list = test_list[:6] 

    if args.rank==0:print(f"[!]len(train) {len(train_list)} len(val) {len(val_list)} len(test) {len(test_list)}")
    
    train_transforms = Compose(
        [
            # Randomzoomd(keys="image"),
            transforms.RandSpatialCropd(keys=["image"], roi_size=args.img_patch_size, random_size=False),
            transforms.RandRotated(keys=["image"],range_x=0.1, range_y=0.1, range_z=0.1, prob=0.3, keep_size=True),
            transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"])  # 
        ]
    )

    val_transforms = Compose(
        [   
            transforms.CenterSpatialCropd(keys=["image"],roi_size = args.img_patch_size),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transforms = Compose(
    [   
        transforms.CenterSpatialCropd(keys=["image"],roi_size = args.img_patch_size),
        transforms.ToTensord(keys=["image", "label"]),
    ]
    )
    
    
    if args.test_mode:
        test_ds = CustomizedMonaiDataset(data=test_list, transform=test_transforms)
        val_sampler =  None
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, sampler=val_sampler, pin_memory=True,)
        loader = test_loader
    else:
        if args.cache_dataset:
            train_ds = CacheDataset(data=train_list, transform=train_transforms,cache_rate=1.0, num_workers=args.num_workers)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_ds = CacheDataset(data=val_list, transform=val_transforms,cache_rate=1.0, num_workers=args.num_workers)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        else:
            train_ds = CustomizedMonaiDataset(data=train_list, transform=train_transforms)
            train_val_ds = CustomizedMonaiDataset(data=train_list, transform=val_transforms)
            if not args.distributed:
                train_sampler = WeightedRandomSampler(weights=train_ds.weights,num_samples=len(train_ds))
                # train_sampler = None
            else:
                train_sampler = DistributedWeightedRandomSampler(train_ds,weights=train_ds.weights,num_samples=len(train_ds))
            # train_sampler = Sampler(train_ds) if args.distributed else None
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                # shuffle=None if args.distributed else True,
                shuffle=False,
                num_workers=args.num_workers,
                sampler=train_sampler,
                pin_memory=True,
                drop_last=False
               )
            val_ds = CustomizedMonaiDataset(data=val_list, transform=val_transforms)
            val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
            # val_sampler = None
            val_loader = DataLoader(
                val_ds, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=args.num_workers, 
                sampler=val_sampler, 
                pin_memory=True, 
            )
            train_val_loader = DataLoader(
                train_val_ds, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=args.num_workers, 
                sampler=val_sampler, 
                pin_memory=True, 
            )
        loader = [train_loader, val_loader,train_val_loader]

    return loader
