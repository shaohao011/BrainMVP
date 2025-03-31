# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from monai.data import CacheDataset,PersistentDataset, ThreadDataLoader,DataLoader, Dataset, DistributedSampler, SmartCacheDataset
import re
import os
import os.path as osp
import json
import random
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ToTensord,
    ScaleIntensityRangePercentilesd,
    RandSpatialCropSamplesd,
    Orientationd,
    Spacingd,
    SpatialPadd,
    EnsureChannelFirstd
)
from utils.custom_trans import Sample_fix_seqd,RecordAffined,CenterCropForegroundd
from .masked_input_dataset import MaskedInputDataset

def load_dataset_info(dst_json_file):
    with open(dst_json_file, 'r') as fr: 
        dst_json = json.load(fr)
    return dst_json

def load_mmri_sampled_list(dataset_name,dst2mod_json, dst_json_file, data_split, img_dir):
    ### Load dataset info. e.g. {"dataset1": "series_name": [...]}
    dst2mod_json = load_dataset_info(dst2mod_json)

    modality_name = dst2mod_json[dataset_name]["series_name"]
    ### Get data list
    with open(dst_json_file, 'r') as fr:
        json_data = json.load(fr)
    
    data_list = json_data[data_split]
    random.shuffle(data_list)
    post_img_list = []
    for idx, item in enumerate(data_list):
        img_list = item["image"]
        if not isinstance(img_list, list):
            raise ValueError("Image names must be a list, but got %s." % type(img_list).__name__)
        random.shuffle(img_list) # change modality position
        # check modality here
        if not all(re.split('[-|_|]', osp.split(x)[-1][:-len(".nii.gz")])[-1] in modality_name for x in img_list):
            raise ValueError("The split modalit is not in the dict")
        post_img_list.append(
            {
                "image": [osp.join(img_dir, x) for x in img_list],
                "series_name": ",".join(re.split('[-|_|]', osp.split(x)[-1][:-len(".nii.gz")])[-1] for x in img_list)
            }
        )
    return post_img_list


def get_loader(args):
    num_workers = args.num_workers
    
    dataset_names = args.dataset
    base_img_dir = args.base_dir
    base_json_dir = "./jsons"
    
    dst2mod_json = os.path.join(base_json_dir, "./dst_dict.json")
    assert  isinstance(dataset_names,list), "You should input List-like dataset name"
    
    train_list = []
    val_list   = []
    for idx,dataset_name in enumerate(dataset_names):
        json_file = os.path.join(base_json_dir,"dataset_"+dataset_name+".json")
        img_file = os.path.join(base_img_dir,dataset_name)
        data_train_list = load_mmri_sampled_list(dataset_name,dst2mod_json=dst2mod_json,dst_json_file=json_file, data_split="training", img_dir=img_file)
        data_val_list   = load_mmri_sampled_list(dataset_name,dst2mod_json=dst2mod_json,dst_json_file=json_file, data_split="validation", img_dir=img_file)
        train_list += data_train_list
        val_list   += data_val_list
        if args.rank ==0:
            print(f"[!] Dataset {dataset_name}, Training data: {len(data_train_list)}, Validation data: {len(data_val_list)}")
    
    # we use whole data for pre-training
    train_list = train_list + val_list 
    random.shuffle(train_list)
    
    if args.debug:
        train_list = train_list[:500]
    
    if args.rank==0:   
        print("[!] All training: number of data: {}".format(len(train_list)))
        print("[!] All validation: number of data: {}".format(len(val_list)))
    
    train_transforms = Compose(
        [
            # Use fixed sample sequences for better data collation efficiency. You can also set k=2.
            Sample_fix_seqd(keys=["image","series_name"],k=4),
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"],pixdim=(1.0,1.0,1.0),mode="bilinear"),
            CenterCropForegroundd(keys=["image"], source_key="image"),
            ScaleIntensityRangePercentilesd(
                keys=["image"], 
                lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True, channel_wise=True
            ),
            CenterCropForegroundd(keys=["image"], source_key="image"),
            # SaveImaged(keys=["image"],output_dir="./rec_pictures"),
            RecordAffined(keys=["image"]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            ToTensord(keys=["image"]),
        ]
    )

    if args.cache_dataset:
        if args.rank == 0 : print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=train_list, transform=train_transforms,num_workers=num_workers,copy_cache=False)
    elif args.smartcache_dataset:
        if args.rank == 0: print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=train_list,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        if args.rank == 0: print(">>>Using MaskedInputDataset dataset")
        train_ds = Dataset(data=train_list, transform=train_transforms) #
        train_ds = MaskedInputDataset(data=train_list, transform=train_transforms, args=args) #

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, num_replicas=args.world_size, rank=args.rank)
    else:
        from torch.utils.data import RandomSampler
        train_sampler = RandomSampler
    
    if args.cache_dataset:
        # train_loader = ThreadDataLoader(
        #     train_ds, batch_size=args.batch_size, sampler=train_sampler,num_workers=num_workers,
        # )
        train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, 
        sampler=train_sampler,num_workers=num_workers,
        drop_last=True
        # collate_fn=pad_list_data_collate
    )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, 
            sampler=train_sampler,num_workers=num_workers,
            drop_last=True,
            pin_memory = True
            # collate_fn=pad_list_data_collate
        )
        
    return train_loader