import numpy as np
import numpy as np
from collections.abc import Callable, Sequence
from typing import Dict, List, Any, Union, Tuple
from monai.data import Dataset as MonaiDataset,load_decathlon_datalist,DataLoader
from monai.transforms import Compose,apply_transform
import time
from monai.data.meta_tensor import MetaTensor
import os
import json
from tqdm import tqdm
from monai import transforms

def load_preprocessed_data(data_path:str):
    data = np.load(data_path, allow_pickle=True).item()
    data["image"] = MetaTensor(data["image"])
    data["label"] = MetaTensor(data["label"])
    data["seg"] = MetaTensor(data["seg"])#.unsqueeze(dim=0) Attention data shape
    return data

class CustomizedMonaiDataset(MonaiDataset):
    def __init__(self, data: Sequence, transform: Union[Callable[..., Any], None] = None) -> None:
        super().__init__(data, transform)
    
    @property
    def weights(self):
        cls_cnt = {}
        for data_i in self.data:
            label_i = data_i.split("/")[-3]
            if label_i not in cls_cnt: cls_cnt[label_i] = 0
            cls_cnt[label_i] += 1
        cls_cnt = {k: 1 / cls_cnt[k] for k in cls_cnt}
        weights = np.zeros((len(self.data),), dtype=np.float32)
        for idx, data_i in enumerate(self.data):
            label_i = data_i.split("/")[-3]
            weights[idx] = cls_cnt[label_i]
        return weights
    
    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = load_preprocessed_data(self.data[index])
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
 

def save_data(data:Dict[str, Any], save_path:str):
    data = {k: data[k].get_array() if isinstance(data[k], MetaTensor) else data[k] for k in data}
    np.save(save_path, data, allow_pickle=True)


def transform_and_index_dataset(dataset_folder:str, full_metadata_file:str, save_folder:str, transform,save_json_path:str):
    dataset_folder = os.path.abspath(dataset_folder)
    new_data ={}
    train_list = load_decathlon_datalist(full_metadata_file, False, f"training", base_dir=dataset_folder)
    val_list = load_decathlon_datalist(full_metadata_file, False, f"validation", base_dir=dataset_folder)
    test_list = load_decathlon_datalist(full_metadata_file, False, f"test", base_dir=dataset_folder)
    
    test_ds = MonaiDataset(data=test_list, transform=transform)
    test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=12, sampler=None, pin_memory=False)
    
    train_ds = MonaiDataset(data=train_list, transform=transform)
    train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=False, num_workers=12, sampler=None, pin_memory=False)
    
    val_ds = MonaiDataset(data=val_list, transform=transform)
    val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=12, sampler=None, pin_memory=False)
    
    loaders = [train_loader,val_loader,test_loader]
    splits = ['training','validation','test']
    for idx,loader in enumerate(loaders):
        new_data[splits[idx]] = []
        for batch in tqdm(loader, desc=f"{splits[idx]}"):
            # print(batch['image'].shape,batch['seg'].shape,batch['age'].shape)
            # exit()
            batch['image'] = batch['image'].squeeze(dim=0)
            batch['seg'] = batch['seg'].squeeze(dim=0)
            batch['label'] = batch['label'].squeeze(dim=0)
            # batch['survival'] = batch['survival'].squeeze(dim=0)
            sample_name = batch['image'].meta['filename_or_obj'][0].split('/')[-2]
            save_path = os.path.join(save_folder,splits[idx],sample_name+ ".npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            new_data[splits[idx]].append(os.path.abspath(save_path))
            save_data(batch, save_path)
    with open(save_json_path,'w') as f:
        f.write(json.dumps(new_data, indent=4, ensure_ascii=False))
    return None
if __name__=="__main__":
    import os
    # Fixed transforms
    preproc_transform = Compose(
    [
        transforms.LoadImaged(keys=["image","seg"]),
        transforms.EnsureChannelFirstd(keys=["image","seg"]),
        transforms.Orientationd(keys=["image","seg"], axcodes="RAS"),
        transforms.Spacingd(keys=["image","seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear","nearest")),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], 
        lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True, channel_wise=True), 
        # transforms.CropForegroundd(
        #     keys=["image"], source_key="image"
        # ),
        transforms.Resized(keys=["image","seg"],spatial_size=(128,128,64),mode = ("trilinear","nearest")),  
        ]
        )
    data_path = ""
    full_metadata_file = "jsons/brats18_cls.json"
    
    save_json_path = "jsons/brats18_cls_npy.json"
    save_folder = "./data/brats18_cls"
    index_res = transform_and_index_dataset(dataset_folder=data_path, full_metadata_file=full_metadata_file, save_folder=save_folder, transform=preproc_transform,save_json_path=save_json_path)
    
