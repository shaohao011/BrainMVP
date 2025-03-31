from pathlib import Path
import os
import json
from monai.data import Dataset
from .transforms import custom_transform


def get_modality_names(dataset_name: str = 'upenngbm') -> list:
    if dataset_name == 'isles22':
        modality_list = ['ADC', 'DWI', 'FLAIR']
    elif dataset_name == 'mrbrains13':
        modality_list = ['T1', 'T1', 'T2']
    elif dataset_name == 'vsseg':
        modality_list = ['T2']
    else:
        modality_list = ['T1', 'T1C', 'T2', 'FLAIR']
    return modality_list

def format_input(data_root: Path, 
                 filename_list: list, 
                 mix_template: str = False, 
                 template_dir: str = '',
                 dataset_name: str = '') -> list:
    fullname_list = [
        {
            'image': [data_root / y for y in x['image']],
            'label': data_root / x['label']
        } for x in filename_list
    ]
    if mix_template:
        templates = {'template': [Path(template_dir)/ (y+'.nii.gz') for y in get_modality_names(dataset_name)]}
        fullname_list = [{**x, **templates} for x in fullname_list]
    # print(fullname_list[0])
    return fullname_list
    

def get_datasets(args):
    data_root = Path(args.data_root)
    json_root = str(data_root).rsplit("/", 1)[0]

    with open(os.path.join(json_root, args.json_file), 'r') as fr:
        data_list = json.load(fr)

    train_list = format_input(data_root, data_list["training"], args.mix_template, args.template_dir, args.dataset)
    val_list = format_input(data_root, data_list["validation"])
    test_list = format_input(data_root, data_list["test"])
    print("Train: %d \nValidation: %d \nTest: %d" % (len(train_list), len(val_list), len(test_list)))
    
    train_transform = custom_transform(patch_shape=args.patch_shape, mode='train', enable_channel_cutmix=args.mix_template, pair_aug=args.use_cl)
    val_transform = custom_transform(patch_shape=args.patch_shape, mode='val')
    train_dataset = Dataset(data=train_list, transform=train_transform)
    val_dataset = Dataset(data=val_list, transform=val_transform)
    test_dataset = Dataset(data=test_list, transform=val_transform)

    return train_dataset, val_dataset, test_dataset
