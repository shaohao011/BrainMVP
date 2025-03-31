from typing import Any, Dict, List, Tuple, Union
from collections.abc import Callable, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from monai.transforms import apply_transform

class MaskedInputDataset(Dataset):
    def __init__(
        self, 
        data:List[Dict[str, str]], 
        transform:Union[Callable[[Dict[str, str]], Dict[str, Any]], Dict[str, Dict[str, Callable]]], 
        args
    ) -> None:
        """
            Class-balanced multi-task dataset, composed of several partially labeled datasets.
            The length of this dataset will be sum of (dataset size times number of annotated labels) for all datasets

            Params:
            ----------
                data:           data list, {task_name: [{"image": image_path, "label": label_path}]}
                tasks_metadata: multi-dataset metadata description, {task_name: {label_name: [label_int_1, label_int_2, ...]}}
                transform:      callable transformation function
                ref_data:
                ref_transform:
            
            Returns:
            ----------
                None
        """
        self._data = data
        self.transform = transform
        self.args = args

    @torch.no_grad()
    def _transform(self, index:int):
        """
            Fetch single data item from `self.data`.
        """
        data_i = self._data[index]
        applied_transform = self.transform
        
        transformed_data_i = apply_transform(applied_transform, data_i) if applied_transform is not None else data_i

        if isinstance(transformed_data_i, list):
            for idx in range(len(transformed_data_i)): 
                transformed_data_i[idx]["gt_img"] = transformed_data_i[idx]["image"].clone().detach()
                # transformed_data_i[idx]["modality_in"] = modality_list[idx]
                # transformed_data_i[idx]["image"] = augmented_list[idx]
        else:
            transformed_data_i["gt_img"] = transformed_data_i["image"].clone().detach() 
            # transformed_data_i["image"],transformed_data_i["modality_in"] = aug_rand_list(self.args, [transformed_data_i["image"]],[transformed_data_i["series_name"]])[0]

        return transformed_data_i

    def __len__(self): 
        
        return len(self._data)
    
    def __getitem__(self, index):
        """
            Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, Sequence):
            return Subset(dataset=self, indices=index)
        return self._transform(index)