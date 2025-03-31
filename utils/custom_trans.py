from __future__ import annotations
from typing import Optional
from numpy import ndarray
import random
import numpy as np
import torch
from collections.abc import Callable, Hashable, Mapping, Sequence
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms.croppad.array import CropForeground
from monai.transforms.transform import LazyTransform, MapTransform, Randomizable
from monai.transforms import Cropd
from monai.transforms.utils import is_positive
from monai.utils import PytorchPadMode, deprecated_arg_default, ensure_tuple_rep
import copy

class Sample_fix_seqd(Randomizable,MapTransform):
    def __init__(
        self,
        keys,
        k : int = 4,
        allow_missing_keys: bool = False,
    ) -> None:
        self.k = k
        self.index = []
        super().__init__(keys, allow_missing_keys)
    

    def __call__(self, data):
        d = dict(data) 
        if len(d["image"])==4:
            return d
        for key in self.key_iterator(d):    
            if key == "image":
                d[key],sampled_indices = self.sample_with_index(d[key],k=self.k) 
                self.index = sampled_indices
            if key == "series_name":
                d[key] = d[key].split(',')
                assert self.index != [] ,"self.index is []"
                # print(self.index)
                d[key] = [d[key][i] for i in self.index] 
                d[key] = ",".join(d[key])
                self.index = []
        return d

    
    def sample_with_index(slef,data, k):
        indices = list(range(len(data))) 
        if len(data)>k:
            sampled_indices = random.sample(indices, k=k) # 无放回
        else:
            sampled_indices = random.choices(indices, k=k-len(data)) # 有放回
            sampled_indices = indices + sampled_indices
        sampled_data = [data[i] for i in sampled_indices]  
        assert len(sampled_data)==k
        return sampled_data, sampled_indices

class CenterCropForeground(CropForeground):
    def __init__(
        self,
        select_fn: Callable = is_positive,
        channel_indices: IndexSelection | None = None,
        margin: Sequence[int] | int = 0,
        allow_smaller: bool = True,
        return_coords: bool = False,
        k_divisible: Sequence[int] | int = 1,
        mode: str = PytorchPadMode.CONSTANT,
        lazy: bool = False,
        **pad_kwargs,
    ) -> None:
        super().__init__(select_fn,
        channel_indices,
        margin,
        allow_smaller,
        return_coords,
        k_divisible,
        mode,
        lazy,
        **pad_kwargs)
        
    def compute_bounding_box(self, img: torch.Tensor) -> tuple[ndarray, ndarray]:
        box_start, box_end = super().compute_bounding_box(img)
      
        center_coord = np.asarray(img.shape[1:]) / 2 # shape (3,)
        start_dist = np.abs(box_start - center_coord) # shape (3,)
        end_dist = np.abs(box_end - center_coord) # shape (3,)
        refined_dist = np.max(np.stack([start_dist, end_dist]), axis=0) # shape (3,)
        
        n_box_start = center_coord - refined_dist
        n_box_end = center_coord + refined_dist
        
        # print(box_start, box_end, n_box_start, n_box_end, img.shape[1:])
        
        return n_box_start.astype(int), n_box_end.astype(int)

class CenterCropForegroundd(Cropd):
    @deprecated_arg_default("allow_smaller", old_default=True, new_default=False, since="1.2", replaced="1.5")
    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        select_fn: Callable = is_positive,
        channel_indices: IndexSelection | None = None,
        margin: Sequence[int] | int = 0,
        allow_smaller: bool = True,
        k_divisible: Sequence[int] | int = 1,
        mode: SequenceStr = PytorchPadMode.CONSTANT,
        start_coord_key: str | None = "foreground_start_coord",
        end_coord_key: str | None = "foreground_end_coord",
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **pad_kwargs,
    ) -> None:
        self.source_key = source_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        cropper = CenterCropForeground(
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=margin,
            allow_smaller=allow_smaller,
            k_divisible=k_divisible,
            lazy=lazy,
            **pad_kwargs,
        )
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        self.cropper.lazy = value

    @property
    def requires_current_data(self):
        return True

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        # self.cropper: CropForeground
        box_start, box_end = self.cropper.compute_bounding_box(img=d[self.source_key])
        if self.start_coord_key is not None:
            d[self.start_coord_key] = box_start 
        if self.end_coord_key is not None:
            d[self.end_coord_key] = box_end 

        lazy_ = self.lazy if lazy is None else lazy
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m, lazy=lazy_)
        return d
    
class RecordAffined(MapTransform):
    def __init__(
        self,
        keys,
        allow_missing_keys: bool = False,
    ) -> None:
    
        super().__init__(keys, allow_missing_keys)
    

    def __call__(self, data):
        d = dict(data) 
        d['image'].meta['orient_affine'] = copy.deepcopy(d['image'].meta['affine'])
        d['image'].meta['new_dim'] = copy.deepcopy(torch.tensor(d['image'].shape[1:]))
        return d