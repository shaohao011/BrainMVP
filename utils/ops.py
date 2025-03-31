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

import numpy as np
import torch
from numpy.random import randint
import random
import time
import math

mapping = {
    't1': 't1',
    'T1': 't1',
    't2': 't2',
    't1ce': 't1c',
    'flair': 'flair',
    'FLAIR': 'flair',
    't1n': 't1',
    't1c': 't1c',
    'T1c': 't1c',
    't2w': 't2',
    't2f': 'flair',
    'MRA': 'mra',
    'PD': 'pd',
    "DWI": "dwi",
    "dwi": "dwi",
    'T1': 't1',
    'T2': 't2',
    "ADC": "adc"
}

def sample_without_i(lst, i):
    lst_copy = lst.copy()  
    cur_modal = (lst[i])
    set_lst = set(lst_copy)
    set_lst.remove(cur_modal)
    sampled_element = random.choices(list(set_lst),k=1)[0]
    sampled_element = lst.index(sampled_element)  
    return sampled_element

def Uniform_block_replace(args,x_modal_1,x_modal_2=None,template=None):
    x_modal_1_aug = x_modal_1.detach().clone()
   
    mask_rate = args.mask_rate 
    mask_block_size = args.mask_block_size 
    
    h, w, z = x_modal_1_aug.size() # 96,96,96
    assert h % mask_block_size == 0 and w % mask_block_size == 0 and z % mask_block_size == 0
    blk_idx_h, blk_idx_w, blk_idx_z = h // mask_block_size, w // mask_block_size, z // mask_block_size
    cor_1,cor_2,cor_3 = torch.meshgrid(torch.arange(blk_idx_h),torch.arange(blk_idx_w),torch.arange(blk_idx_z))
    cor = torch.stack((cor_1,cor_2,cor_3), dim=3).reshape(-1, 3)
    cor_shuffle = cor[torch.randperm(len(cor))][:math.floor(len(cor)*(1-mask_rate))]
    sparse_tensor = torch.sparse_coo_tensor(cor_shuffle.T, torch.ones((cor_shuffle.shape[0],), dtype=bool), size=(blk_idx_h,blk_idx_w,blk_idx_z)).to_dense()
    preserve_mask = sparse_tensor.unsqueeze(3).unsqueeze(2).unsqueeze(1).expand(-1, mask_block_size, -1, mask_block_size, -1, mask_block_size).reshape(h, w, z)
    mask = ~preserve_mask
    if x_modal_2 is None:
        x_modal_1_aug[mask] = template[mask]
    else:
        x_modal_1_aug[mask] = x_modal_2[mask] 
    return x_modal_1_aug

def aug_rand_with_learnable_rep(args, x,rep_template,index,rep_start_point,rearranged_series,use_temp=False,series_name=None):
    B = x.shape[0]
    x_aug = x.detach().clone()
    template_index = args.template_index
    for i in range(B):
        x_modal_1 = x[i][index]
        h_s,w_s,d_s = [int(i) for i in rep_start_point[i]]
        cur_series = rearranged_series[index].split(',')[i]
        mapped_series = mapping[cur_series]
        index_rep = template_index.index(mapped_series)
        
        template = rep_template[index_rep]
        template = (template - template.min()) / (template.max() - template.min()+1e-9)
        template = template[h_s:h_s+96,w_s:w_s+96,d_s:d_s+96]

        assert list(template.shape)==[96,96,96],f"template: {template.shape}, input: {x_modal_1.shape}"

        if use_temp:
            x_modal_2 = None
        else:
            assert series_name!=None
            sample_list = series_name[i].split(',')
            select_index = sample_without_i(sample_list,index)
            assert select_index!=index
            x_modal_2 = x[i][select_index]
        
        x_aug[i][index] = Uniform_block_replace(args,x_modal_1,x_modal_2,template)
    return x_aug[:,index,:,:,:].unsqueeze(1)