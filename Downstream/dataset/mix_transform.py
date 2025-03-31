from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
import numpy as np
from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from typing import Dict, Hashable, Mapping


class RandomChannelCutmix(Transform):
    """
    Random substitute N channels with template
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def random_sample_index(self, num_sample = 0):
        max_num = self.max_num_channel
        sample_list = []
        num_sample = np.random.randint(0, num_sample+1)
        num_sample = (max_num - 1) if num_sample > (max_num - 1) else num_sample  
        for _ in range(num_sample):
            sidx = np.random.randint(0, max_num)
            while sidx in sample_list:
                sidx = np.random.randint(0, max_num)
            sample_list.append(sidx)
        return sample_list
    
    def __init__(self, num_mix: int, pair_aug=False, max_num_channel: int=4) -> None:
        self.num_mix = num_mix
        self.pair_aug = pair_aug
        self.max_num_channel = max_num_channel

    def __call__(self, d) -> NdarrayOrTensor:

        image = d['image']
        mask = np.zeros_like(image) if d.get('template') is None else d['template'] 
        sidx_list = self.random_sample_index(num_sample=self.num_mix)
        if self.pair_aug:
            n_channel = image.shape[0]
            image = np.repeat(image, 2, axis=0)
            mask = np.repeat(mask, 2, axis=0)
            sidx_list = sidx_list + [x + n_channel for x in self.random_sample_index(num_sample=self.num_mix)]
        image[sidx_list, :, :, :] = mask[sidx_list,:, :, :]
        return image


class RandomChannelCutmixd(RandomChannelCutmix, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandScaleIntensity`.
    """

    backend = RandomChannelCutmix.backend

    def __init__(
        self,
        keys: KeysCollection,
        num_mix: int = 0,
        pair_aug: bool = False,
        max_num_channel: int = 4,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.scaler = RandomChannelCutmix(num_mix, pair_aug, max_num_channel)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        d['image'] = self.scaler(d)
        return d