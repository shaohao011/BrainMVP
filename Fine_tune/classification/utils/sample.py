from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class DistributedWeightedRandomSampler(DistributedSampler):
    """Sampler that combines the features of `WeightedRandomSampler` and `DistributedSampler`.

        Reference: https://www.zhihu.com/question/520374320/answer/2540264613
        In `DistributedSampler`, the dataset is assumed to be of constant size and constant element order.

    """
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, dataset: Dataset, weights: Sequence[float], num_samples: int, num_replicas: Optional[int] = None,
        rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, drop_last: bool = False, replacement: bool = True, generator=None
    ) -> None:
        super(DistributedWeightedRandomSampler, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={num_samples}")
        if not isinstance(replacement, bool):
            raise ValueError(f"replacement should be a boolean value, but got replacement={replacement}")
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError(f"weights should be a 1d sequence but given weights have shape {tuple(weights_tensor.shape)}")
        self.replacement = replacement
        self.generator = generator
        self.weights = weights_tensor[self.rank::self.num_replicas]
        self.num_samples = num_samples // self.num_replicas
    
    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor = self.rank + rand_tensor * self.num_replicas
        return iter(rand_tensor.tolist())
    
    def __len__(self):
        return self.num_samples

