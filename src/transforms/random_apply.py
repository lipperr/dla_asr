import random
from typing import Callable

from torch import Tensor, nn


class RandomApply(nn.Module):
    def __init__(self, p: float, augmentation: Callable):
        super().__init__()
        assert 0 <= p <= 1
        self.p = p
        self.augmentation = augmentation

    def __call__(self, data: Tensor) -> Tensor:
        if random.random() < self.p:
            return self.augmentation(data)
        else:
            return data
