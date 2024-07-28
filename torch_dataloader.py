import os
from typing import Callable, Optional

import numpy as np
from ring import utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tree_utils

T = tree_utils.PyTree[np.ndarray]


class _Dataset(Dataset):
    def __init__(self, *paths, transform):

        self.files = [self.listdir(path) for path in paths]
        Ns = set([len(f) for f in self.files])
        assert len(Ns) == 1, f"{Ns}"

        self.P = len(self.files)
        self.N = list(Ns)[0]
        self.transform = transform

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        element = [utils.pickle_load(self.files[p][idx]) for p in range(self.P)]
        if self.transform is not None:
            element = self.transform(element)
        return element

    @staticmethod
    def listdir(path: str) -> list:
        return [utils.parse_path(path, file) for file in os.listdir(path)]


def pytorch_generator(
    *paths,
    batch_size: int,
    transform: Optional[Callable[[T], T]] = None,
    **kwargs,
):

    def collate_fn(data: list[T]) -> T:
        return tree_utils.tree_batch(data)

    ds = _Dataset(*paths, transform=transform)
    _kwargs = dict(shuffle=True)
    _kwargs.update(kwargs)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, **_kwargs)
    dl_iter = iter(dl)

    def generator(_):
        nonlocal dl, dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            return next(dl_iter)

    return generator
