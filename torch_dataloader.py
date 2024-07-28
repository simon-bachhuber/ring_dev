import concurrent.futures
from typing import Callable, Optional

import numpy as np
from ring import utils
import tree_utils

T = tree_utils.PyTree[np.ndarray]


def pytorch_generator(
    data: list[T],
    batchsize: int,
    shuffle: bool = True,
    transform: Optional[Callable[[T], T]] = None,
    use_futures: bool = False,
    future_max_workers: int | None = None,
):
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset

    class _Dataset(Dataset):
        def __len__(self):
            return len(data)

        def __getitem__(self, idx: int):
            element = utils.pytree_deepcopy(data[idx])
            if transform is not None and not use_futures:
                element = transform(element)
            return element

    def collate_fn(data: list[T]) -> T:
        if transform is not None and use_futures:
            with concurrent.futures.ProcessPoolExecutor(future_max_workers) as exec:
                data = list(exec.map(transform, data))
        return tree_utils.tree_batch(data)

    dl = DataLoader(
        _Dataset(),
        batch_size=batchsize,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    dl_iter = iter(dl)

    def generator(_):
        nonlocal dl, dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            return next(dl_iter)

    return generator
