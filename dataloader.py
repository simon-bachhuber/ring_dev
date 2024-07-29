import os
from typing import Callable, Optional

from matplotlib.pylab import Generator
import numpy as np
import ring.utils as utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
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

    def __call__(self, idx: int):
        return self[idx]


def pytorch_generator(
    *paths,
    batch_size: int,
    transform: Optional[Callable[[T], T]] = None,
    **kwargs,
):

    ds = _Dataset(*paths, transform=transform)
    _kwargs = dict(shuffle=True)
    _kwargs.update(kwargs)
    dl = DataLoader(
        ds, batch_size=batch_size, collate_fn=tree_utils.tree_batch, **_kwargs
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


def make_generator(*paths, batch_size, transform, **kwargs):
    try:
        return pygrain_generator(
            *paths, batch_size=batch_size, transform=transform, **kwargs
        )
    except ImportError:
        return eager_generator(*paths, batch_size=batch_size, transform=transform)


def eager_generator(
    *paths, batch_size: int, transform: Optional[Callable[[T], T]] = None
):
    from ring import RCMG

    def _transform(element):
        return transform(element, np.random.default_rng())

    ds = _Dataset(*paths, transform=_transform)
    data = [ds[i] for i in tqdm.tqdm(range(len(ds)), total=len(ds))]
    return RCMG.eager_gen_from_list(data, batch_size)


def pygrain_generator(
    *paths, batch_size: int, transform=None, worker_count: int = 0, seed: int = 1
):

    import grain.python as pygrain

    class _Transform(pygrain.RandomMapTransform):
        def random_map(self, element, rng: Generator):
            return transform(element, rng)

    ds = _Dataset(*paths, transform=None)
    dl = pygrain.load(
        ds,
        batch_size=batch_size,
        worker_count=worker_count,
        shuffle=True,
        seed=seed,
        transformations=[_Transform()],
    )
    iter_dl = iter(dl)

    def generator(_):
        return next(iter_dl)

    return generator
