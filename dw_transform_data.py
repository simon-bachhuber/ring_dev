import os
from pathlib import Path
import pickle

import fire
import h5py
import jax
import numpy as np
from torch.utils.data import Dataset
import tqdm


def load_array_pickle(path):
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def save_array_dict(filename, array_dict, compression="gzip"):
    """
    Saves a dictionary of NumPy arrays to an HDF5 file with compression.

    Parameters:
    filename (str): Name of the HDF5 file to save the arrays.
    array_dict (dict): A dictionary where keys are dataset names
        and values are NumPy arrays.
    compression (str or int): Compression strategy (e.g., 'gzip', 'lzf')
        or level (e.g., 1-9 for gzip).
    """
    with h5py.File(filename, "w") as h5file:
        for key, array in array_dict.items():
            h5file.create_dataset(key, data=array, compression=compression)


def load_array_dict(filename):
    """
    Loads a dictionary of NumPy arrays from an HDF5 file.

    Parameters:
    filename (str): Name of the HDF5 file to load the arrays from.

    Returns:
    dict: A dictionary where keys are dataset names and values are loaded NumPy arrays.
    """
    array_dict = {}
    with h5py.File(filename, "r") as h5file:
        for key in h5file.keys():
            array_dict[key] = h5file[key][:]
    return array_dict


class FolderOfFiles(Dataset):
    def __init__(self, path, transform=None, loader=load_array_pickle):
        self.files = self.listdir(path)
        self.transform = transform
        self.N = len(self.files)
        self.loader = loader

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        element = self.loader(self.files[idx])
        if self.transform is not None:
            element = self.transform(element)
        return element

    @staticmethod
    def listdir(path: str) -> list[Path]:
        return [FolderOfFiles.parse_path(path, file) for file in os.listdir(path)]

    @staticmethod
    def parse_path(path: str, *other_paths, mkdir=True) -> Path:
        path = Path(os.path.expanduser(path))

        for p in other_paths:
            path = path.joinpath(p)

        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)

        return path


def _to_dict(key, X) -> dict:
    xyz = "xyz" if X.shape[-1] == 3 else "uxyz"
    return {key + "_" + x: X[..., i] for i, x in enumerate(xyz)}


def _flatten_transform(ele):
    X_d, y_d = ele
    T = y_d["seg2"].shape[0]
    for i in [1, 2]:
        X_d[f"seg{i}"]["imu_to_joint"] = np.repeat(
            X_d[f"seg{i}"]["imu_to_joint_m"][None], T, axis=0
        )

    X = dict()
    for i in [1, 2]:
        for f in ["acc", "gyr", "imu_to_joint"]:
            X.update(_to_dict(f + str(i), X_d[f"seg{i}"][f]))
        X.update(_to_dict("quat" + str(i), y_d[f"seg{i}"]))
    return jax.tree.map(lambda a: a.astype(np.float32), X)


def main(
    path_folder_in,
    path_folder_out,
    filename_prefix: str = "seq",
    compression: bool = False,
):
    ds = FolderOfFiles(path_folder_in, _flatten_transform)

    folder_out = Path(path_folder_out)
    folder_out.mkdir(parents=True, exist_ok=True)

    for i in tqdm.tqdm(range(len(ds))):
        save_array_dict(
            folder_out.joinpath(filename_prefix + str(i) + ".h5"),
            ds[i],
            compression="gzip" if compression else None,
        )


if __name__ == "__main__":
    fire.Fire(main)
