import os
from pathlib import Path
import pickle

import fire
import h5py
import numpy as np
from torch.utils.data import Dataset
import tqdm


def load_array_pickle(path):
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def save_array_h5(filename, array, dataset_name="dataset", compression="gzip"):
    """
    Saves a NumPy array to an HDF5 file with compression.

    Parameters:
    filename (str): Name of the HDF5 file to save the array.
    array (np.ndarray): The NumPy array to save.
    dataset_name (str): Name of the dataset in the HDF5 file.
    compression (str or int): Compression strategy (e.g., 'gzip', 'lzf') or
        level (e.g., 1-9 for gzip).
    """
    with h5py.File(filename, "w") as h5file:
        h5file.create_dataset(dataset_name, data=array, compression=compression)


def load_array_h5(filename, dataset_name="dataset"):
    with h5py.File(filename, "r") as h5file:
        return h5file[dataset_name][:]


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


def _flatten_transform(ele):
    X_d, y_d = ele
    seg1, seg2 = X_d["seg1"], X_d["seg2"]
    a1, a2 = seg1["acc"], seg2["acc"]
    g1, g2 = seg1["gyr"], seg2["gyr"]
    p1, p2 = seg1["imu_to_joint_m"], seg2["imu_to_joint_m"]

    qrel = y_d["seg2"]

    F = 22
    X = np.zeros((a1.shape[0], F))
    X[:, 0:3] = a1
    X[:, 3:6] = a2
    X[:, 6:9] = g1
    X[:, 9:12] = g2
    X[:, 12:15] = p1
    X[:, 15:18] = p2
    X[:, 18:22] = qrel

    return X.astype(np.float32)


def main(path_folder_in, path_folder_out, filename_prefix: str = "seq"):
    ds = FolderOfFiles(path_folder_in, _flatten_transform)

    folder_out = Path(path_folder_out)
    folder_out.mkdir(parents=True, exist_ok=True)

    for i in tqdm.tqdm(range(len(ds))):
        save_array_h5(folder_out.joinpath(filename_prefix + str(i) + ".h5"), ds[i])


if __name__ == "__main__":
    fire.Fire(main)
