import fire
import numpy as np
import ring
from ring.utils import dataloader_torch
from torch.utils.data import ConcatDataset
import wandb


class Transform:

    def __call__(self, ele: list):
        X_d, y_d = ele

        seg1, seg2 = X_d["seg1"], X_d["seg2"]
        a1, a2 = seg1["acc"] / 9.81, seg2["acc"] / 9.81
        g1, g2 = seg1["gyr"] / 3.14, seg2["gyr"] / 3.14

        T = a1.shape[0]
        F = 13

        X = np.zeros((T, F))
        X[:, 0:3] = a1
        X[:, 3:6] = a2
        X[:, 6:9] = g1
        X[:, 9:12] = g2
        X[:, 12] = X_d["dt"] * 10

        return X[:, None], y_d["seg2"][:, None]


def _params(unique_id: str = ring.ml.unique_id()) -> str:
    home = "/home/woody/iwb3/iwb3004h/simon/" if ring.ml.on_cluster() else "~/"
    return home + f"params/{unique_id}.pickle"


def main(
    paths: str,
    bs: int,
    episodes: int,
    rnn_w: int = 400,
    rnn_d: int = 2,
    lin_w: int = 200,
    lin_d: int = 2,
    seed: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "RING_2D",
    wandb_name: str = None,
    lr: float = 1e-3,
    num_workers: int = 1,
    warmstart: str = None,
    lstm: bool = False,
):
    np.random.seed(seed)

    if use_wandb:
        unique_id = ring.ml.unique_id()
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    gen = dataloader_torch.dataset_to_generator(
        ConcatDataset(
            [
                dataloader_torch.FolderOfPickleFilesDataset(p, Transform())
                for p in paths.split(",")
            ]
        ),
        batch_size=bs,
        seed=seed,
        num_workers=num_workers,
    )

    params = _params(hex(warmstart)) if warmstart else None
    celltype = "lstm" if lstm else "gru"

    net = ring.ml.RNNO(
        4,
        return_quats=True,
        eval=False,
        v1=True,
        rnn_layers=[rnn_w] * rnn_d,
        linear_layers=[lin_w] * lin_d,
        act_fn_rnn=lambda X: X,
        params=params,
        celltype=celltype,
        scale_X=False,
    )

    ring.ml.train_fn(
        gen,
        episodes,
        net,
        ring.ml.make_optimizer(
            lr,
            episodes,
            adap_clip=None,
            glob_clip=1.0,
        ),
        callback_kill_after_seconds=23.5 * 3600,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        seed_network=seed,
        callback_save_params=_params(),
    )


if __name__ == "__main__":
    fire.Fire(main)
