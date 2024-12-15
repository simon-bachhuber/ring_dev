from dataclasses import replace
import os
import random

from diodem.benchmark import benchmark
from diodem.benchmark import IMTP
import fire
import jax.numpy as jnp
import numpy as np
import qmt
import ring
from ring import ml
from ring.utils import dataloader_torch
from torch.utils.data import Dataset
import wandb

lam = [-1, -1, 1, -1, 3, 4, -1, 6, 7, 8]

link_names = [
    "seg3_1Seg",
    "seg3_2Seg",
    "seg4_2Seg",
    "seg3_3Seg",
    "seg4_3Seg",
    "seg5_3Seg",
    "seg2_4Seg",
    "seg3_4Seg",
    "seg4_4Seg",
    "seg5_4Seg",
]

dropout_rates_hard = dict(
    seg4_2Seg=dict(imu=0.0, ja_1d=2 / 3, ja_2d=1 / 2, dof=1 / 2),
    seg4_3Seg=dict(imu=0.5, ja_1d=2 / 3, ja_2d=1 / 2, dof=1 / 2),
    seg5_3Seg=dict(imu=0.0, ja_1d=2 / 3, ja_2d=1 / 2, dof=1 / 2),
    seg3_4Seg=dict(imu=0.5, ja_1d=2 / 3, ja_2d=1 / 2, dof=1 / 2),
    seg4_4Seg=dict(imu=0.5, ja_1d=2 / 3, ja_2d=1 / 2, dof=1 / 2),
    seg5_4Seg=dict(imu=0.0, ja_1d=2 / 3, ja_2d=1 / 2, dof=1 / 2),
)

dropout_rates_easy = dict(
    seg4_2Seg=dict(imu=0.0, ja_1d=1 / 3, ja_2d=1 / 3, dof=0.0),
    seg4_3Seg=dict(imu=1 / 3, ja_1d=1 / 3, ja_2d=1 / 3, dof=0.0),
    seg5_3Seg=dict(imu=0.0, ja_1d=1 / 3, ja_2d=1 / 3, dof=0.0),
    seg3_4Seg=dict(imu=1 / 3, ja_1d=1 / 3, ja_2d=1 / 3, dof=0.0),
    seg4_4Seg=dict(imu=1 / 3, ja_1d=1 / 3, ja_2d=1 / 3, dof=0.0),
    seg5_4Seg=dict(imu=0.0, ja_1d=1 / 3, ja_2d=1 / 3, dof=0.0),
)


def _params(unique_id: str = ring.ml.unique_id()) -> str:
    home = "/bigwork/nhkbbach/" if ring.ml.on_cluster() else "~/"
    return home + f"params/{unique_id}.pickle"


def _make_ring(lam, warmstart):
    dry_run = not ring.ml.on_cluster()
    message_dim = 400 if not dry_run else 10
    hidden_state_dim = 600 if not dry_run else 20
    params = _params(hex(warmstart)) if warmstart else None
    ringnet = ml.RING(
        lam=lam,
        message_dim=message_dim,
        hidden_state_dim=hidden_state_dim,
        params=params,
        send_message_n_layers=0,
        stack_rnn_cells=2,
    )
    return ringnet


def _loss_fn(q, qhat):
    "(T, N, F) -> Scalar"
    loss = jnp.array(0.0)
    for i, p in enumerate(lam):
        if p == -1:
            loss += jnp.mean(ring.maths.inclination_loss(q[:, i], qhat[:, i]) ** 2)
        else:
            loss += jnp.mean(ring.maths.angle_error(q[:, i], qhat[:, i]) ** 2)
    return loss / len(lam)


class Transform:
    def __init__(self, imtp: IMTP, dropout_rates: dict):
        self.imtp = imtp
        self.dropout_rates = dropout_rates

    def __call__(self, lam2_1, lam2_2, lam3, lam4):
        dropout_rates = self.dropout_rates
        imtp = self.imtp

        X1, Y1 = lam2_1
        X2, Y2 = lam2_2
        X3, Y3 = lam3
        X4, Y4 = lam4

        X1["seg3_1Seg"] = X1.pop("seg3_2Seg")
        X1.pop("seg4_2Seg")
        Y1["seg3_1Seg"] = Y1.pop("seg3_2Seg")
        Y1.pop("seg4_2Seg")

        dt1 = X1.pop("dt")
        dt2 = X2.pop("dt")
        dt3 = X3.pop("dt")
        dt4 = X4.pop("dt")

        X1.update(X2)
        X1.update(X3)
        X1.update(X4)
        Y1.update(Y2)
        Y1.update(Y3)
        Y1.update(Y4)

        T = Y1["seg3_1Seg"].shape[0]
        X = np.zeros((imtp.F, 10, T))
        Y = np.zeros((10, T, 4))

        if imtp.dt:
            X[imtp.slice("dt"), 0] = dt1 / imtp.scale_dt
            X[imtp.slice("dt"), 1:3] = dt2 / imtp.scale_dt
            X[imtp.slice("dt"), 3:6] = dt3 / imtp.scale_dt
            X[imtp.slice("dt"), 6:] = dt4 / imtp.scale_dt

        draw = lambda p: 1.0 - np.random.binomial(1, p=p)

        for i, (name, p) in enumerate(zip(link_names, lam)):
            factor = 1.0
            if imtp.sparse and (p != -1):
                factor = draw(dropout_rates[name]["imu"])
            X[imtp.slice("acc"), i] = (X1[name]["acc"].T / imtp.scale_acc) * factor
            X[imtp.slice("gyr"), i] = (X1[name]["gyr"].T / imtp.scale_gyr) * factor
            if imtp.mag:
                X[imtp.slice("mag"), i] = (X1[name]["mag"].T / imtp.scale_mag) * factor
            if p != -1:
                if imtp.joint_axes_1d and X1[name]["dof"] == 1:
                    X[imtp.slice("ja_1d"), i] = X1[name]["joint_params"]["rr"][
                        "joint_axes"
                    ][:, None] * draw(dropout_rates[name]["ja_1d"])
                if imtp.joint_axes_2d and X1[name]["dof"] == 2:
                    X[imtp.slice("ja_2d"), i] = X1[name]["joint_params"]["rsaddle"][
                        "joint_axes"
                    ].reshape(6, 1) * draw(dropout_rates[name]["ja_2d"])
                if imtp.dof:
                    dof_array = np.zeros((3,))
                    dof_array[X1[name]["dof"] - 1] = 1.0 * draw(
                        dropout_rates[name]["dof"]
                    )
                    X[imtp.slice("dof"), i] = dof_array[:, None]

            q_p = np.array([1.0, 0, 0, 0]) if p == -1 else Y1[link_names[p]]
            q_i = Y1[name]
            Y[i] = qmt.qrel(q_p, q_i)

        return X.transpose((2, 1, 0)), Y.transpose((1, 0, 2))


def _make_exp_callbacks(ringnet, imtp: IMTP):

    callbacks, metrices_name = [], []

    def add_callback(segments: list[str], exp_id, motion_start):
        cb = benchmark(
            imtp=replace(imtp, segments=segments),
            exp_id=exp_id,
            motion_start=motion_start,
            filter=ringnet,
            return_cb=True,
        )
        callbacks.append(cb)
        # exclude the first element because it connects to -1 and thus its `mae_deg`
        # loss will be not meaningful
        for segment in segments[1:]:
            metrices_name.append([cb.metric_identifier, "mae_deg", segment])

    add_callback(["seg3"], 2, "slow_fast_mix")
    add_callback(["seg1", "seg2"], 6, "slow1")
    add_callback(["seg2", "seg3"], 6, "fast")
    add_callback(["seg3", "seg4", "seg5"], 1, "slow1")
    add_callback(["seg3", "seg4", "seg5"], 1, "fast")
    add_callback(["seg2", "seg3", "seg4", "seg5"], 1, "slow1")
    add_callback(["seg2", "seg3", "seg4", "seg5"], 1, "fast")

    for zoom_in in metrices_name:
        print(zoom_in)
    callbacks += [ml.callbacks.AverageMetricesTLCB(metrices_name, "exp_val_mae_deg")]

    return callbacks


def main(
    path_lam2,
    path_lam3,
    path_lam4,
    bs: int,
    episodes: int,
    use_wandb: bool = False,
    wandb_project: str = "RING",
    wandb_name: str = None,
    warmstart: str = None,
    seed: int = 1,
    exp_cbs: bool = False,
    lr: float = 1e-3,
    tbp: int = 1000,
    dropout_rates: str = "hard",
):
    np.random.seed(seed)

    if use_wandb:
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    ringnet = _make_ring(lam, warmstart)
    imtp = IMTP(
        ["seg1"],
        sparse=True,
        joint_axes_1d=True,
        joint_axes_1d_field=True,
        joint_axes_2d=True,
        joint_axes_2d_field=True,
        dof=True,
        dof_field=True,
        dt=True,
        scale_acc=9.81,
        scale_gyr=2.2,
        scale_dt=0.1,
    )
    ds_train = MultiDataset(
        [
            dataloader_torch.FolderOfPickleFilesDataset(p)
            for p in [path_lam2, path_lam2, path_lam3, path_lam4]
        ],
        Transform(
            imtp, dict(easy=dropout_rates_easy, hard=dropout_rates_hard)[dropout_rates]
        ),
    )
    generator = dataloader_torch.dataset_to_generator(
        ds_train,
        batch_size=bs,
        seed=seed,
        drop_last=True,
        num_workers=os.cpu_count() if ring.ml.on_cluster() else 1,
    )

    callbacks = _make_exp_callbacks(ringnet, imtp) if exp_cbs else []

    n_decay_episodes = 9500
    optimizer = ring.ml.make_optimizer(
        lr, n_decay_episodes, int(6000 / tbp), adap_clip=0.5, glob_clip=None
    )

    ml.train_fn(
        generator,
        episodes,
        ringnet,
        optimizer=optimizer,
        callbacks=callbacks,
        callback_kill_after_seconds=23.5 * 3600,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        callback_save_params=_params(),
        callback_save_params_track_metrices=[["exp_val_mae_deg"]] if exp_cbs else None,
        seed_network=seed,
        link_names=link_names,
        tbp=tbp,
        loss_fn=_loss_fn,
    )


class MultiDataset(Dataset):
    def __init__(self, datasets, transform=None):
        """
        Args:
            datasets: A list of datasets to sample from.
            transform: A function that takes N items (one from each dataset) and combines them.
        """  # noqa: E501
        self.datasets = datasets
        self.transform = transform

    def __len__(self):
        # Length is defined by the smallest dataset in the list
        return min(len(ds) for ds in self.datasets)

    def __getitem__(self, idx):
        # Randomly sample one element from each dataset
        sampled_items = [ds[random.randint(0, len(ds) - 1)] for ds in self.datasets]

        if self.transform:
            # Apply the transformation to all sampled items
            return self.transform(*sampled_items)

        return tuple(sampled_items)


if __name__ == "__main__":
    fire.Fire(main)
