import copy
from pathlib import Path

from diodem import load_data
from diodem.benchmark import IMTP
import fire
import jax.numpy as jnp
import numpy as np
import ring
from ring.utils import dataloader_torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
import tree
import wandb

from train_support.transform_2S import Transform


def _params(unique_id: str = ring.ml.unique_id()) -> str:
    home = "/bigwork/nhkbbach/" if ring.ml.on_cluster() else "~/"
    return home + f"params/{unique_id}.pickle"


class TwoSegDiodemDataset(Dataset):
    def __init__(
        self,
        exp_id: int,
        seg1: str,
        seg2: str,
        motion_start: str,
        transform,
        dof: bool,
        # set to -1 to use the entire length
        T: float = 60,
        Ts: float = 0.01,
        T_offset: float = 3.0,
        motion_stop=-1,
    ):
        self.seg1 = seg1
        self.seg2 = seg2
        self.exp_id = exp_id
        data = load_data(
            exp_id, motion_start, motion_stop=motion_stop, resample_to_hz=1 / Ts
        )
        N = data["seg1"]["quat"].shape[0]
        self.N = int(T / Ts) if T != -1 else (N - 1)
        self.data = {
            f"seg{i + 1}": {
                "acc": data[seg]["imu_rigid"]["acc"],
                "gyr": data[seg]["imu_rigid"]["gyr"],
                "quat": data[seg]["quat"],
            }
            for i, seg in enumerate([seg1, seg2])
        }
        window = int(T_offset / Ts)
        self.ts = list(range(0, N - self.N, window))
        self._dof = IMTP([seg1, seg2]).dof(exp_id)[seg2] if dof else None
        self.transform = transform
        self.cb_identifier = f"real_{exp_id}_{motion_start}_{seg1}_{seg2}"

    def __getitem__(self, index):
        data = self._get_ele(index)
        self.transform.diodem()
        return self.transform(data)

    def _get_ele(self, index):
        t = self.ts[index]
        data = tree.map_structure(
            lambda a: a[t : (t + self.N)].copy(), self.data  # noqa: E203
        )
        self.transform.setDOF(self._dof)
        return data

    def __len__(self):
        return len(self.ts)

    def to_cb(self, net):
        if self._dof is not None:
            print(f"DOF for {self.seg1}-{self.seg2} is: ", self._dof)

        assert len(self) == 1
        data = self._get_ele(0)
        self.transform.diodem(cb=True)
        X, Y = self.transform(data)
        return ring.ml.callbacks.EvalXyTrainingLoopCallback(
            net,
            dict(
                mae_deg=lambda q, qhat: jnp.rad2deg(
                    jnp.mean(ring.maths.angle_error(q, qhat)[..., 500:])
                )
            ),
            X,
            Y,
            None,
            self.cb_identifier,
        )


def _dof_from_path_str(path: str) -> int:
    digits = str(Path(path).name).split("_")[0]
    return sum(int(char) for char in digits if char.isdigit())


def _load_sim_ds_from_path(path: str, transform, dof: bool) -> Dataset:
    trafo = copy.deepcopy(transform)
    trafo.sim()
    trafo.setDOF(_dof_from_path_str(path) if dof else None)
    return dataloader_torch.FolderOfPickleFilesDataset(path, trafo)


def _build_train_dataset_and_sampler(
    paths: str, transform, dof: bool, T, Ts, W: float = 1.0
):
    sim_ds = ConcatDataset(
        [_load_sim_ds_from_path(p, transform, dof) for p in paths.split(",")]
    )

    chain1 = ["seg1", "seg2", "seg3", "seg4", "seg5"]
    chain2 = ["seg5"] + chain1[:-1]

    max_exp_id = 11 if ring.ml.on_cluster() else 2
    real_datasets = []
    for exp_id in range(1, max_exp_id + 1):
        chain = chain1 if exp_id < 6 else chain2
        motion_start = 7 if exp_id == 1 else 1
        for seg1, seg2 in zip(chain, chain[1:]):
            ds = TwoSegDiodemDataset(
                exp_id, seg1, seg2, motion_start, transform, dof, T, Ts
            )
            real_datasets.append(ds)
    real_ds = ConcatDataset(real_datasets)

    print("Number of simulated datapoints: ", len(sim_ds))
    print("Number of real-world datapoints: ", len(real_ds))

    weights = [1.0] * len(sim_ds) + [len(sim_ds) * W / len(real_ds)] * len(real_ds)
    ds = ConcatDataset([sim_ds, real_ds])
    return ds, WeightedRandomSampler(weights, len(ds))


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
    dof: bool = False,
    rand_ori: bool = False,
    tbp: int = 1000,
    loggers=[],
    adap_clip=None,
    glob_clip=1.0,
    n_decay_episodes=None,
    W: float = 1.0,
    lpf_cutoff: float = None,
    rnno: bool = False,
    layernorm: bool = True,
    celltype: str = "gru",
    rel_only: bool = False,
):
    np.random.seed(seed)

    if use_wandb:
        unique_id = ring.ml.unique_id()
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    transform = Transform(rand_ori, hz=100.0, cutoff=lpf_cutoff, rel_only=rel_only)

    train_ds, sampler = _build_train_dataset_and_sampler(
        paths, transform, dof, 60.0, 0.01, W
    )
    gen = dataloader_torch.dataset_to_generator(
        train_ds,
        batch_size=bs,
        seed=seed,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
    )

    params = _params(hex(warmstart)) if warmstart else None

    if rnno:
        kwargs = {
            "forward_factory": ring.ml.rnno_v1.rnno_v1_forward_factory,
            "rnn_layers": [rnn_w] * rnn_d,
            "linear_layers": [lin_w] * lin_d,
            "act_fn_rnn": lambda X: X,
            "output_dim": 8,
        }
    else:
        kwargs = {
            "hidden_state_dim": rnn_w,
            "stack_rnn_cells": rnn_d,
            "message_dim": lin_w,
            "send_message_n_layers": lin_d,
        }

    net = ring.ml.RING(
        params=params, celltype=celltype, lam=(-1, 0), layernorm=layernorm, **kwargs
    )
    if rnno:
        net = ring.ml.base.NoGraph_FilterWrapper(net, quat_normalize=True)

    callbacks = []
    for i, p in enumerate(paths.split(",")):
        path = p + "_val"
        ds_val = _load_sim_ds_from_path(path, transform, dof)
        X_val, y_val = dataloader_torch.dataset_to_generator(ds_val, len(ds_val))(None)
        callbacks.append(
            ring.ml.callbacks.EvalXyTrainingLoopCallback(
                net,
                dict(
                    mae_deg=lambda q, qhat: jnp.rad2deg(
                        jnp.mean(ring.maths.angle_error(q, qhat))
                    )
                ),
                X_val,
                y_val,
                None,
                path.split("/")[-1] + "_",
            )
        )
        if i == 0:
            T = X_val.shape[1]
            # print("T: ", T)

    track_metrices = []

    def append_diodem_eval_callback(seg1, seg2, motion_start, track=False) -> None:
        cb = TwoSegDiodemDataset(
            1, seg1, seg2, motion_start, transform, dof, T=-1, motion_stop=None
        ).to_cb(net)
        callbacks.append(cb)
        if track:
            track_metrices.append([cb.metric_identifier, "mae_deg", "link1"])

    for motion in ["slow1", "fast"]:
        append_diodem_eval_callback("seg1", "seg2", motion, track=True)
        append_diodem_eval_callback("seg2", "seg3", motion)
        append_diodem_eval_callback("seg3", "seg4", motion)

    callbacks.append(
        ring.ml.callbacks.AverageMetricesTLCB(
            track_metrices,
            "real_mae_deg_link1",
        )
    )

    n_decay_episodes = episodes if n_decay_episodes is None else n_decay_episodes
    opt = ring.ml.make_optimizer(
        lr, n_decay_episodes, int(T / tbp), adap_clip=adap_clip, glob_clip=glob_clip
    )

    ring.ml.train_fn(
        gen,
        episodes,
        net,
        opt,
        # callback_kill_after_seconds=23.5 * 3600,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        seed_network=seed,
        callback_save_params=_params(),
        callbacks=callbacks,
        tbp=tbp,
        loggers=loggers,
    )


if __name__ == "__main__":
    fire.Fire(main)
