from pathlib import Path

from diodem import load_data
import fire
import jax.numpy as jnp
import numpy as np
import qmt
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


def _diodem_cb(exp_id: int, net, seg1, seg2, motion_start, dof: int | None):
    data = load_data(exp_id, motion_start=motion_start)

    N = data["seg1"]["quat"].shape[0]
    F = 12 if dof is None else 15
    X = np.zeros((N, F))
    X[..., :3] = data[seg1]["imu_rigid"]["acc"] / 9.81
    X[..., 3:6] = data[seg2]["imu_rigid"]["acc"] / 9.81
    X[..., 6:9] = data[seg1]["imu_rigid"]["gyr"] / 2.2
    X[..., 9:] = data[seg2]["imu_rigid"]["gyr"] / 2.2

    if dof is not None:
        X[..., 12 + dof - 1] = 1.0
    X = X[:, None]

    Y = qmt.qmult(qmt.qinv(data[seg1]["quat"]), data[seg2]["quat"])
    Y = Y[:, None]

    return ring.ml.callbacks.EvalXyTrainingLoopCallback(
        net,
        dict(
            mae_deg=lambda q, qhat: jnp.rad2deg(
                jnp.mean(ring.maths.angle_error(q, qhat))
            )
        ),
        X,
        Y,
        None,
        f"real_{exp_id}_{motion_start}_{seg1}_{seg2}",
    )


class TwoSegDiodemDataset(Dataset):
    def __init__(
        self,
        exp_id: int,
        seg1: str,
        seg2: str,
        motion_start: str,
        transform,
        T: float = 60,
        Ts: float = 0.01,
        T_offset: float = 3.0,
    ):
        self.N = int(T / Ts)
        self.T = T
        self.Ts = Ts
        self.T_offset = T_offset
        self.seg1 = seg1
        self.seg2 = seg2
        self.exp_id = exp_id
        data = load_data(exp_id, motion_start, motion_stop=-1, resample_to_hz=1 / Ts)
        N = data["seg1"]["quat"].shape[0]
        self.data = {
            "seg1": {
                "acc": data[seg1]["imu_rigid"]["acc"],
                "gyr": data[seg1]["imu_rigid"]["gyr"],
                "imu_to_joint_m": np.zeros(
                    (
                        N,
                        3,
                    )
                ),
            },
            "seg2": {
                "acc": data[seg2]["imu_rigid"]["acc"],
                "gyr": data[seg2]["imu_rigid"]["gyr"],
                "imu_to_joint_m": np.zeros(
                    (
                        N,
                        3,
                    )
                ),
                "quat": qmt.qmult(qmt.qinv(data[seg1]["quat"]), data[seg2]["quat"]),
            },
        }
        window = int(T_offset / Ts)
        self.ts = list(range(0, N - self.N, window))
        self.transform = transform

    def __getitem__(self, index):
        t = self.ts[index]
        data = tree.map_structure(lambda a: a[t : (t + self.N)].copy(), self.data)
        y_d = {"seg2": data["seg2"].pop("quat")}
        return self.transform((data, y_d))

    def __len__(self):
        return len(self.ts)


def _dof_from_path_str(path: str) -> int:
    digits = str(Path(path).name).split("_")[0]
    return sum(int(char) for char in digits)


def _build_train_dataset_and_sampler(paths: str, transform, T, Ts, W: float = 1.0):
    sim_ds = ConcatDataset(
        [
            dataloader_torch.FolderOfPickleFilesDataset(p, transform)
            for p in paths.split(",")
        ]
    )

    chain1 = ["seg1", "seg2", "seg3", "seg4", "seg5"]
    chain2 = ["seg5"] + chain1[:-1]

    max_exp_id = 11 if ring.ml.on_cluster() else 2
    real_datasets = []
    for exp_id in range(1, max_exp_id + 1):
        chain = chain1 if exp_id < 6 else chain2
        motion_start = 7 if exp_id == 1 else 1
        for seg1, seg2 in zip(chain, chain[1:]):
            ds = TwoSegDiodemDataset(exp_id, seg1, seg2, motion_start, transform, T, Ts)
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
    lstm: bool = False,
    dof: bool = False,
    rand_ori: bool = False,
    tbp: int = 1000,
    pos: bool = False,
    use_vqf: bool = False,
    loggers=[],
    adap_clip=None,
    glob_clip=1.0,
    n_decay_episodes=None,
    W: float = 1.0,
):
    np.random.seed(seed)

    assert not dof

    if use_wandb:
        unique_id = ring.ml.unique_id()
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    transform = Transform(None, rand_ori, pos, use_vqf)

    train_ds, sampler = _build_train_dataset_and_sampler(
        paths, transform, 60.0, 0.01, W
    )
    gen = dataloader_torch.dataset_to_generator(
        train_ds,
        batch_size=bs,
        seed=seed,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=False,
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
    ).unwrapped  # get ride of GroundTruthWrapper

    callbacks = []
    for i, p in enumerate(paths.split(",")):
        path = p + "_val"
        ds_val = dataloader_torch.FolderOfPickleFilesDataset(path, transform)
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

    callbacks.append(_diodem_cb(1, net, "seg1", "seg2", "slow1", 3 if dof else None))
    callbacks.append(_diodem_cb(1, net, "seg2", "seg3", "slow1", 1 if dof else None))
    callbacks.append(_diodem_cb(1, net, "seg3", "seg4", "slow1", 1 if dof else None))
    callbacks.append(_diodem_cb(1, net, "seg1", "seg2", "fast", 3 if dof else None))
    callbacks.append(_diodem_cb(1, net, "seg2", "seg3", "fast", 1 if dof else None))
    callbacks.append(_diodem_cb(1, net, "seg3", "seg4", "fast", 1 if dof else None))

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
