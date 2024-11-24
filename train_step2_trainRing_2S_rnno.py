from diodem import load_data
import fire
import jax.numpy as jnp
import numpy as np
import qmt
import ring
from ring.utils import dataloader_torch
from torch.utils.data import ConcatDataset
import wandb

from train_support.transform_2S import Transform


def _params(unique_id: str = ring.ml.unique_id()) -> str:
    home = "/bigwork/nhkbbach/" if ring.ml.on_cluster() else "~/"
    return home + f"params/{unique_id}.pickle"


def _diodem_cb(exp_id: int, net, seg1, seg2, motion_start):
    data = load_data(exp_id, motion_start=motion_start)

    N = data["seg1"]["quat"].shape[0]
    X = np.zeros((N, 12))
    X[..., :3] = data[seg1]["imu_rigid"]["acc"] / 9.81
    X[..., 3:6] = data[seg2]["imu_rigid"]["acc"] / 9.81
    X[..., 6:9] = data[seg1]["imu_rigid"]["gyr"] / 2.2
    X[..., 9:] = data[seg2]["imu_rigid"]["gyr"] / 2.2
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


def _sum_str(digist: str) -> int:
    return sum(int(char) for char in digist)


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
):
    np.random.seed(seed)

    if use_wandb:
        unique_id = ring.ml.unique_id()
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    transform = lambda dof: Transform(dof, rand_ori, pos, use_vqf)

    gen = dataloader_torch.dataset_to_generator(
        ConcatDataset(
            [
                dataloader_torch.FolderOfPickleFilesDataset(
                    p, transform(_sum_str(p.split("_")[0]) if dof else None)
                )
                for i, p in enumerate(paths.split(","))
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
    ).unwrapped  # get ride of GroundTruthWrapper

    callbacks = []
    for i, p in enumerate(paths.split(",")):
        path = p + "_val"
        ds_val = dataloader_torch.FolderOfPickleFilesDataset(
            path, transform(i + 1 if dof else None)
        )
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

    callbacks.append(_diodem_cb(1, net, "seg1", "seg2", "slow1"))
    callbacks.append(_diodem_cb(1, net, "seg2", "seg3", "slow1"))
    callbacks.append(_diodem_cb(1, net, "seg3", "seg4", "slow1"))
    callbacks.append(_diodem_cb(1, net, "seg1", "seg2", "fast"))
    callbacks.append(_diodem_cb(1, net, "seg2", "seg3", "fast"))
    callbacks.append(_diodem_cb(1, net, "seg3", "seg4", "fast"))

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
