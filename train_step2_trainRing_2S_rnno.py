import fire
import jax.numpy as jnp
import numpy as np
import qmt
import ring
from ring.utils import dataloader_torch
from torch.utils.data import ConcatDataset
import wandb


class Transform:

    def __init__(self, dof: int | None, rand_ori: bool, pos: bool, use_vqf: bool):
        assert dof in [1, 2, 3, None]
        self.dof = dof
        self.rand_ori = rand_ori
        self.pos = pos
        self.use_vqf = use_vqf

    def __call__(self, ele: list):
        X_d, y_d = ele

        seg1, seg2 = X_d["seg1"], X_d["seg2"]
        a1, a2 = seg1["acc"], seg2["acc"]
        g1, g2 = seg1["gyr"], seg2["gyr"]
        p1, p2 = seg1["imu_to_joint_m"], seg2["imu_to_joint_m"]

        q1 = qmt.randomQuat() if self.rand_ori else np.array([1.0, 0, 0, 0])
        q2 = qmt.randomQuat() if self.rand_ori else np.array([1.0, 0, 0, 0])
        a1, g1, p1 = qmt.rotate(q1, a1), qmt.rotate(q1, g1), qmt.rotate(q1, p1)
        a2, g2, p2 = qmt.rotate(q2, a2), qmt.rotate(q2, g2), qmt.rotate(q2, p2)
        qrel = y_d["seg2"]
        qrel = qmt.qmult(q1, qmt.qmult(qrel, qmt.qinv(q2)))

        F = 12
        if self.dof is not None:
            F += 3
        if self.pos:
            F += 6
        if self.use_vqf:
            F += 8
        dt = X_d.get("dt", None)
        if dt is not None:
            F += 1

        X = np.zeros((a1.shape[0], F))
        grav, pi = 9.81, 3.14
        X[:, 0:3] = a1 / grav
        X[:, 3:6] = a2 / grav
        X[:, 6:9] = g1 / pi
        X[:, 9:12] = g2 / pi

        i = 12
        if self.dof is not None:
            X[:, i + self.dof - 1] = 1.0
            i += 3
        if self.pos:
            X[:, i : (i + 3)] = p1  # noqa: E203
            X[:, (i + 3) : (i + 6)] = p2  # noqa: E203
            i += 6
        if self.use_vqf:
            _dt = 0.01 if dt is None else dt
            X[:, i : (i + 4)] = qmt.oriEstVQF(  # noqa: E203
                g1, a1, params=dict(Ts=float(_dt))
            )
            X[:, (i + 4) : (i + 8)] = qmt.oriEstVQF(  # noqa: E203
                g2, a2, params=dict(Ts=float(_dt))
            )
            i += 8
        if dt is not None:
            X[:, -1] = dt * 10

        return X[:, None], qrel[:, None]


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
    dof: bool = False,
    rand_ori: bool = False,
    tbp: int = 1000,
    pos: bool = False,
    use_vqf: bool = False,
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
                    p, transform(i + 1 if dof else None)
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
            print("T: ", T)

    opt = ring.ml.make_optimizer(
        lr, episodes, int(T / tbp), adap_clip=None, glob_clip=1.0
    )

    ring.ml.train_fn(
        gen,
        episodes,
        net,
        opt,
        callback_kill_after_seconds=23.5 * 3600,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        seed_network=seed,
        callback_save_params=_params(),
        callbacks=callbacks,
        tbp=tbp,
    )


if __name__ == "__main__":
    fire.Fire(main)
