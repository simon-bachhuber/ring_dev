from dataclasses import replace
import os

from diodem.benchmark import benchmark
from diodem.benchmark import IMTP
import fire
import jax.numpy as jnp
import numpy as np
import qmt
import ring
from ring import maths
from ring import ml
from ring.utils.dataloader_torch import dataset_to_generator
from ring.utils.dataloader_torch import dataset_to_Xy
from ring.utils.dataloader_torch import FolderOfFilesDataset
from ring.utils.dataloader_torch import MultiDataset
from ring.utils.dataloader_torch import ShuffledDataset
from torch.utils.data import random_split
import wandb


def _home() -> str:
    home = f"/bigwork/{os.environ.get('USER')}/" if ring.ml.on_cluster() else "~/"
    return os.environ.get("RING_HOME", home)


def _params(unique_id: str = ring.ml.unique_id()) -> str:
    return _home() + f"params/{unique_id}.pickle"


class Transform:
    chain = ["seg2_4Seg", "seg3_4Seg", "seg4_4Seg", "seg5_4Seg"]
    inner = ["seg4_3Seg", "seg3_4Seg", "seg4_4Seg"]
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

    def __init__(
        self,
        imtp: IMTP,
        drop_imu_1d,
        drop_imu_2d,
        drop_imu_3d,
        drop_ja_1d,
        drop_ja_2d,
        drop_dof,
        delay: int,
    ):
        self.imtp = imtp
        self.drop_imu = {1: drop_imu_1d, 2: drop_imu_2d, 3: drop_imu_3d}
        self.drop_ja_1d = drop_ja_1d
        self.drop_ja_2d = drop_ja_2d
        self.drop_dof = drop_dof
        self.delay = delay

    def _lamX_from_lam4(self, lam4, rename_to: list[str]):
        N = len(rename_to)
        start = np.random.choice(list(range((5 - N))))
        rename_from = self.chain[start : (start + N)]  # noqa: E203
        X, y = lam4
        for old_name, new_name in zip(rename_from, rename_to):
            X[new_name] = X[old_name]
            y[new_name] = y[old_name]
        for old_name in self.chain:
            X.pop(old_name)
            y.pop(old_name)
        return X, y

    def __call__(self, lam41, lam42, lam43, lam44):
        imtp = self.imtp
        slices = imtp.getSlices()
        lam = self.lam
        link_names = self.link_names

        X1, Y1 = self._lamX_from_lam4(lam41, ["seg3_1Seg"])
        X2, Y2 = self._lamX_from_lam4(lam42, ["seg3_2Seg", "seg4_2Seg"])
        X3, Y3 = self._lamX_from_lam4(lam43, ["seg3_3Seg", "seg4_3Seg", "seg5_3Seg"])
        X4, Y4 = lam44

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

        del X2, X3, X4, Y2, Y3, Y4

        T = Y1["seg3_1Seg"].shape[0]
        # X.shape = (F, N, T)
        # Y.shape = (N, T, 4)
        X = np.zeros((imtp.getF(), 10, T))
        Y = np.zeros((10, T, 4))

        if imtp.dt:
            X[slices["dt"], 0] = dt1 / imtp.scale_dt
            X[slices["dt"], 1:3] = dt2 / imtp.scale_dt
            X[slices["dt"], 3:6] = dt3 / imtp.scale_dt
            X[slices["dt"], 6:] = dt4 / imtp.scale_dt

        draw = lambda p: 1.0 - np.random.binomial(1, p=p)

        for i, (name, p) in enumerate(zip(link_names, lam)):

            imu_factor = 1.0
            if imtp.sparse and name in self.inner:
                dof = int(X1[name]["dof"])
                imu_factor = draw(self.drop_imu[dof])

            X[slices["acc"], i] = (X1[name]["acc"].T / imtp.scale_acc) * imu_factor
            X[slices["gyr"], i] = (X1[name]["gyr"].T / imtp.scale_gyr) * imu_factor
            if imtp.mag:
                X[slices["mag"], i] = (X1[name]["mag"].T / imtp.scale_mag) * imu_factor

            if p != -1:
                dof = int(X1[name]["dof"])
                if imtp.joint_axes_1d and dof == 1:
                    X[slices["ja_1d"], i] = (
                        X1[name]["joint_params"]["rr"]["joint_axes"][:, None]
                        / imtp.scale_ja
                        * draw(self.drop_ja_1d)
                    )
                if imtp.joint_axes_2d and dof == 2:
                    X[slices["ja_2d"], i] = (
                        X1[name]["joint_params"]["rsaddle"]["joint_axes"].reshape(6, 1)
                        / imtp.scale_ja
                        * draw(self.drop_ja_2d)
                    )
                if imtp.dof:
                    dof_array = np.zeros((3,))
                    dof_array[dof - 1] = 1.0 * draw(self.drop_dof)
                    X[slices["dof"], i] = dof_array[:, None]

            q_p = np.array([1.0, 0, 0, 0]) if p == -1 else Y1[link_names[p]]
            q_i = Y1[name]
            Y[i] = qmt.qrel(q_p, q_i)

        # X.shape = (T, N, F)
        # Y.shape = (T, N, 4)
        X, Y = X.transpose((2, 1, 0)), Y.transpose((1, 0, 2))

        if self.delay > 0:
            _Y = Y.copy()
            # for the first `delay` timesteps, Y stays constant at the initial pose,
            # until the delay is reached
            Y[: self.delay] = _Y[0]
            Y[self.delay :] = _Y[0 : -self.delay]  # noqa: E203

        return X, Y


def _cb_metrices_factory(delay: int):
    return dict(
        mae_deg=lambda q, qhat: jnp.rad2deg(
            jnp.mean(
                maths.angle_error(q[:, :-delay] if delay > 0 else q, qhat[:, delay:])
            )
        ),
    )


def _make_exp_callbacks(ringnet, imtp: IMTP, delay: int):

    callbacks, metrices_name = [], []

    def add_callback(segments: list[str], exp_id, motion_start):
        cb = benchmark(
            imtp=replace(imtp, segments=segments),
            exp_id=exp_id,
            motion_start=motion_start,
            filter=ringnet,
            return_cb=True,
            cb_metrices=_cb_metrices_factory(delay),
        )
        callbacks.append(cb)
        # exclude the first element because it connects to -1 and thus its `mae_deg`
        # loss will be not meaningful
        for segment in segments[1:]:
            metrices_name.append([cb.metric_identifier, "mae_deg", segment])

    add_callback(["seg3"], 2, "slow_fast_mix")
    add_callback(["seg1", "seg2"], 1, "slow1")
    add_callback(["seg3", "seg4"], 1, "fast")
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
    drop_imu_1d: float = 0.75,
    drop_imu_2d: float = 0.25,
    drop_imu_3d: float = 0.1,
    drop_ja_1d: float = 0.5,
    drop_ja_2d: float = 0.5,
    drop_dof: float = 0.0,
    n_val: int = 256,
    rnn_w: int = 400,
    rnn_d: int = 2,
    lin_w: int = 200,
    lin_d: int = 0,
    layernorm: bool = False,
    celltype: str = "gru",
    delay: int = 0,
):
    """
    Main function for training and benchmarking RING neural networks on motion datasets that allows to incorporate some delay.
    So the idea is that RING does not predict the current orientation but the orientation from <seconds> ago.

    Parameters:
        path_lam4 (str): Path to the dataset containing lam4 sequences.
        bs (int): Batch size for training.
        episodes (int): Number of training episodes.
        use_wandb (bool, optional): Whether to log training progress using Weights & Biases. Defaults to False.
        wandb_project (str, optional): Name of the Weights & Biases project. Defaults to "RING".
        wandb_name (str, optional): Optional name for the Weights & Biases run. Defaults to None.
        warmstart (str, optional): Path to warmstart parameters. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 1.
        exp_cbs (bool, optional): Whether to include experimental evaluation callbacks. Defaults to False.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        tbp (int, optional): Truncated backpropagation length. Defaults to 1000.
        drop_imu_1d (float, optional): Probability of dropping IMUs of segments that connect to parent via 1D
            joints during training. Defaults to 0.75.
        drop_imu_2d (float, optional): Probability of dropping IMUs of segments that connect to parent via 2D
            joints during training. Defaults to 0.25.
        drop_imu_3d (float, optional): Probability of dropping IMUs of segments that connect to parent via 3D
            joints during training. Defaults to 0.1.
        drop_ja_1d (float, optional): Probability of dropping joint axes information of segments that connect to
            parent via 1D joint. Defaults to 0.5.
        drop_ja_2d (float, optional): Probability of dropping 2D joint axes information of segments that connect
            to parent via 2D joint. Defaults to 0.5.
        drop_dof (float, optional): Probability of dropping degrees of freedom. Defaults to 0.0.
        n_val (int, optional): Number of samples to use for validation. Defaults to 256.
        rnn_w (int, optional): Width of the RNN layers. Defaults to 400.
        rnn_d (int, optional): Depth of the RNN layers. Defaults to 2.
        lin_w (int, optional): Width of the linear layers. Defaults to 200.
        lin_d (int, optional): Depth of the linear layers. Defaults to 0.
        layernorm (bool, optional): Whether to use layer normalization. Defaults to False.
        celltype (str, optional): Type of RNN cell to use (e.g., "gru", "lstm"). Defaults to "gru".
        delay (int, optional): Number of timesteps to delay estimated orientation.

    Returns:
        None: Trains the network and optionally logs results to Weights & Biases.
    """  # noqa: E501
    np.random.seed(seed)

    if use_wandb:
        unique_id = ring.ml.unique_id()
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    if not ring.ml.on_cluster():
        rnn_w = 10
        rnn_d = 1
        lin_w = 10
        lin_d = 0

    net = ring.ml.RING(
        params=_params(hex(warmstart)) if warmstart else None,
        celltype=celltype,
        lam=Transform.lam,
        layernorm=layernorm,
        hidden_state_dim=rnn_w,
        stack_rnn_cells=rnn_d,
        message_dim=lin_w,
        send_message_n_layers=lin_d,
    )

    imtp = IMTP(
        segments=None,
        sparse=(
            False
            if ((drop_imu_1d == 0) and (drop_imu_2d == 0) and (drop_imu_3d == 0))
            else True
        ),
        joint_axes_1d=False if drop_ja_1d == 1 else True,
        joint_axes_1d_field=False if drop_ja_1d == 1 else True,
        joint_axes_2d=False if drop_ja_2d == 1 else True,
        joint_axes_2d_field=False if drop_ja_2d == 1 else True,
        dof=False if drop_dof == 1 else True,
        dof_field=False if drop_dof == 1 else True,
        dt=True,
        scale_acc=9.81,
        scale_gyr=2.2,
        scale_dt=0.01,
        scale_ja=0.33,
        # this value only influences at which frequency the `diodem` experimental-data
        # validation callbacks are evaluated, so either you train on a) 100 Hz data or
        # b) on multiple sampling rates to achieve generalisation
        hz=100,
    )
    ds = MultiDataset(
        [ShuffledDataset(FolderOfFilesDataset(p)) for p in [path_lam4] * 4],
        Transform(
            imtp,
            drop_imu_1d,
            drop_imu_2d,
            drop_imu_3d,
            drop_ja_1d,
            drop_ja_2d,
            drop_dof,
            delay,
        ),
    )
    ds_train, ds_val = random_split(ds, [len(ds) - n_val, n_val])
    generator = dataset_to_generator(
        ds_train,
        batch_size=bs,
        seed=seed,
        drop_last=True,
        num_workers=None if ring.ml.on_cluster() else 0,
    )
    X_val, y_val = dataset_to_Xy(ds_val)
    callbacks = [
        ring.ml.callbacks.EvalXyTrainingLoopCallback(
            net,
            dict(
                mae_deg=lambda q, qhat: jnp.rad2deg(
                    jnp.mean(maths.angle_error(q, qhat))
                ),
            ),
            X_val,
            y_val,
            Transform.lam,
            "val",
            link_names=Transform.link_names,
        )
    ]

    if exp_cbs:
        callbacks.extend(_make_exp_callbacks(net, imtp, delay))

    n_decay_episodes = int(0.95 * episodes)
    n_steps_per_episode = int(6000 / tbp)
    optimizer = ring.ml.make_optimizer(
        lr, n_decay_episodes, n_steps_per_episode, adap_clip=0.5, glob_clip=None
    )

    def loss_fn(q, qhat):
        "T, N, F -> Scalar"
        loss = jnp.array(0.0)
        for i, p in enumerate(Transform.lam):
            if p == -1:
                loss += jnp.mean(ring.maths.inclination_loss(q[:, i], qhat[:, i]) ** 2)
            else:
                loss += jnp.mean(ring.maths.angle_error(q[:, i], qhat[:, i]) ** 2)
        return loss / len(Transform.lam)

    ml.train_fn(
        generator,
        episodes,
        net,
        optimizer=optimizer,
        callbacks=callbacks,
        callback_kill_if_nan=True,
        callback_save_params=_params(),
        seed_network=seed,
        link_names=Transform.link_names,
        tbp=tbp,
        loss_fn=loss_fn,
        metrices=None,
        callback_create_checkpoint=False,
    )


if __name__ == "__main__":
    fire.Fire(main)
