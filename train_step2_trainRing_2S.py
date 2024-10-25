import fire
import numpy as np
import ring
from ring.utils.dataloader import make_generator
import wandb


class TransformFactory:

    def __call__(self, ele: list, rng):
        X_d, y = ele[0]
        if isinstance(X_d, tuple):
            X_d, y = X_d

        seg1, seg2 = X_d["seg1"], X_d["seg2"]
        a1, a2 = seg1["acc"], seg2["acc"]
        g1, g2 = seg1["gyr"], seg2["gyr"]

        T = a1.shape[0]
        F = 6

        X = np.zeros((T, 2, F))

        X[:, 0, :3] = a1 / 9.81
        X[:, 0, 3:6] = g1 / 3.14
        X[:, 1, :3] = a2 / 9.81
        X[:, 1, 3:6] = g2 / 3.14

        return X, y["seg2"][:, None]


class Wrapper(ring.ml.base.AbstractFilterWrapper):
    def apply(self, X, params=None, state=None, y=None, lam=None):
        yhat, state = super().apply(X, params, state, y, lam)
        return yhat[..., 1:2, :], state


def _params(unique_id: str = ring.ml.unique_id()) -> str:
    home = "/home/woody/iwb3/iwb3004h/simon/" if ring.ml.on_cluster() else "~/"
    return home + f"params/{unique_id}.pickle"


def main(
    data_path: str,
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
    rnno: bool = False,
):
    np.random.seed(seed)

    if use_wandb:
        unique_id = ring.ml.unique_id()
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    gen = make_generator(
        data_path,
        batch_size=bs,
        transform=TransformFactory(),
        backend="torch",
        num_workers=num_workers,
    )
    params = _params(hex(warmstart)) if warmstart else None
    celltype = "lstm" if lstm else "gru"

    if rnno:
        net = ring.ml.RNNO(
            8,
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
    else:
        net = ring.ml.RING(
            lam=(-1, 0),
            hidden_state_dim=rnn_w,
            message_dim=lin_w,
            celltype=celltype,
            stack_rnn_cells=rnn_d,
            send_message_n_layers=lin_d,
        )

    net = Wrapper(net)

    ring.ml.train_fn(
        gen,
        episodes,
        net,
        ring.ml.make_optimizer(lr, episodes, adap_clip=None, glob_clip=1.0),
        callback_kill_after_seconds=23.5 * 3600,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        seed_network=seed,
        callback_save_params=_params(),
    )


if __name__ == "__main__":
    fire.Fire(main)
