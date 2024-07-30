import fire
import jax
import jax.numpy as jnp
import numpy as np
import qmt
import ring
from ring import maths
from ring import ml
import tree as tree_lib
import tree_utils
import wandb

import dataloader
from exp_cbs import make_exp_callbacks

lam = [-1, 0]
link_names = [
    "seg3_2Seg",
    "seg4_2Seg",
]


def batch_concat_acme(
    tree,
    num_batch_dims: int = 1,
) -> np.ndarray:
    """Flatten and concatenate nested array structure, keeping batch dims.
    IGNORES the ordered of elements in an `OrderedDict`, see EngineeringLog @ 18.02.23
    """

    def _flatten(x: np.ndarray, num_batch_dims: int) -> np.ndarray:
        if x.ndim < num_batch_dims:
            return x
        return np.reshape(x, list(x.shape[:num_batch_dims]) + [-1])

    flatten_fn = lambda x: _flatten(x, num_batch_dims)
    flat_leaves = tree_lib.map_structure(flatten_fn, tree)
    return np.concatenate(tree_lib.flatten(flat_leaves), axis=-1)


def _flatten(seq: list):
    seq = tree_utils.tree_batch(seq, backend=None)
    seq = batch_concat_acme(seq, num_batch_dims=3).transpose((1, 2, 0, 3))
    return seq


def _expand_then_flatten(X, y):
    gyr = X["0"]["gyr"]

    batched = True
    if gyr.ndim == 2:
        batched = False
        X, y = tree_utils.add_batch_dim((X, y))

    N = len(link_names)

    def dict_to_tuple_X(d: dict[str, np.ndarray]):
        tup = (d["acc"], d["gyr"])
        if "joint_axes" in d:
            tup = tup + (d["joint_axes"],)
        if "dt" in d:
            tup = tup + (d["dt"],)
        return tup

    def dict_to_tuple_Y(d: dict[str, np.ndarray]):
        return (d["quat"], d["pos"])

    X = [dict_to_tuple_X(X[str(i)]) for i in range(N)]
    y = [dict_to_tuple_Y(y[str(i)]) for i in range(N)]
    X, y = _flatten(X), _flatten(y)
    if not batched:
        X, y = jax.tree.map(lambda arr: arr[0], (X, y))
    return X, y


def _qinv_root_(y: np.ndarray) -> np.ndarray:
    for i, p in enumerate(lam):
        if p == -1:
            y[..., i, :] = qmt.qinv(y[..., i, :])


def c_to_parent_TO_c_to_eps_(y: np.ndarray) -> np.ndarray:
    for i, p in enumerate(lam):
        if p == -1:
            continue
        y[..., i, :] = qmt.qmult(y[..., p, :], y[..., i, :])


def c_to_eps_TO_c_to_parent_(y: np.ndarray) -> np.ndarray:
    y_eps = y.copy()
    for i, p in enumerate(lam):
        if p == -1:
            continue
        y[..., i, :] = qmt.qmult(qmt.qinv(y_eps[..., p, :]), y_eps[..., i, :])


def rand_quats(imu_available: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    "Returns array (B, 10, 4) or (10, 4)"
    qrand = rng.standard_normal(size=imu_available.shape + (4,))
    qrand = qrand / np.linalg.norm(qrand, axis=-1, keepdims=True)
    qrand[~imu_available.astype(bool)] = np.array([1.0, 0, 0, 0])
    return qrand


def rotate_X_(qrand, X):
    batched = X.ndim > 3
    qrand = qrand[None, None] if batched else qrand[None]
    X[..., :3] = qmt.rotate(qrand, X[..., :3])
    X[..., 3:6] = qmt.rotate(qrand, X[..., 3:6])
    return X


def rotate_y_quat(qrand, y):

    _qinv_root_(y)
    c_to_parent_TO_c_to_eps_(y)
    y = qmt.qmult(y, qmt.qinv(qrand[None]))
    c_to_eps_TO_c_to_parent_(y)
    _qinv_root_(y)

    for i, p in enumerate(lam):
        if p == -1:
            y[..., i, :] = qmt.qmult(
                y[..., i, :],
                qmt.qinv(-qmt.quatProject(y[..., i, :], [0, 0, 1.0])["projQuat"]),
            )

    return y


class Transform:
    def __init__(self, rand_imus: bool):
        self.rand_imus = rand_imus

    def __call__(self, data: list, rng: np.random.Generator):
        X, y = data[0]
        factor_imus = np.array([1.0, 1.0])
        self._rename_links(X)
        self._rename_links(y)

        X, y = _expand_then_flatten(X, y)
        if self.rand_imus:
            qrand = rand_quats(factor_imus, rng)
            rotate_X_(qrand, X)
            y[..., :4] = rotate_y_quat(qrand, y[..., :4].copy())
            y[..., 4:] = qmt.rotate(qrand[None], y[..., 4:])
        return X, y

    @staticmethod
    def _rename_links(d: dict[str, dict]):
        for key in list(d.keys()):
            if key in link_names:
                d[str(link_names.index(key))] = d.pop(key)


def _make_ring(lam, params_warmstart: str | None, dry_run: bool):
    hidden_state_dim = 400 if not dry_run else 20
    message_dim = 200 if not dry_run else 10

    def link_output_transform(y):
        y = y.at[..., :4].set(maths.safe_normalize(y[..., :4]))
        return y

    ringnet = ml.RING(
        lam=lam,
        hidden_state_dim=hidden_state_dim,
        message_dim=message_dim,
        params=params_warmstart,
        link_output_dim=7,
        link_output_normalize=False,
        link_output_transform=link_output_transform,
    )
    ringnet = ml.base.ScaleX_FilterWrapper(ringnet)
    return ringnet


class OnlyQuatWrapper(ml.base.AbstractFilterWrapper):
    def apply(self, X, params=None, state=None, y=None, lam=None):
        yhat, state = super().apply(X, params, state, y, lam)
        return yhat[..., :4], state


class OnlyPosWrapper(ml.base.AbstractFilterWrapper):
    def apply(self, X, params=None, state=None, y=None, lam=None):
        yhat, state = super().apply(X, params, state, y, lam)
        return yhat[..., 4:], state


def main(
    path_lam2,
    bs: int,
    episodes: int,
    path_trained_params: str | None = None,
    use_wandb: bool = False,
    wandb_project: str = "RING",
    params_warmstart: str = None,
    seed: int = 1,
    dry_run: bool = False,
    exp_cbs: bool = False,
    rand_imus: bool = False,
    worker_count: int = 0,
    lr: float = 1e-3,
):
    """Train RING using data from step1.

    Args:
        path_lam1 (str): path to data of model xml `lam1`
        path_lam2 (str): path to data of model xml `lam2`
        path_lam3 (str): path to data of model xml `lam3`
        path_lam4 (str): path to data of model xml `lam4`. This corresponds
            to the four segment kinematic chain system.
        bs (int): batchsize for training
        episodes (int): number of training iterations
        path_trained_params (str): path where trained parameters will be saved
        use_wandb (bool, optional): use wandb for tracking. Defaults to False.
        wandb_project (str, optional): wandb project name. Defaults to "RING".
        params_warmstart (str, optional): warmstart training from parameters saved at
            this path. Defaults to None.
        seed (int, optional): PRNG used for initial params of RING (if
            `params_warmstart` is not given). Defaults to 1.
        dry_run (bool, optional): Make RING network tiny for faster training (for
            testing purposes). Defaults to False.
    """

    np.random.seed(seed)

    if use_wandb:
        wandb.init(project=wandb_project, config=locals())

    if path_trained_params is None:
        path_trained_params = f"~/params/{ring.ml.unique_id()}.pickle"

    ringnet = _make_ring(lam, params_warmstart, dry_run)

    generator = dataloader.make_generator(
        path_lam2,
        batch_size=bs,
        transform=Transform(rand_imus),
        worker_count=worker_count,
    )

    ringnet_exp = ml.base.GroundTruthHeading_FilterWrapper(OnlyQuatWrapper(ringnet))
    callbacks = (
        make_exp_callbacks(
            ringnet_exp,
            seg1=False,
            seg4=False,
            seg2_ja=False,
            seg2_flex=False,
            seg2_dt=False,
        )
        if exp_cbs
        else []
    )

    metrices = {
        "mae_deg": (
            lambda y, yhat: maths.angle_error(y[..., :4], yhat[..., :4]),
            lambda arr: jnp.rad2deg(jnp.mean(arr, axis=(0, 1))),
        ),
        "mae_pos_m": (
            lambda y, yhat: jnp.linalg.norm(y[..., 4:] - yhat[..., 4:]),
            jnp.mean,
        ),
    }

    optimizer = ml.make_optimizer(
        lr,
        episodes,
        n_steps_per_episode=6,
        skip_large_update_max_normsq=100.0,
    )

    def loss_fn(y, yhat):
        # (T, N, F) -> Scalar
        dpos = y[..., 4:] - yhat[..., 4:]
        return jnp.mean(maths.angle_error(y[..., :4], yhat[..., :4]) ** 2) + jnp.mean(
            jnp.sum(dpos**2, axis=-1)
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
        callback_save_params=path_trained_params,
        callback_save_params_track_metrices=[["exp_val_mae_deg"]] if exp_cbs else None,
        seed_network=seed,
        link_names=link_names,
        loss_fn=loss_fn,
        metrices=metrices,
    )

    print(f"Trained parameters saved to {path_trained_params}")


if __name__ == "__main__":
    fire.Fire(main)
