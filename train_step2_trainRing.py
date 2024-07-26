import fire
import jax
import jax.numpy as jnp
import numpy as np
import qmt
import ring
from ring import maths
from ring import ml
from ring.algorithms.generator.finalize_fns import GeneratorTrafoLambda
from ring.utils import pickle_load
from ring.utils import replace_elements_w_nans
import tree_utils
import wandb

from exp_cbs import make_exp_callbacks

dropout_rates = dict(
    seg3_1Seg=(0.0, 1.0),
    seg3_2Seg=(0.0, 1.0),
    seg4_2Seg=(0.0, 0.5),
    seg3_3Seg=(0.0, 1.0),
    # this used to be 2/3 and not 4/5
    seg4_3Seg=(4 / 5, 0.5),
    seg5_3Seg=(0.0, 0.5),
    seg2_4Seg=(0.0, 1.0),
    seg3_4Seg=(3 / 4, 1 / 4),
    seg4_4Seg=(3 / 4, 1 / 4),
    seg5_4Seg=(0.0, 1 / 4),
)
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


def load_data(
    path_lam1,
    path_lam2,
    path_lam3,
    path_lam4,
) -> list:
    "Returns list of data of 10-body system"

    def loader(path):
        Xy = pickle_load(path)
        # convert batch axis to list
        N = tree_utils.tree_shape(Xy)
        Xy = [jax.tree_map(lambda a: a[i], Xy) for i in range(N)]
        return replace_elements_w_nans(Xy, verbose=1)

    # 4xN
    data_in = [loader(p) for p in [path_lam1, path_lam2, path_lam3, path_lam4]]
    # convert to Nx4
    N = min([len(ele) for ele in data_in])
    data_out = []
    for i in range(N):
        data_out.append([ele[i] for ele in data_in])
        for j in range(4):
            data_in[j][i] = None
    del data_in

    # combine to 10-body system
    for i in range(N):
        data_i = data_out[i]
        dts = {f"dt_{j + 1}Seg": data_i[j][0].pop("dt") for j in range(4)}
        X, y = dts, {}
        for j in range(4):
            X.update(data_i[j][0])
            y.update(data_i[j][1])
        data_out[i] = (X, y)

    return data_out


def train_val_split(
    data: list,
    bs: int,
    transform_gen,
):

    n_batches_for_val = 1
    len_val = n_batches_for_val * bs
    train_data, val_data = data[:-len_val], data[-len_val:]

    X_val, y_val = transform_gen(ring.RCMG.eager_gen_from_list(val_data, len_val))(
        jax.random.PRNGKey(420)
    )

    generator = transform_gen(
        ring.RCMG.eager_gen_from_list(
            train_data,
            bs,
        )
    )

    return generator, (X_val, y_val)


def _flatten(seq: list):
    seq = tree_utils.tree_batch(seq, backend=None)
    seq = tree_utils.batch_concat_acme(seq, num_batch_dims=3).transpose((1, 2, 0, 3))
    return np.array(seq)


def _expand_dt(X: dict, T: int):
    for seg in link_names:
        new_name = str(link_names.index(seg))
        dt = X[f"dt_{seg[5:10]}"]
        X[new_name]["dt"] = np.repeat(dt[:, None, None], T, axis=1)
    for i in range(1, 5):
        X.pop(f"dt_{i}Seg")
    return X


def _expand_then_flatten(Xy):
    X, y = Xy
    gyr = X["0"]["gyr"]

    batched = True
    if gyr.ndim == 2:
        batched = False
        X, y = tree_utils.add_batch_dim((X, y))

    X = _expand_dt(X, gyr.shape[-2])

    N = len(X)

    def dict_to_tuple(d: dict[str, jax.Array]):
        tup = (d["acc"], d["gyr"])
        if "joint_axes" in d:
            tup = tup + (d["joint_axes"],)
        if "dt" in d:
            tup = tup + (d["dt"],)
        return tup

    X = [dict_to_tuple(X[str(i)]) for i in range(N)]
    y = [y[str(i)] for i in range(N)]
    X, y = _flatten(X), _flatten(y)
    if not batched:
        X, y = jax.tree_map(lambda arr: arr[0], (X, y))
    return X, y


def rand_quat(shape=()) -> np.ndarray:
    q = np.random.normal(size=shape + (4,))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def c_to_parent_TO_c_to_eps(lam, y):
    y = y.copy()
    for i, p in enumerate(lam):
        if p == -1:
            continue
        y[:, :, i] = qmt.qmult(y[:, :, p], y[:, :, i])
    return y


def c_to_eps_TO_c_to_parent(lam, y):
    y_i_to_p = y.copy()
    for i, p in enumerate(lam):
        if p == -1:
            continue
        y_i_to_p[:, :, i] = qmt.qmult(qmt.qinv(y[:, :, p]), y[:, :, i])
    return y_i_to_p


def randomize_imus(lam, X, y, imus: np.ndarray):
    B, T, N, F = X.shape
    # segment to IMU
    qrand = rand_quat((B, N))
    unit_quats = np.zeros((B, N, 4))
    unit_quats[..., 0] = 1.0
    # (bs, 10) -> (bs, 10, 4)
    imus = np.repeat(imus[..., None], 4, axis=-1)
    qrand = np.where(imus, qrand, unit_quats)
    qrand = np.repeat(qrand[:, None], T, axis=1)
    X[..., :3] = qmt.rotate(qrand, X[..., :3])
    X[..., 3:6] = qmt.rotate(qrand, X[..., 3:6])
    X[..., 6:9] = qmt.rotate(qrand, X[..., 6:9])
    y = c_to_parent_TO_c_to_eps(lam, y)
    y = qmt.qinv(y)
    y = qmt.qmult(qrand, y)
    y = qmt.qinv(y)
    y = c_to_eps_TO_c_to_parent(lam, y)
    for i, p in enumerate(lam):
        if p == -1:
            i_to_eps = y[:, :, i]
            eps_to_i = ring.maths.quat_project(
                qmt.qinv(i_to_eps), np.array([0, 0, 1.0])
            )[1]
            y[:, :, i] = qmt.qinv(eps_to_i)
    return X, y


def output_transform_factory(link_names, batchsize: int, do_randomize_imus: bool):

    def _rename_links(d: dict[str, dict]):
        for key in list(d.keys()):
            if key in link_names:
                d[str(link_names.index(key))] = d.pop(key)

    def output_transform(tree):
        X, y = tree

        draw = lambda p: np.random.binomial(1, p, size=batchsize).astype(float)[
            :, None, None
        ]

        factor_imus = []
        for segments, (imu_rate, jointaxes_rate) in dropout_rates.items():
            factor_imu = draw(1 - imu_rate)
            factor_imus.append(factor_imu[None, :, 0, 0])
            factor_ja = draw(1 - jointaxes_rate)

            X[segments]["acc"] *= factor_imu
            X[segments]["gyr"] *= factor_imu
            X[segments]["joint_axes"] *= factor_ja

        _rename_links(X)
        _rename_links(y)
        factor_imus = np.concatenate(factor_imus).T
        assert factor_imus.shape == (
            batchsize,
            10,
        ), f"{factor_imus.shape} != {(batchsize, 10)}"
        X, y = _expand_then_flatten((X, y))
        if do_randomize_imus:
            return randomize_imus(lam, X, y, factor_imus)
        else:
            return X, y

    return output_transform


def _make_ring(lam, params_warmstart: str | None, dry_run: bool):
    hidden_state_dim = 400 if not dry_run else 20
    message_dim = 200 if not dry_run else 10
    ringnet = ml.RING(
        lam=lam,
        hidden_state_dim=hidden_state_dim,
        message_dim=message_dim,
        params=params_warmstart,
    )
    ringnet = ml.base.ScaleX_FilterWrapper(ringnet)
    ringnet = ml.base.GroundTruthHeading_FilterWrapper(ringnet)
    return ringnet


def main(
    path_lam1,
    path_lam2,
    path_lam3,
    path_lam4,
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

    data = load_data(path_lam1, path_lam2, path_lam3, path_lam4)
    generator, (X_val, y_val) = train_val_split(
        data,
        bs,
        transform_gen=GeneratorTrafoLambda(
            output_transform_factory(link_names, bs, rand_imus)
        ),
    )

    _mae_metrices = dict(
        mae_deg=lambda q, qhat: jnp.rad2deg(
            jnp.mean(maths.angle_error(q, qhat)[:, 2500:])
        )
    )

    callbacks = make_exp_callbacks(ringnet) if exp_cbs else []

    callbacks += [
        ml.callbacks.EvalXyTrainingLoopCallback(
            ringnet,
            _mae_metrices,
            X_val,
            y_val,
            None,
            "val",
            link_names=link_names,
        )
    ]

    optimizer = ml.make_optimizer(
        1e-3,
        episodes,
        n_steps_per_episode=6,
        skip_large_update_max_normsq=100.0,
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
        seed_network=seed,
        link_names=link_names,
    )

    print(f"Trained parameters saved to {path_trained_params}")


if __name__ == "__main__":
    fire.Fire(main)
