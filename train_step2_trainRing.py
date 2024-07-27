import fire
import jax
import jax.numpy as jnp
import numpy as np
import qmt
import ring
from ring import maths
from ring import ml
from ring import utils
from ring.algorithms.generator.finalize_fns import GeneratorTrafoLambda
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
        Xy = utils.pickle_load(path)
        # convert batch axis to list
        N = tree_utils.tree_shape(Xy)
        Xy = [jax.tree.map(lambda a: a[i], Xy) for i in range(N)]
        return utils.replace_elements_w_nans(Xy, verbose=1)

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
    return seq


def _expand_dt(X: dict, T: int):
    for seg in link_names:
        new_name = str(link_names.index(seg))
        dt = X[f"dt_{seg[5:10]}"]
        X[new_name]["dt"] = np.repeat(dt[:, None, None], T, axis=1)
    for i in range(1, 5):
        X.pop(f"dt_{i}Seg")
    return X


def _expand_then_flatten(X, y):
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
        X, y = jax.tree.map(lambda arr: arr[0], (X, y))
    return X, y


def _qinv_root(lam, y):
    for i, p in enumerate(lam):
        if p == -1:
            y = y.at[:, :, i].set(maths.quat_inv(y[:, :, i]))
    return y


def c_to_parent_TO_c_to_eps(lam, y):
    y = y.copy()
    for i, p in enumerate(lam):
        if p == -1:
            continue
        y = y.at[:, :, i].set(maths.quat_mul(y[:, :, p], y[:, :, i]))
    return y


def c_to_eps_TO_c_to_parent(lam, y):
    y_i_to_p = y.copy()
    for i, p in enumerate(lam):
        if p == -1:
            continue
        y_i_to_p = y_i_to_p.at[:, :, i].set(
            maths.quat_mul(maths.quat_inv(y[:, :, p]), y[:, :, i])
        )
    return y_i_to_p


def rand_quats(imu_available) -> np.ndarray:
    "Returns array (B, 10, 4)"
    B = imu_available.shape[0]
    qrand = qmt.randomQuat((B, 10))
    qrand[~imu_available.astype(bool)] = np.array([1.0, 0, 0, 0])
    return qrand


def pmap(f):
    pmap_f = jax.pmap(f)

    def _f(*args):
        B = tree_utils.tree_shape(args)
        sizes = utils.distribute_batchsize(B)
        args = utils.expand_batchsize(args, *sizes)
        out = pmap_f(*args)
        return utils.merge_batchsize(out, *sizes)

    return _f


def _rotate(qrand, vec):
    assert qrand.shape[1:] == (10, 4)
    return jax.vmap(maths.rotate, (1, None), out_axes=1)(vec, qrand)


@pmap
def rotate_X(qrand, X):
    X_0to3 = _rotate(qrand, X[..., :3])
    X_3to6 = _rotate(qrand, X[..., 3:6])
    X_6to9 = _rotate(qrand, X[..., 6:9])
    return jnp.concatenate((X_0to3, X_3to6, X_6to9, X[..., 9:10]), axis=-1)


@pmap
def rotate_y(qrand, y):

    y = _qinv_root(lam, y)
    y = c_to_parent_TO_c_to_eps(lam, y)
    y = jax.vmap(maths.quat_mul, (1, None), 1)(y, maths.quat_inv(qrand))
    y = c_to_eps_TO_c_to_parent(lam, y)
    y = _qinv_root(lam, y)

    for i, p in enumerate(lam):
        if p == -1:
            y = y.at[:, :, i].set(
                maths.quat_project(y[:, :, i], jnp.array([0, 0, 1.0]))[1]
            )

    return y


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

        X, y = _expand_then_flatten(X, y)
        if do_randomize_imus:
            qrand = rand_quats(factor_imus)
            X, y = rotate_X(qrand, X), rotate_y(qrand, y)
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
        callback_save_params_track_metrices=[["exp_val_mae_deg"]] if exp_cbs else None,
        seed_network=seed,
        link_names=link_names,
    )

    print(f"Trained parameters saved to {path_trained_params}")


if __name__ == "__main__":
    fire.Fire(main)
