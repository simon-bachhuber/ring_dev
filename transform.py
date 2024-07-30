import numpy as np
import qmt
import tree as tree_lib
import tree_utils

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


def tree_batch(
    trees,
):
    trees = tree_lib.map_structure(lambda arr: arr[None], trees)

    if len(trees) == 0:
        return trees
    if len(trees) == 1:
        return trees[0]

    return tree_lib.map_structure(lambda *arrs: np.concatenate(arrs, axis=0), *trees)


def _flatten(seq: list):
    seq = tree_batch(seq)
    seq = batch_concat_acme(seq, num_batch_dims=3).transpose((1, 2, 0, 3))
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

    def dict_to_tuple(d: dict[str, np.ndarray]):
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
        X, y = tree_lib.map_structure(lambda arr: arr[0], (X, y))
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
    X[..., 6:9] = qmt.rotate(qrand, X[..., 6:9])
    return X


def rotate_y(qrand, y):

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


def make_10_body_system(data: list) -> tuple:

    dts = {f"dt_{j + 1}Seg": data[j][0].pop("dt") for j in range(4)}
    X, y = dts, {}
    for j in range(4):
        X.update(data[j][0])
        y.update(data[j][1])
    return X, y


class Transform:
    def __init__(self, rand_imus: bool):
        self.rand_imus = rand_imus

    def __call__(self, data: list, rng: np.random.Generator):
        X, y = make_10_body_system(data)
        draw = lambda p: np.array(rng.binomial(1, p), dtype=float)

        factor_imus = []
        for segments, (imu_rate, jointaxes_rate) in dropout_rates.items():
            factor_imu = draw(1 - imu_rate)
            factor_imus.append(factor_imu[None])
            factor_ja = draw(1 - jointaxes_rate)

            X[segments]["acc"] *= factor_imu
            X[segments]["gyr"] *= factor_imu
            X[segments]["joint_axes"] *= factor_ja

        self._rename_links(X)
        self._rename_links(y)
        factor_imus = np.concatenate(factor_imus).T
        assert factor_imus.shape == (10,), f"{factor_imus.shape} != (10,)"

        X, y = _expand_then_flatten(X, y)
        if self.rand_imus:
            qrand = rand_quats(factor_imus, rng)
            rotate_X_(qrand, X)
            y = rotate_y(qrand, y)
        return X, y

    @staticmethod
    def _rename_links(d: dict[str, dict]):
        for key in list(d.keys()):
            if key in link_names:
                d[str(link_names.index(key))] = d.pop(key)
