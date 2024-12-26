from absl import flags
from diodem.benchmark import IMTP
import numpy as np
import qmt
from ring.utils.dataloader_torch import FolderOfFilesDataset
from ring.utils.dataloader_torch import MultiDataset
from ring.utils.dataloader_torch import ShuffledDataset
from torch.utils.data import random_split

FLAGS = flags.FLAGS
flags.DEFINE_float(
    "drop_imu_1d",
    0.75,
    "probability of dropping IMUs of segments that connect to parent via "
    "1D joints during training and validation.",
)
flags.DEFINE_float(
    "drop_imu_2d",
    0.25,
    "probability of dropping IMUs of segments that connect to parent via "
    "2D joints during training and validation.",
)
flags.DEFINE_float(
    "drop_imu_3d",
    0.1,
    "probability of dropping IMUs of segments that connect to parent via "
    "3D joints during training and validation.",
)
flags.DEFINE_float(
    "drop_ja_1d",
    0.5,
    "Probability of dropping joint axes information of segments that connect "
    "to parent via 1D joint",
)
flags.DEFINE_float(
    "drop_ja_2d",
    0.5,
    "Probability of dropping joint axes information of segments that connect "
    "to parent via 2D joint",
)
flags.DEFINE_float("drop_dof", 0.0, "Probability of dropping degrees of freedom")
flags.DEFINE_bool(
    "three_seg", False, "Whether to train on 1-Seg, 2-Seg, and 3-Seg chains"
)
flags.DEFINE_bool(
    "four_seg", False, "Whether to train on 1-Seg, 2-Seg, 3-Seg, and 4-Seg chains"
)


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
        self, imtp: IMTP, rnno: bool = False, flatten: bool = False, T: int = 6000
    ):
        self.imtp = imtp
        self.drop_imu = {
            1: FLAGS.drop_imu_1d,
            2: FLAGS.drop_imu_2d,
            3: FLAGS.drop_imu_3d,
        }
        self.drop_ja_1d = FLAGS.drop_ja_1d
        self.drop_ja_2d = FLAGS.drop_ja_2d
        self.drop_dof = FLAGS.drop_dof
        self.rnno = rnno
        self.three_seg = FLAGS.three_seg
        self.four_seg = FLAGS.four_seg
        self.flatten = flatten
        self.T = T

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

        X, Y = X.transpose((2, 1, 0)), Y.transpose((1, 0, 2))

        if self.rnno:
            X, Y = self._rnno_output_transform(X, Y)
        else:
            if self.four_seg:
                M = 10
            elif self.three_seg:
                M = 6
            else:
                M = 3
            X, Y = X[:, :M], Y[:, :M]

        if self.flatten:
            X = X.reshape((T, -1))
            Y = Y.reshape((T, -1))

        start = np.random.randint(0, T - self.T + 1)
        r = slice(start, start + self.T)
        X, Y = X[r], Y[r]

        return X, Y

    def _rnno_output_transform(self, _X, _Y):
        "X: (T, Nseg, F), Y: (T, Nseg, 4) -> (T, 4, 5)"
        starts = [0, 1, 3, 6]
        if self.four_seg:
            Ms = [1, 2, 3, 4]
        elif self.three_seg:
            Ms = [1, 2, 3]
        else:
            Ms = [1, 2]
        M = np.random.choice(Ms)

        T = _X.shape[0]
        F = _X.shape[-1]
        X, Y = np.zeros((T, 4, F)), np.zeros((T, 4, 4))

        r = slice(starts[M - 1], starts[M - 1] + M)
        X[:, :M] = _X[:, r]
        Y[:, :M] = _Y[:, r]
        Y[:, M:] = np.array([1.0, 0, 0, 0])[None, None]

        return X, Y


def load_imtp() -> IMTP:
    return IMTP(
        segments=None,
        sparse=True,
        joint_axes_1d=True,
        joint_axes_1d_field=True,
        joint_axes_2d=True,
        joint_axes_2d_field=True,
        dof=True,
        dof_field=True,
        dt=True,
        scale_acc=9.81,
        scale_gyr=2.2,
        scale_dt=0.01,
        scale_ja=0.3,
    )


flags.DEFINE_integer("n_val", 256, "Number of samples to use for validation")
flags.DEFINE_string(
    "path_lam4", None, "Path to the dataset containing lam4 sequences", required=True
)


def load_ds_train_ds_val(rnno: bool, flatten: bool, T: int = 6000):

    ds = MultiDataset(
        [ShuffledDataset(FolderOfFilesDataset(p)) for p in [FLAGS.path_lam4] * 4],
        Transform(load_imtp(), rnno, flatten, T),
    )
    ds_train, ds_val = random_split(ds, [len(ds) - FLAGS.n_val, FLAGS.n_val])
    return ds_train, ds_val
