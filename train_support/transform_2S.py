from jpos.jpos import _lpf
import numpy as np
import qmt
import tree


def rand_quat_like(q, max_deg: float = 180.0):
    assert q.shape[-1] == 4
    qrand = qmt.randomQuat(q.shape[:-1])
    angle, axis = qmt.quatAngle(qrand), qmt.quatAxis(qrand)
    angle = angle * (max_deg / 180)
    return qmt.quatFromAngleAxis(angle, axis)


class LPF:
    def __init__(self, hz, cutoff: float | None):
        self.hz = hz
        self.cutoff = cutoff

    def __call__(self, x):
        if self.cutoff is None:
            return x
        return _lpf(x, self.hz, self.cutoff)


class Transform:

    def __init__(
        self,
        rand_ori: bool,
        hz: float,
        cutoff: float | None,
        AR: bool,
        max_deg: float | None,
    ):
        self._rand_ori = rand_ori
        self.mode = None
        self.lpf = LPF(hz, cutoff)
        self.F = None
        self.AR = AR
        self.max_deg = max_deg

    def sim(self):
        self.mode = "sim"
        self.rand_ori = self._rand_ori
        self.rand_swap = True

    def diodem(self, cb: bool = False):
        self.mode = "diodem"
        if cb:
            self.rand_ori = False
            self.rand_swap = False
        else:
            self.rand_ori = self._rand_ori
            self.rand_swap = True

    def setDOF(self, dof: int | None):
        assert dof in [1, 2, 3, None]
        self.F = 6
        if dof is not None:
            self.F += 3
        if self.AR:
            self.F += 4
        self._dof = dof

    def __call__(self, ele):

        assert self.mode is not None
        assert self.F is not None

        unpack = self._unpack_dio if self.mode == "diodem" else self._unpack_sim
        a1, a2, g1, g2, q1, q2 = unpack(ele)

        q1r = qmt.randomQuat() if self.rand_ori else np.array([1.0, 0, 0, 0])
        q2r = qmt.randomQuat() if self.rand_ori else np.array([1.0, 0, 0, 0])
        a1, g1 = qmt.rotate(q1r, a1), qmt.rotate(q1r, g1)
        a2, g2 = qmt.rotate(q2r, a2), qmt.rotate(q2r, g2)
        q1, q2 = qmt.qmult(q1, qmt.qinv(q1r)), qmt.qmult(q2, qmt.qinv(q2r))

        if self.rand_swap and np.random.choice([False, True]):
            a1, a2 = a2, a1
            g1, g2 = g2, g1
            q1, q2 = q2, q1

        X = np.zeros((a1.shape[0], 2, self.F))
        grav, pi = 9.81, 2.2
        X[:, 0, 0:3] = a1 / grav
        X[:, 1, 0:3] = a2 / grav
        X[:, 0, 3:6] = g1 / pi
        X[:, 1, 3:6] = g2 / pi

        if self._dof is not None:
            X[:, 1, 6 + self._dof - 1] = 1.0

        Y = np.zeros((a1.shape[0], 2, 4))
        # my `maths.quat_project` and `qmt.quatProject` are identical under q -> q^-1
        # so normally i would do `maths.quat_project(quat_inv(q1))` here but so instead
        # we do qmt.quatProject(q1)
        Y[:, 0] = qmt.quatProject(q1, [0, 0, 1.0])["resQuat"]
        Y[:, 1] = qmt.qmult(qmt.qinv(q1), q2)

        if self.AR:
            X[..., -4:] = np.concatenate((Y[0:1], Y[0:-1]))
            if self.max_deg is not None:
                X[..., -4:] = qmt.qmult(
                    rand_quat_like(X[..., -4:], self.max_deg), X[..., -4:]
                )

        return X, Y

    @staticmethod
    def _unpack_sim(ele):
        X_d, y_d = ele

        seg1, seg2 = X_d["seg1"], X_d["seg2"]
        a1, a2 = seg1["acc"], seg2["acc"]
        g1, g2 = seg1["gyr"], seg2["gyr"]

        # -> from body1 to epsilon
        if "floatBase" in y_d:
            qEB1 = qmt.qmult(y_d["floatBase"], y_d["seg1"])
            qEB2 = qmt.qmult(y_d["floatBase"], y_d["seg2"])
        else:
            qEB1 = y_d["seg1"]
            qB1B2 = y_d["seg2"]
            qEB2 = qmt.qmult(qEB1, qB1B2)

        return a1, a2, g1, g2, qEB1, qEB2

    def _unpack_dio(self, ele):
        a1, a2, g1, g2 = tree.map_structure(
            self.lpf,
            (
                ele["seg1"]["acc"],
                ele["seg2"]["acc"],
                ele["seg1"]["gyr"],
                ele["seg2"]["gyr"],
            ),
        )
        return a1, a2, g1, g2, ele["seg1"]["quat"], ele["seg2"]["quat"]
