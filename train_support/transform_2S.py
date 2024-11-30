from jpos.jpos import _lpf
import numpy as np
import qmt
import tree


class LPF:
    def __init__(self, hz, cutoff: float | None):
        self.hz = hz
        self.cutoff = cutoff

    def __call__(self, x):
        if self.cutoff is None:
            return x
        return _lpf(x, self.hz, self.cutoff)


class Transform:

    def __init__(self, rand_ori: bool, hz: float, cutoff: float | None, rel_only: bool):
        self._rand_ori = rand_ori
        self.mode = None
        self.lpf = LPF(hz, cutoff)
        self.F = None
        self.rel_only = rel_only

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
        self.F = 6 if dof is None else 9
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

        if self.F == 9:
            X[:, 1, 6 + self._dof - 1] = 1.0

        Y = np.zeros((a1.shape[0], 2, 4))
        Y[:, 0] = qmt.quatProject(qmt.qinv(q1), [0, 0, 1.0])["resQuat"]
        Y[:, 1] = qmt.qmult(qmt.qinv(q1), q2)

        if self.rel_only:
            Y[:, 1] = qmt.qmult(Y[:, 1], qmt.qinv(Y[0, 1]))

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
