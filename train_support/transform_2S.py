import numpy as np
import qmt


class Transform:

    def __init__(self, dof: int | None, rand_ori: bool, pos: bool, use_vqf: bool):
        assert dof in [1, 2, 3, None]
        self.dof = dof
        self.rand_ori = rand_ori
        self.pos = pos
        self.use_vqf = use_vqf

    def __call__(self, ele):
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
        del q1, q2

        F = 12
        if self.dof is not None:
            F += 3
        if self.pos:
            F += 6
        if self.use_vqf:
            F += 12
        dt = X_d.get("dt", None)
        if dt is not None:
            F += 1

        X = np.zeros((a1.shape[0], F))
        grav, pi = 9.81, 2.2
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
            q1 = qmt.oriEstVQF(g1, a1, params=dict(Ts=float(_dt)))
            q2 = qmt.oriEstVQF(g2, a2, params=dict(Ts=float(_dt)))
            X[:, i : (i + 4)] = q1  # noqa: E203
            X[:, (i + 4) : (i + 8)] = q2  # noqa: E203
            X[:, (i + 8) : (i + 12)] = qmt.qmult(qmt.qinv(q1), q2)  # noqa: E203
            i += 12
        if dt is not None:
            X[:, -1] = dt * 10

        return X[:, None], qrel[:, None]
