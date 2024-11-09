import fire
import jax.numpy as jnp
import numpy as np
import qmt
import ray
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ring
from ring.ml.ml_utils import MixinLogger
from ring.utils import dataloader_torch
from torch.utils.data import ConcatDataset
import wandb


class RayLogger(MixinLogger):
    def __init__(self):
        self._last_step = None
        self._reports = dict()

    def log_params(self, path):
        pass

    def log_command_output(self, command):
        pass

    def log_key_value(self, key, value, step=None):
        if self._last_step is None:
            self._last_step = step

        if step > self._last_step and len(self._reports) > 0:
            self._reports.update({"i_episode": self._last_step})
            train.report(self._reports)
            self._last_step = step
            self._reports = dict()

        self._reports.update({key: value})


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


def _params(unique_id: str = ring.ml.unique_id()) -> str:
    home = "/home/woody/iwb3/iwb3004h/simon/" if ring.ml.on_cluster() else "~/"
    return home + f"params/{unique_id}.pickle"


def main(
    paths: str,
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
    dof: bool = False,
    rand_ori: bool = False,
    tbp: int = 1000,
    pos: bool = False,
    use_vqf: bool = False,
    loggers=[],
    adap_clip=None,
    glob_clip=1.0,
):
    np.random.seed(seed)

    if use_wandb:
        unique_id = ring.ml.unique_id()
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    transform = lambda dof: Transform(dof, rand_ori, pos, use_vqf)

    gen = dataloader_torch.dataset_to_generator(
        ConcatDataset(
            [
                dataloader_torch.FolderOfPickleFilesDataset(
                    p, transform(i + 1 if dof else None)
                )
                for i, p in enumerate(paths.split(","))
            ]
        ),
        batch_size=bs,
        seed=seed,
        num_workers=num_workers,
    )

    params = _params(hex(warmstart)) if warmstart else None
    celltype = "lstm" if lstm else "gru"

    net = ring.ml.RNNO(
        4,
        return_quats=True,
        eval=False,
        v1=True,
        rnn_layers=[rnn_w] * rnn_d,
        linear_layers=[lin_w] * lin_d,
        act_fn_rnn=lambda X: X,
        params=params,
        celltype=celltype,
        scale_X=False,
    ).unwrapped  # get ride of GroundTruthWrapper

    callbacks = []
    for i, p in enumerate(paths.split(",")):
        path = p + "_val"
        ds_val = dataloader_torch.FolderOfPickleFilesDataset(
            path, transform(i + 1 if dof else None)
        )
        X_val, y_val = dataloader_torch.dataset_to_generator(ds_val, len(ds_val))(None)
        callbacks.append(
            ring.ml.callbacks.EvalXyTrainingLoopCallback(
                net,
                dict(
                    mae_deg=lambda q, qhat: jnp.rad2deg(
                        jnp.mean(ring.maths.angle_error(q, qhat))
                    )
                ),
                X_val,
                y_val,
                None,
                path.split("/")[-1] + "_",
            )
        )
        if i == 0:
            T = X_val.shape[1]
            # print("T: ", T)

    opt = ring.ml.make_optimizer(
        lr, episodes, int(T / tbp), adap_clip=adap_clip, glob_clip=glob_clip
    )

    ring.ml.train_fn(
        gen,
        episodes,
        net,
        opt,
        # callback_kill_after_seconds=23.5 * 3600,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        seed_network=seed,
        callback_save_params=_params(),
        callbacks=callbacks,
        tbp=tbp,
        loggers=loggers,
    )


def ray_main(paths: str, n_gpus_per_job: int, walltime_hours: float):
    ray.init()
    num_cpus = ray.available_resources().get("CPU", 0)
    num_gpus = ray.available_resources().get("GPU", 0)
    n_jobs = num_gpus // n_gpus_per_job
    n_cpus_per_job = num_cpus // n_jobs

    def trainable(config):
        main(
            paths,
            bs=config.get("bs", 64),
            episodes=config.get("n_decay_episodes", 1500),
            rnn_w=config.get("rnn_w", 200),
            rnn_d=config.get("rnn_d", 2),
            lin_w=config.get("lin_w", 200),
            lin_d=config.get("lin_d", 2),
            seed=config.get("seed", 1),
            use_wandb=False,
            lr=config.get("lr", 1e-3),
            num_workers=int(n_cpus_per_job // 2),
            lstm=config.get("celltype", "gru") == "lstm",
            tbp=config.get("tbp", 1000),
            pos=config.get("use_pos", False),
            use_vqf=config.get("use_vqf", False),
            adap_clip=config.get("adap_clip", 100),
            glob_clip=config.get("glob_clip", 1.0),
            loggers=[RayLogger()],
        )

    trainable_with_resources = tune.with_resources(
        trainable, {"gpu": n_gpus_per_job, "cpu": n_cpus_per_job}
    )
    param_space = {
        "bs": tune.choice([16, 32, 64, 128, 256]),
        "n_decay_episodes": tune.randint(100, 10000),
        "rnn_w": tune.choice([1, 2, 3]),
        "rnn_d": tune.choice([100, 200, 400, 800]),
        "lin_w": tune.choice([1, 2, 3]),
        "lin_d": tune.choice([100, 200, 400, 800]),
        "seed": tune.randint(0, 1000),
        "lr": tune.loguniform(1e-5, 1e-2),
        "celltype": tune.choice(["gru", "lstm"]),
        "tbp": tune.choice([300, 600, 1000, 2000, 6000]),
        "use_pos": tune.choice([True, False]),
        "use_vqf": tune.choice([True, False]),
        "adap_clip": tune.choice([0.1, 0.2, 0.5, 1.0, 100.0]),
        "glob_clip": tune.choice([0.1, 0.2, 0.5, 1.0, 100.0]),
    }

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            mode="min",
            metric="mae_deg_link0",
            time_budget_s=walltime_hours * 3600,
            num_samples=-1,
            scheduler=ASHAScheduler(
                "i_episode", "mae_deg_link0", "min", max_t=5000, grace_period=4
            ),
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    fire.Fire(ray_main)
