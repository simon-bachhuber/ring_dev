import os

import fire
import ray
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ring
from ring.ml.ml_utils import MixinLogger

from train_step2_trainRing_2S_rnno import main


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

        if step is not None and step > self._last_step and len(self._reports) > 0:
            self._reports.update({"i_episode": self._last_step})
            train.report(self._reports)
            self._last_step = step
            self._reports = dict()

        self._reports.update({key: value})


def _params(unique_id: str = ring.ml.unique_id()) -> str:
    home = "/bigwork/nhkbbach/" if ring.ml.on_cluster() else "~/"
    return home + f"params/{unique_id}.pickle"


max_t = 10_000


def ray_main(
    paths: str,
    n_gpus_per_job: int,
    walltime_hours: float,
    tqdm: bool = False,
):
    if not tqdm:
        os.environ["TQDM_DISABLE"] = "1"

    ray.init()
    num_cpus = ray.available_resources().get("CPU", 0)
    num_gpus = ray.available_resources().get("GPU", 0)
    if num_gpus > 0:
        n_jobs = num_gpus // n_gpus_per_job
        n_cpus_per_job = num_cpus // n_jobs
    else:
        n_cpus_per_job = 4

    def trainable(config):
        main(
            paths,
            bs=config.get("bs", 4),
            n_decay_episodes=config.get("n_decay_episodes", 1500),
            rnn_w=config.get("rnn_w", 20),
            rnn_d=config.get("rnn_d", 2),
            lin_w=config.get("lin_w", 20),
            lin_d=config.get("lin_d", 0),
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
            episodes=max_t + 100,
            dof=config.get("dof", False),
            rand_ori=config.get("rand_ori", False),
        )

    trainable_with_resources = tune.with_resources(
        trainable, {"gpu": n_gpus_per_job, "cpu": n_cpus_per_job}
    )

    if ring.ml.on_cluster():
        param_space = {
            "bs": tune.choice([256]),
            "n_decay_episodes": tune.randint(4000, 11000),
            "rnn_d": tune.choice([1, 2, 3]),
            "rnn_w": tune.choice([600, 800, 1000]),
            "lin_d": tune.choice(
                [
                    0,
                    1,
                ]
            ),
            "lin_w": tune.choice([400, 600]),
            "seed": tune.randint(0, 1000),
            "lr": tune.loguniform(1e-5, 3e-3),
            "celltype": tune.choice(["gru", "lstm"]),
            "tbp": tune.choice([150, 300, 600, 1000]),
            "use_pos": tune.choice([True, False]),
            "use_vqf": tune.choice([True, False]),
            "adap_clip": tune.choice([0.2, 1.0, None]),
            "glob_clip": tune.choice([0.2, 1.0, None]),
            "layernorm": tune.choice([True, False]),
            "dof": tune.choice([False, True]),
            "rand_ori": tune.choice([False, True]),
        }
    else:
        param_space = {}

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            mode="min",
            metric="mae_deg_link0",
            time_budget_s=walltime_hours * 3600,
            num_samples=-1 if ring.ml.on_cluster() else 1,
            scheduler=ASHAScheduler(
                "i_episode",
                max_t=max_t if ring.ml.on_cluster() else 10,
                grace_period=50 if ring.ml.on_cluster() else 10,
            ),
            max_concurrent_trials=4,
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    fire.Fire(ray_main)
