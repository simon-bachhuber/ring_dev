import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import fire
import numpy as np
import optax
import ring
from ring import ml
import wandb

import dataloader
from exp_cbs import make_exp_callbacks
import transform

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


def _make_ring(lam, params_warmstart: str | None, dry_run: bool):
    message_dim = 200 if not dry_run else 10
    hidden_state_dim = 400 if not dry_run else 20
    ringnet = ml.RING(
        lam=lam,
        message_dim=message_dim,
        hidden_state_dim=hidden_state_dim,
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
    wandb_name: str = None,
    params_warmstart: str = None,
    seed: int = 1,
    dry_run: bool = False,
    exp_cbs: bool = False,
    rand_imus: bool = False,
    dl_worker_count: int = 0,
    dl_backend: str = "eager",
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
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    if path_trained_params is None:
        path_trained_params = f"~/params/{ring.ml.unique_id()}.pickle"

    ringnet = _make_ring(lam, params_warmstart, dry_run)

    kwargs = {}
    if dl_backend != "eager":
        keyword = "num_workers" if dl_backend == "torch" else "worker_count"
        kwargs.update({keyword: dl_worker_count})
    generator = dataloader.make_generator(
        path_lam1,
        path_lam2,
        path_lam3,
        path_lam4,
        batch_size=bs,
        transform=transform.Transform(rand_imus),
        seed=seed,
        backend=dl_backend,
        **kwargs,
    )

    callbacks = make_exp_callbacks(ringnet) if exp_cbs else []

    if params_warmstart is None:
        optimizer = ml.make_optimizer(
            lr,
            episodes,
            n_steps_per_episode=6,
            skip_large_update_max_normsq=100.0,
        )
    else:
        warmup = 1 / 3
        optimizer = optax.lamb(
            optax.warmup_cosine_decay_schedule(
                1e-5,
                lr,
                6 * int(episodes * warmup),
                6 * episodes,
            ),
            eps=1e-5,
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
