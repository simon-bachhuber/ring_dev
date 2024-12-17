from dataclasses import replace
import itertools
from typing import Optional

import fire
import jax
import jax.numpy as jnp
import ring
from ring.sys_composer.morph_sys import _autodetermine_new_parents
from ring.sys_composer.morph_sys import identify_system
from ring.sys_composer.morph_sys import Node
from ring.utils import dict_union
from ring.utils import randomize_sys

from train_step1_generateData_2S import _change_joint_type

dof = dict(rr=1, rsaddle=2, spherical=3, rr_imp=1)


def body_to_eps_rots(
    sys: ring.System,
    x: ring.Transform,
    sys_x: ring.System,
) -> dict[str, jax.Array]:
    # (time, nlinks, 4) -> (nlinks, time, 4)
    rots = x.rot.transpose((1, 0, 2))
    l_map = sys_x.idx_map("l")
    segments = sys.findall_segments()

    y = dict()

    def f(_, __, name: str):
        if name not in segments:
            return
        q_i = rots[l_map[name]]
        q_i = ring.maths.quat_inv(q_i)
        y[name] = q_i

    sys.scan(f, "l", sys.link_names)

    return y


def _lookup_new_index(structure: Node):
    "Returns link index in new indicies for joint params and dof"
    if not structure.parent_changed:
        return structure.link_idx_new_indices
    return structure.old_parent_new_indices


def finalize_fn_factory(sys_pred: ring.System):
    segs = set(sys_pred.findall_segments()) - set(
        [sys_pred.find_body_to_world(name=True)]
    )

    def finalize_fn(key, q, x, sys: ring.System):
        anchor = sys.find_body_to_world(name=True)
        structures, *_ = identify_system(
            sys_pred,
            _autodetermine_new_parents(
                sys_pred.link_parents,
                sys_pred.name_to_idx(anchor),
            ),
        )

        X = {"dt": jnp.array(sys.dt)}
        for seg in segs:
            structure = structures[sys_pred.name_to_idx(seg)]
            i = _lookup_new_index(structure)
            print(f"For `{seg}` and anchor `{anchor}` use link `{sys.idx_to_name(i)}`")
            X = dict_union(
                X,
                {
                    seg: dict(
                        dof=dof[sys.link_types[i]],
                        joint_params=sys.links[i].joint_params,
                        parent_changed=jnp.array(structure.parent_changed),
                    )
                },
            )

        y = body_to_eps_rots(sys_pred, x, sys)
        return X, y

    return finalize_fn


def main(
    xml_path: str,
    size: int,  # 32 * n_mconfigs * n_gens (= n_anchors * 3**(N-1)) * X
    output_path: str,
    configs: list[str] = ["standard", "expSlow", "expFast", "hinUndHer"],
    seed: int = 1,
    anchors: Optional[list[str]] = None,
    mot_art: bool = False,
    dyn_sim: bool = False,
    sampling_rates: list[float] = [40, 60, 80, 100, 120, 140, 160, 180, 200],
    T: float = 150.0,
):
    sys = ring.System.create(xml_path)

    syss = []
    segs = [s for s in sys.findall_segments() if s != sys.find_body_to_world(name=True)]
    print(segs)
    for dofs in list(itertools.product(*([[1, 2, 3]] * len(segs)))):
        _sys = sys
        for seg, dof in zip(segs, dofs):
            _sys = _change_joint_type(_sys, seg, dof)
        syss.extend(randomize_sys.randomize_anchors(_sys, anchors))
    print(len(syss))

    if mot_art:
        dyn_sim = True

    ring.RCMG(
        syss,
        [replace(ring.MotionConfig.from_register(c), T=T) for c in configs],
        add_X_imus=True,
        dynamic_simulation=dyn_sim,
        imu_motion_artifacts=mot_art,
        imu_motion_artifacts_kwargs=dict(
            prob_rigid=0.5,
            pos_min_max=0.05,
            all_imus_either_rigid_or_flex=True,
            disable_warning=True,
        ),
        randomize_joint_params=True,
        randomize_motion_artifacts=True,
        randomize_positions=True,
        randomize_hz=True,
        randomize_hz_kwargs=dict(sampling_rates=sampling_rates, add_dt=False),
        cor=True,
        finalize_fn=finalize_fn_factory(sys),
        sys_ml=sys,
    ).to_folder(output_path, size, seed, overwrite=False)


if __name__ == "__main__":
    fire.Fire(main)
