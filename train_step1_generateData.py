from dataclasses import replace
from pathlib import Path
from typing import Optional

import fire
import jax
import ring
from ring.utils import randomize_sys


def setup_fn_factory(p: float):
    def setup_fn(key, sys: ring.System):
        children = sys.children(sys.find_body_to_world(name=True))
        bodies = sys.findall_bodies_with_jointtype("rr_imp")
        start = [i for i in bodies if i in children][0]

        joint_axes = sys.links.joint_params["rr_imp"]["joint_axes"]
        axis = joint_axes[start]

        bodies.remove(start)
        for i in bodies:
            key, consume = jax.random.split(key)
            duplicate = jax.random.bernoulli(consume, p=p)
            joint_axes = jax.lax.cond(
                duplicate, lambda i: joint_axes.at[i].set(axis), lambda i: joint_axes, i
            )

        sys.links.joint_params["rr_imp"]["joint_axes"] = joint_axes
        return sys.replace(links=sys.links.replace(joint_params=sys.links.joint_params))

    return setup_fn


def main(
    xml_path: str,
    size: int,
    configs: list[str] = ["standard"],
    seed: int = 1,
    output_path: Optional[str] = None,
    anchors: Optional[list[str]] = None,
    imu_motion_artifacts: bool = False,
    sampling_rates: list[float] = [100],
    p_duplicate_ja: float = 0.8,
):
    """Generate training data for RING and serialize to pickle file.

    Args:
        xml_path (str): path to xml model file, see folder `train_xmls`
        size (int): number of sequences to generate, each sequence is 60s
        configs (list[str]): which MotionConfigs to use, possible values are e.g.
            `langsam`, `standard`, `hinUndHer`, `expSlow`, `expFast`, ...
            Defaults to ["standard"].
        seed (int, optional): PRNG seed. Defaults to 1.
        output_path (str, optional): output path file, where to save the pickle file.
            By default tries to create the folder `ring_data` and creates the pickle
            file with a descriptive name inside this folder.
        anchors (Optional[list[str]], optional): the anchors to use, every non-IMU body
            in the xml model can act as an anchor that connects to the root worldbody.
            For example, for the two-segment kinematic chain both the body `seg3_2Seg`
            and the body `seg4_2Seg` could act as anchor. Defaults to None, which uses
            the system as it is defined in the xml model file.
        imu_motion_artifacts (bool, optional): wether or not to simualte non-rigidly
            attached IMUs. Defaults to False.
        sampling_rates (list[float]): sampling rates to use for generating data, in
            Hz (1 / second). Defaults to [100].
    """
    ring.setup(rr_imp_joint_kwargs=dict(ang_max_deg=3))

    sys = ring.System.create(xml_path)

    if output_path is None:
        folder = Path(__file__).parent.joinpath("ring_data")
        folder.mkdir(exist_ok=True)
        output_path = folder.joinpath(
            f"data_{sys.model_name}_{'-'.join(configs)}_Hz"
            + f"{'-'.join([str(int(s)) for s in sampling_rates])}_size{size}_seed{seed}"
            + ".pickle"
        )
    else:
        output_path = Path(output_path).with_suffix(".pickle")

    ring.RCMG(
        randomize_sys.randomize_anchors(sys, anchors) if anchors else sys,
        [replace(ring.MotionConfig.from_register(c), T=150.0) for c in configs],
        add_X_imus=True,
        add_X_jointaxes=True,
        add_X_jointaxes_kwargs=dict(randomly_flip=True),
        add_y_relpose=True,
        add_y_rootincl=True,
        dynamic_simulation=True,
        imu_motion_artifacts=imu_motion_artifacts,
        imu_motion_artifacts_kwargs=dict(
            prob_rigid=0.25,
            pos_min_max=0.05,
            all_imus_either_rigid_or_flex=True,
            disable_warning=True,
        ),
        randomize_joint_params=True,
        randomize_motion_artifacts=True,
        randomize_positions=True,
        randomize_hz=True,
        randomize_hz_kwargs=dict(sampling_rates=sampling_rates),
        cor=True,
        setup_fn=setup_fn_factory(p_duplicate_ja),
    ).to_pickle(output_path, size, seed, overwrite=False)

    print(f"Saved data at {str(output_path)}")


if __name__ == "__main__":
    fire.Fire(main)
