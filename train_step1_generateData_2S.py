from dataclasses import replace
from typing import Optional

import fire
import ring
from ring.utils import randomize_sys


def main(
    xml_path: str,
    size: int,
    output_path: str,
    configs: list[str] = ["standard", "expSlow", "expFast", "hinUndHer"],
    seed: int = 1,
    anchors: Optional[list[str]] = ["seg1", "seg2"],
    sampling_rates: list[float] = [40, 60, 80, 100, 120, 140, 160, 180, 200],
    T: float = 150.0,
):
    sys = ring.System.create(xml_path)

    ring.RCMG(
        randomize_sys.randomize_anchors(sys, anchors) if anchors else sys,
        [replace(ring.MotionConfig.from_register(c), T=T) for c in configs],
        add_X_imus=True,
        add_y_relpose=True,
        dynamic_simulation=True,
        imu_motion_artifacts=True,
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
    ).to_folder(output_path, size, seed, overwrite=False)


if __name__ == "__main__":
    fire.Fire(main)
