from dataclasses import replace
from typing import Optional

import fire
import ring
from ring.utils import randomize_sys


def finalize_fn(key, q, x, sys: ring.System):
    idx_map = sys.idx_map("l")
    X, y = {
        f"seg{i}": dict(imu_to_joint_m=-sys.links.transform1.pos[idx_map[f"imu{i}"]])
        for i in [1, 2]
    }, dict()
    return X, y


def main(
    xml_path: str,
    size: int,
    output_path: str,
    configs: list[str] = ["standard", "expSlow", "expFast", "hinUndHer"],
    seed: int = 1,
    anchors: Optional[list[str]] = None,
    # sampling_rates: list[float] = [40, 60, 80, 100, 120, 140, 160, 180, 200],
    T: float = 60.0,  # 150
    motion_arti: bool = False,
):
    sys = ring.System.create(xml_path)

    ring.RCMG(
        randomize_sys.randomize_anchors(sys, anchors) if anchors else sys,
        [replace(ring.MotionConfig.from_register(c), T=T) for c in configs],
        add_X_imus=True,
        add_y_relpose=True,
        dynamic_simulation=True,
        imu_motion_artifacts=motion_arti,
        imu_motion_artifacts_kwargs=dict(
            prob_rigid=0.25,
            pos_min_max=0.05,
            all_imus_either_rigid_or_flex=True,
            disable_warning=True,
        ),
        randomize_joint_params=True,
        randomize_motion_artifacts=True,
        randomize_positions=True,
        # randomize_hz=True,
        # randomize_hz_kwargs=dict(sampling_rates=sampling_rates),
        cor=True,
        finalize_fn=finalize_fn,
    ).to_folder(output_path, size, seed, overwrite=False)


if __name__ == "__main__":
    fire.Fire(main)
