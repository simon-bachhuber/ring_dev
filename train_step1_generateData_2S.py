from dataclasses import replace
from typing import Optional

import fire
import numpy as np
import ring
from ring.utils import randomize_sys

sys_str = """
<x_xy model="lam2">
  <options dt="0.01" gravity="0.0 0.0 9.81"/>
  <worldbody>
    <body joint="free" name="floatBase" damping="5.0 5.0 5.0 25.0 25.0 25.0">
      <body joint="frozen" name="seg1">
        <geom mass="1.0" type="box" dim=".2 .2 .2"/>
        <body joint="frozen" name="imu1" pos_min="-.3 -.3 -.3" pos_max=".3 .3 .3">
          <geom mass="0.1" type="box" dim=".2 .2 .2"/>
        </body>
      </body>
      <body joint="spherical" name="seg2" damping="5.0 5.0 5.0">
        <geom mass="1.0" type="box" dim=".2 .2 .2"/>
        <body joint="frozen" name="imu2" pos_min="-.3 -.3 -.3" pos_max=".3 .3 .3">
          <geom mass="0.1" type="box" dim=".2 .2 .2"/>
        </body>
      </body>
    </body>
  </worldbody>
</x_xy>
"""

dof_joint_types = {0: "frozen", 1: "rr", 2: "rsaddle", 3: "spherical"}
dof_joint_dampings = {
    0: np.array([]),
    1: np.array([3.0]),
    2: np.array([3.0, 3.0]),
    3: np.array([5.0, 5.0, 5.0]),
}


def finalize_fn(key, q, x, sys: ring.System):
    idx_map = sys.idx_map("l")
    X, y = {
        f"seg{i}": dict(imu_to_joint_m=-sys.links.transform1.pos[idx_map[f"imu{i}"]])
        for i in [1, 2]
    }, dict()
    return X, y


def main(
    size: int,
    output_path: str,
    configs: list[str] = ["standard", "expSlow", "expFast", "hinUndHer"],
    seed: int = 1,
    anchors: Optional[list[str]] = None,
    # sampling_rates: list[float] = [40, 60, 80, 100, 120, 140, 160, 180, 200],
    T: float = 60.0,  # 150
    motion_arti: bool = False,
    dof1: int = 0,
    dof2: int = 3,
):
    sys = ring.System.create(sys_str)
    sys = sys.change_joint_type(
        "seg1", dof_joint_types[dof1], new_damp=dof_joint_dampings[dof1]
    )
    sys = sys.change_joint_type(
        "seg2", dof_joint_types[dof2], new_damp=dof_joint_dampings[dof2]
    )

    ring.RCMG(
        randomize_sys.randomize_anchors(sys, anchors) if anchors else sys,
        [replace(ring.MotionConfig.from_register(c), T=T) for c in configs],
        add_X_imus=True,
        add_y_relpose=True,
        add_y_rootfull=True,
        add_y_rootfull_kwargs=dict(child_to_parent=True),
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
