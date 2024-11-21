"""
srun python ring_dev/train_step1_generateData_2S.py 24576 $BIGWORK/data/10 "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 10 --dof1 1 --dof2 0
srun python ring_dev/train_step1_generateData_2S.py 24576 $BIGWORK/data/11 "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 11 --dof1 1 --dof2 1
srun python ring_dev/train_step1_generateData_2S.py 24576 $BIGWORK/data/12 "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 12 --dof1 1 --dof2 2
srun python ring_dev/train_step1_generateData_2S.py 24576 $BIGWORK/data/13 "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 13 --dof1 1 --dof2 3
srun python ring_dev/train_step1_generateData_2S.py 24576 $BIGWORK/data/22 "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 22 --dof1 2 --dof2 2
srun python ring_dev/train_step1_generateData_2S.py 24576 $BIGWORK/data/23 "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 23 --dof1 2 --dof2 3

srun python ring_dev/train_step1_generateData_2S.py 96 $BIGWORK/data/10_val "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 101 --dof1 1 --dof2 0
srun python ring_dev/train_step1_generateData_2S.py 96 $BIGWORK/data/11_val "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 111 --dof1 1 --dof2 1
srun python ring_dev/train_step1_generateData_2S.py 96 $BIGWORK/data/12_val "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 121 --dof1 1 --dof2 2
srun python ring_dev/train_step1_generateData_2S.py 96 $BIGWORK/data/13_val "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 131 --dof1 1 --dof2 3
srun python ring_dev/train_step1_generateData_2S.py 96 $BIGWORK/data/22_val "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 221 --dof1 2 --dof2 2
srun python ring_dev/train_step1_generateData_2S.py 96 $BIGWORK/data/23_val "['standard', 'standard-S', 'standard-S+', 'expSlow', 'expSlow-S', 'expSlow-S+', 'expFast', 'expFast-S', 'expFast-S+', 'hinUndHer', 'hinUndHer-S', 'hinUndHer-S+']" 231 --dof1 2 --dof2 3
"""

from dataclasses import replace
from typing import Optional

import fire
import numpy as np
import ring
from ring.utils import randomize_sys

sys_str1 = """
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

sys_str2 = """
<x_xy model="lam2">
  <options dt="0.01" gravity="0.0 0.0 9.81"/>
  <worldbody>
    <body joint="free" name="seg1" pos="0.4 0.0 0.0" pos_min="0.2 -0.05 -0.05" pos_max="0.55 0.05 0.05" damping="5.0 5.0 5.0 25.0 25.0 25.0">
      <geom pos="0.1 0.0 0.0" mass="1.0" color="dustin_exp_blue" edge_color="black" type="box" dim="0.2 0.05 0.05"/>
      <geom pos="0.05 0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
      <geom pos="0.15 -0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
      <body joint="frozen" name="imu1" pos="0.099999994 0.0 0.035" pos_min="0.050000012 -0.05 -0.05" pos_max="0.15 0.05 0.05">
        <geom mass="0.1" color="dustin_exp_orange" edge_color="black" type="box" dim="0.05 0.03 0.02"/>
      </body>
      <body joint="rr_imp" name="seg2" pos="0.20000002 0.0 0.0" pos_min="0.0 -0.05 -0.05" pos_max="0.35 0.05 0.05" damping="3.0 3.0">
        <geom pos="0.1 0.0 0.0" mass="1.0" color="dustin_exp_white" edge_color="black" type="box" dim="0.2 0.05 0.05"/>
        <geom pos="0.1 0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
        <geom pos="0.15 -0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
        <body joint="frozen" name="imu2" pos="0.100000024 0.0 0.035" pos_min="0.050000012 -0.05 -0.05" pos_max="0.14999998 0.05 0.05">
          <geom mass="0.1" color="dustin_exp_orange" edge_color="black" type="box" dim="0.05 0.03 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</x_xy>
"""  # noqa: E501

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
    dof1: int = None,
    dof2: int = None,
    dyn_sim: bool = False,
):
    sys = ring.System.create(sys_str2)

    if dof1 is not None:
        sys = sys.change_joint_type(
            "seg1", dof_joint_types[dof1], new_damp=dof_joint_dampings[dof1]
        )
    if dof2 is not None:
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
        dynamic_simulation=dyn_sim,
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
