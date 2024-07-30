from pathlib import Path
from typing import Optional

import fire
import ring
from ring import utils


def main(
    size: int,
    configs: list[str] = ["standard"],
    seed: int = 1,
    output_path: Optional[str] = None,
    anchors: Optional[list[str]] = None,
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
    sys = ring.System.create(Path(__file__).parent.joinpath("train_xmls/lam2.xml"))

    if output_path is None:
        folder = Path(__file__).parent.joinpath("ring_data")
        folder.mkdir(exist_ok=True)
        output_path = folder.joinpath(
            f"data_pos_{sys.model_name}_{'-'.join(configs)}_Hz"
            + f"{'-'.join([str(int(s)) for s in [100]])}_size{size}_seed{seed}"
            + ".pickle"
        )
    else:
        output_path = Path(output_path).with_suffix(".pickle")

    syss = utils.randomize_sys.randomize_anchors(sys, anchors) if anchors else [sys]
    _, attachment = syss[0].make_sys_noimu()

    def output_transform(data):
        (X, y), (_, _, xs, sys_xs) = data
        xs_eps_to_joint = xs.take(sys_xs.link_types.index("rr_imp"), 2)

        pos = {}
        for imu in sys.findall_imus():
            xs_eps_to_imu = xs.take(sys_xs.name_to_idx(imu), 2)
            pos_imu_to_joint = ring.algebra.transform_mul(
                xs_eps_to_joint, ring.algebra.transform_inv(xs_eps_to_imu)
            ).pos
            seg = attachment[imu]
            pos[seg] = pos_imu_to_joint

        y = utils.dict_union(
            utils.dict_to_nested(y, "quat"), utils.dict_to_nested(pos, "pos")
        )
        return X, y

    ring.RCMG(
        syss,
        [ring.MotionConfig.from_register(c) for c in configs],
        add_X_imus=True,
        add_y_relpose=True,
        add_y_rootincl=True,
        dynamic_simulation=True,
        randomize_joint_params=True,
        randomize_motion_artifacts=True,
        randomize_positions=True,
        cor=True,
        keep_output_extras=True,
        output_transform=output_transform,
    ).to_pickle(output_path, size, seed, overwrite=False)

    print(f"Saved data at {str(output_path)}")


if __name__ == "__main__":
    fire.Fire(main)
