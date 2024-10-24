from diodem.benchmark import benchmark
from diodem.benchmark import IMTP
from ring import ml


def make_exp_callbacks(
    ringnet,
    seg1=True,
    seg2=True,
    seg4=True,
    seg2_ja: bool = True,
    seg2_flex: bool = True,
    seg2_dt: bool = True,
):

    assert not ((seg1 is False) and (seg2 is False) and (seg4 is False))

    callbacks, metrices_name = [], []

    def add_callback(
        imtp: IMTP, exp_id, motion_start, include_in_expval=True, twice=False
    ):
        cb = benchmark(
            imtp=imtp,
            exp_id=exp_id,
            motion_start=motion_start,
            filter=ringnet,
            return_cb=True,
        )
        callbacks.append(cb)
        if include_in_expval:
            for segment in imtp.segments:
                for _ in range((2 if twice else 1)):
                    metrices_name.append([cb.metric_identifier, "mae_deg", segment])

    # 1SEG exp callbacks
    timings = {
        2: ["slow_fast_mix", "slow_fast_freeze_mix"],
    }
    if seg1:
        for anchor_1Seg in ["seg1", "seg2", "seg3", "seg4", "seg5"]:
            for exp_id in timings:
                for phase in timings[exp_id]:
                    add_callback(
                        IMTP([anchor_1Seg], model_name_suffix=f"_{anchor_1Seg}"),
                        exp_id,
                        phase,
                    )

    # 4SEG exp callbacks
    timings = {
        1: ["slow1", "fast"],
        2: ["slow_fast_mix", "slow_fast_freeze_mix"],
    }
    if seg4:
        for exp_id in timings:
            for phase in timings[exp_id]:
                add_callback(
                    IMTP(
                        ["seg2", "seg3", "seg4", "seg5"], joint_axes=True, sparse=True
                    ),
                    exp_id,
                    phase,
                    twice=True,
                )

    # 2 Seg with flexible IMUs callbacks
    axes_S_06_07 = {
        "xaxis": ("seg2", "seg3"),
        "yaxis": ("seg3", "seg4"),
        "zaxis": ("seg4", "seg5"),
    }
    axes = {
        1: axes_S_06_07,
        2: axes_S_06_07,
        10: {"left": ("seg1", "seg5"), "right": ("seg3", "seg4")},
    }

    timings.update({10: ["gait_slow", "gait_fast"]})
    if seg2:
        for exp_id in timings:
            for phase in timings[exp_id]:
                for axis in axes[exp_id]:
                    add_callback(
                        IMTP(
                            list(axes[exp_id][axis]),
                            flex=seg2_flex,
                            joint_axes=seg2_ja,
                            model_name_suffix="_" + axis,
                            joint_axes_field=seg2_ja,
                            dt=seg2_dt,
                        ),
                        exp_id,
                        phase,
                    )

    # create one large "experimental validation" metric
    for zoom_in in metrices_name:
        print(zoom_in)
    callbacks += [ml.callbacks.AverageMetricesTLCB(metrices_name, "exp_val_mae_deg")]

    return callbacks
