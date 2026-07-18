"""Console-script entry points.

The CLI tools keep their if __name__ == "__main__": style, so these shims run
each module as __main__ (via runpy) to trigger that block — that's what the
gpu-flame / gpu-comm binaries call.
"""

import runpy


def gpu_flame() -> None:
    runpy.run_module("nsys_tools.tools.flamegraph", run_name="__main__")


def gpu_comm() -> None:
    runpy.run_module("nsys_tools.tools.gpu_comm", run_name="__main__")


def gpu_deepep_skew() -> None:
    runpy.run_module("nsys_tools.tools.gpu_deepep_skew", run_name="__main__")


def gpu_p2p_skew() -> None:
    runpy.run_module("nsys_tools.tools.gpu_p2p_skew", run_name="__main__")


def gpu_shape() -> None:
    runpy.run_module("nsys_tools.tools.kernel_shapes", run_name="__main__")


def gpu_exporter() -> None:
    runpy.run_module("nsys_tools.tools.exporter", run_name="__main__")


def gpu_groups() -> None:
    runpy.run_module("nsys_tools.tools.parallel_groups", run_name="__main__")


def sim_mcore_pp_sched() -> None:
    runpy.run_module("nsys_tools.tools.mcore_pp_timeline", run_name="__main__")


def sim_dual_pp_sched() -> None:
    runpy.run_module("nsys_tools.tools.dual_pp_time", run_name="__main__")
