#!/usr/bin/env python3
"""
Compare wall-clock time of discrete-event time advance vs tick-by-tick update_time(),
and verify makespan T / per-device bubble (idle) match.

Run from repo root:
  python3 _bench_discrete_vs_tick_time.py
"""
from __future__ import annotations

import importlib.util
import statistics
import time
from pathlib import Path

from simulator.abstract.context import (
    global_context as gpc,
    apply_device_config,
    apply_stage_time_profile,
    refresh_run_output_paths,
)
from simulator.abstract.mutils import TrainingConfig
from simulator.abstract.variables import Schedule
from Executor import Executor

_ROOT = Path(__file__).resolve().parent


def _load_preprocess_head_times():
    spec = importlib.util.spec_from_file_location(
        "pipeline_sim___main__", _ROOT / "__main__.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.preprocess_head_times


preprocess_head_times = _load_preprocess_head_times()


def _build_tc_and_placement(*, scale: int, device_num: int, model_name: str):
    gpc["SCHEDULE_METHOD"] = Schedule.OctoPipe
    apply_device_config(device_num)
    apply_stage_time_profile(model_name)

    arch = gpc.get("ARCH")
    vocab_parallel = False
    f_times = preprocess_head_times(
        gpc["F_TIMES"], arch, vocab_parallel=vocab_parallel, scale=scale,
    )
    b_times = preprocess_head_times(
        gpc["B_TIMES"], arch, vocab_parallel=vocab_parallel, scale=scale,
    )
    w_times = preprocess_head_times(
        gpc["W_TIMES"], arch, vocab_parallel=vocab_parallel, scale=scale,
    )
    layer_num = len(f_times)
    micro_batch_num = 4 * device_num
    tc = TrainingConfig(
        pp_size=device_num,
        tp_size=1,
        dp_size=1,
        bwd_split=False,
        vocab_parallel=vocab_parallel,
        overlap_aware=gpc["OVERLAP_AWARE_SCHEDULE"],
        save_memory=gpc["SAVE_MEMORY"],
        constrain_warmup=gpc["CONSTRAIN_WARMUP"],
        swith_workload_type=gpc["SWITCH_WORKLOAD_TYPE"],
        layer_num=layer_num - 1 if vocab_parallel else layer_num,
        chunk_num=1,
        device_num=device_num,
        micro_batch_num=micro_batch_num,
        micro_batch_size=gpc["MICRO_BATCH_SIZE"],
        layer_f_times=f_times,
        layer_b_times=b_times,
        layer_w_times=w_times,
    )
    refresh_run_output_paths(gpc, micro_batch_num=micro_batch_num)

    partition = [1 for _ in range(layer_num)]
    placement = [
        [i + j * device_num for j in range(layer_num // device_num)]
        for i in range(device_num)
    ]
    assert layer_num % device_num == 0, (layer_num, device_num)
    return tc, placement, partition


def _simulate(
    tc: TrainingConfig,
    placement,
    partition,
    time_limit: int,
    *,
    discrete_event_time: bool,
):
    ex = Executor(
        Schedule.OctoPipe,
        tc,
        nmb_per_dp=[tc.micro_batch_num],
        device_comp_power=[[1 for _ in range(tc.device_num)]],
        discrete_event_time=discrete_event_time,
    )
    t0 = time.perf_counter()
    T, idle0 = ex._simulate_placement(placement, partition, time_limit)
    wall = time.perf_counter() - t0
    bubbles = tuple(idle0)
    exec_times = tuple(
        tuple(p.get_device_execution_time()) for p in ex.pipelines
    )
    return {
        "wall_s": wall,
        "T": T,
        "bubbles_dp0": bubbles,
        "finish": ex.finish_flag,
        "final_clock": ex.get_time(),
        "exec_times": exec_times,
    }


def _median_walls(run_fn, repeats: int = 3):
    return statistics.median([run_fn()["wall_s"] for _ in range(repeats)])


def main():
    device_num = 4
    model_name = "nemotron-nano-v2-9B"
    time_limit = int(gpc["TIME_LIMIT"])
    scales = (1, 10, 100)
    repeats = 3

    print("| scale | tick wall (s) | jump wall (s) | tick÷jump wall time |")
    print("|------:|--------------:|--------------:|-------------------:|")

    cmp_lines = []
    for scale in scales:
        tc, placement, partition = _build_tc_and_placement(
            scale=scale, device_num=device_num, model_name=model_name,
        )

        def run_tick():
            return _simulate(
                tc, placement, partition, time_limit, discrete_event_time=False,
            )

        def run_jump():
            return _simulate(
                tc, placement, partition, time_limit, discrete_event_time=True,
            )

        # Warm jump once (JIT / caches)
        run_jump()

        med_tick = _median_walls(run_tick, repeats)
        med_jump = _median_walls(run_jump, repeats)
        sp = med_tick / med_jump if med_jump > 0 else float("inf")

        print(f"| {scale} | {med_tick:.4f} | {med_jump:.4f} | {sp:.2f}× |")

        r_tick = run_tick()
        r_jump = run_jump()
        t_match = r_tick["T"] == r_jump["T"]
        b_match = r_tick["bubbles_dp0"] == r_jump["bubbles_dp0"]
        fin_match = r_tick["finish"] == r_jump["finish"]
        ex_match = r_tick["exec_times"] == r_jump["exec_times"]
        cmp_lines.append(
            f"scale={scale}: T match={t_match}, bubble(idle) match={b_match}, "
            f"finish match={fin_match}, per-pipeline finish_times match={ex_match}"
        )
        if not (t_match and b_match and fin_match and ex_match):
            cmp_lines.append(
                f"  tick: T={r_tick['T']} bubble={r_tick['bubbles_dp0']} "
                f"finish={r_tick['finish']} clock={r_tick['final_clock']}"
            )
            cmp_lines.append(
                f"  jump: T={r_jump['T']} bubble={r_jump['bubbles_dp0']} "
                f"finish={r_jump['finish']} clock={r_jump['final_clock']}"
            )

    print()
    print(
        "Legend: tick÷jump > 1 ⇒ event jump is faster on wall clock; "
        "< 1 ⇒ +1 tick advance is faster (small sims: scan cost dominates)."
    )
    print()
    print("Correctness (single run after medians, same tc/placement):")
    for line in cmp_lines:
        print(line)


if __name__ == "__main__":
    main()
