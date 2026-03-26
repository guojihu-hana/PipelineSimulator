import random
import cProfile
import pstats
from re import T
from simulator.abstract.mutils import TrainingConfig
from simulator.abstract.variables import Schedule
from simulator.abstract.context import (
    global_context as gpc,
    apply_stage_time_profile,
    apply_device_config,
    refresh_run_output_paths,
)
from Executor import Executor


def preprocess_head_times(times, arch=None, *, vocab_parallel: bool = False, device_num: int = None):
    """
    Given per-layer times and an architecture/type list (arch),
    estimate per-(sub)type average time and return a new list where each layer's
    time is replaced by the sum of its component type averages.

    Important semantics for composite types like \"M+E\":
      arch[i] = \"M+E\" means the observed time times[i] is the *sum* of one M part
      and one E part for that layer: times[i] = M + E.
    Therefore we estimate base types from \"pure\" occurrences first (e.g. \"M\"),
    then infer the remaining component by subtraction (E = times[i] - avg(M)).
    \"-+H\" is handled similarly: H = times[i] - avg(\"-\").
    """
    if device_num is None:
        device_num = gpc["DEVICE_NUM"]
    if arch is None:
        return list(times)
    if len(arch) != len(times):
        raise ValueError(f"len(arch)={len(arch)} must equal len(times)={len(times)}")
    def _parts(a: str):
        return [p.strip() for p in str(a).split("+") if p.strip()]

    # 1) Estimate averages for pure types (no '+')
    sums = {}
    cnts = {}
    for t, a in zip(times, arch):
        ps = _parts(a)
        if len(ps) == 1:
            p = ps[0]
            sums[p] = sums.get(p, 0.0) + float(t)
            cnts[p] = cnts.get(p, 0) + 1
    avgs = {p: (sums[p] / cnts[p]) for p in sums}

    # 2) Iteratively infer unknown components from 2-part composites.
    # Example: "M+E" with known avg(M) => sample(E) = t - avg(M)
    inferred = True
    while inferred:
        inferred = False
        add_sums = {}
        add_cnts = {}
        for t, a in zip(times, arch):
            ps = _parts(a)
            if len(ps) != 2:
                continue
            p0, p1 = ps
            t = float(t)
            if p0 in avgs and p1 not in avgs:
                sample = t - avgs[p0]
                add_sums[p1] = add_sums.get(p1, 0.0) + sample
                add_cnts[p1] = add_cnts.get(p1, 0) + 1
            elif p1 in avgs and p0 not in avgs:
                sample = t - avgs[p1]
                add_sums[p0] = add_sums.get(p0, 0.0) + sample
                add_cnts[p0] = add_cnts.get(p0, 0) + 1
        for p in add_sums:
            if add_cnts[p] > 0 and p not in avgs:
                avgs[p] = add_sums[p] / add_cnts[p]
                inferred = True
    
    # 3) Fill per-layer by summing component averages when available; otherwise fall back to observed.
    filled = []
    for t, a in zip(times, arch):
        ps = _parts(a)
        if all(p in avgs for p in ps):
            filled.append(sum(avgs[p] for p in ps))
        else:
            filled.append(float(t))

    if not vocab_parallel:
        return filled

    # vocab_parallel: split "+H" out as a standalone extra layer.
    # Example: "-+H" at the last position becomes "-" and an additional "H" layer is appended.
    new_times = []
    extra_h = []
    for t, a in zip(filled, arch):
        ps = _parts(a)
        if "H" in ps and len(ps) == 2:
            other = ps[0] if ps[1] == "H" else ps[1]
            # Base part stays in-place; H becomes a new layer.
            base_t = avgs.get(other, None)
            h_t = avgs.get("H", None)
            if base_t is None and h_t is not None:
                base_t = float(t) - h_t
            if h_t is None and base_t is not None:
                h_t = float(t) - base_t
            if base_t is None:
                base_t = float(t)
            if h_t is None:
                h_t = 0.0
            new_times.append(base_t)
            extra_h.append(h_t / float(device_num))
        else:
            new_times.append(float(t))
    return new_times + extra_h


def main():
    random.seed(1024)
    # Change this value to control number of devices used in the simulator.
    DEVICE_NUM = 4

    # Must set gpc["SCHEDULE_METHOD"] before apply_device_config so CHUNK_NUM / paths match the run.
    schedule_method = Schedule.STANDARD_1F1B
    # schedule_method = Schedule.STANDARD_ZBH
    schedule_method = Schedule.STANDARD_INTERLEAVED
    schedule_method = Schedule.OctoPipe
    # schedule_method = Schedule.Mist
    gpc["SCHEDULE_METHOD"] = schedule_method

    apply_device_config(DEVICE_NUM)

    # Show available profiled models to help choose MODEL_NAME.
    try:
        from data.profiled_data import stage_time
        supported_models = ", ".join(sorted(stage_time.keys()))
        print(f"Supported profiled models: {supported_models}")
    except Exception:
        print("[WARN] Failed to load profiled model list from data.profiled_data.")

    # Change this single parameter to switch models.
    MODEL_NAME = "nemotron-nano-v2-9B"
    # MODEL_NAME = "deepseek-16B"
    # MODEL_NAME = "gpt-13B"
    # MODEL_NAME = "None"
    apply_stage_time_profile(MODEL_NAME)

    bwd_split = False
    if schedule_method == Schedule.STANDARD_ZBH:
        bwd_split = True
    elif schedule_method != Schedule.OctoPipe:
        bwd_split = False
    if bwd_split:
        gpc["B_TIMES"] = [time * 0.5 for time in gpc["B_TIMES"]]
        gpc["W_TIMES"] = [time for time in gpc["B_TIMES"]]
    if schedule_method == Schedule.STANDARD_INTERLEAVED:
        chunk_num = gpc["LAYER_NUM"] // gpc["DEVICE_NUM"]
    else:
        chunk_num = 1

    vocab_parallel = True if schedule_method == Schedule.OctoPipe else False
    arch = gpc.get("ARCH")
    f_times = preprocess_head_times(gpc["F_TIMES"], arch, vocab_parallel=vocab_parallel)
    b_times = preprocess_head_times(gpc["B_TIMES"], arch, vocab_parallel=vocab_parallel)
    w_times = preprocess_head_times(gpc["W_TIMES"], arch, vocab_parallel=vocab_parallel)
    layer_num = len(f_times)
    tc = TrainingConfig(
        pp_size=gpc["DEVICE_NUM"],
        tp_size=1,
        dp_size=1,
        bwd_split=bwd_split,
        vocab_parallel  =   vocab_parallel,
        overlap_aware   =   gpc["OVERLAP_AWARE_SCHEDULE"],
        save_memory     =   gpc["SAVE_MEMORY"],
        constrain_warmup    =   gpc["CONSTRAIN_WARMUP"],
        swith_workload_type =   gpc["SWITCH_WORKLOAD_TYPE"],
        layer_num=layer_num - 1 if vocab_parallel else layer_num, # vocab_parallel: split "+H" out as a standalone extra layer.
        chunk_num=chunk_num,
        device_num=gpc["DEVICE_NUM"],
        micro_batch_num=gpc["DEVICE_NUM"] * 2,
        micro_batch_size=gpc["MICRO_BATCH_SIZE"],
        layer_f_times=f_times,
        layer_b_times=b_times,
        layer_w_times=w_times,
    )

    # Align artifact folder ``mb{}`` with the TrainingConfig used in this run.
    refresh_run_output_paths(gpc, micro_batch_num=tc.micro_batch_num)

    executor = Executor(
        schedule_method=schedule_method,
        tc=tc,
        nmb_per_dp=[4 * gpc["DEVICE_NUM"]],
        device_comp_power=[[1 for _ in range(gpc["DEVICE_NUM"])] for _ in range(1)]
    )
    iter_tuning = True if schedule_method == Schedule.OctoPipe else False
    gpc["PROFILE_GENERATION"] = True
    if iter_tuning:
        if schedule_method == Schedule.OctoPipe:
            partition = [1 for _ in range(gpc["LAYER_NUM"])]
            placement = [[i + j * gpc["DEVICE_NUM"] for j in range(gpc["LAYER_NUM"]//gpc["DEVICE_NUM"]) ] for i in range(gpc["DEVICE_NUM"])]
            if gpc["PROFILE_GENERATION"]:
                profiler = cProfile.Profile()
                profiler.enable()
                executor.iterative_tuning(iter_limit=100, beam_width=1024, top_n=2048, placement=placement, partition=partition, verbose=True, sim_k=64, ls_neighbor_cap=None)
                profiler.disable()

                stats = pstats.Stats(profiler).sort_stats("cumtime")
                stats.print_stats(20)  # 打印前 10 个耗时函数
            else:
                # tune_strategy 0: octopipe, 1:random, 2:dfs
                executor.iterative_tuning(iter_limit=100, beam_width=1024, top_n=2048, placement=placement, partition=partition, verbose=True, sim_k=64, ls_neighbor_cap=None)
        else:
            executor.run_all_dp()
    else:
        partition = [1 for _ in range(gpc["LAYER_NUM"])]
        placement = [[i + j * gpc["DEVICE_NUM"] for j in range(gpc["LAYER_NUM"]//gpc["DEVICE_NUM"]) ] for i in range(gpc["DEVICE_NUM"])]
        # partition = [16, 16, 16, 8]
        # partition = [10, 10, 10, 10]
        # placement = [[i] for i in range(gpc["DEVICE_NUM"])]
        executor.one_step_tuning(time_limit=100000, placement=placement, partition=partition, verbose=True)

    DRAW = True
    if DRAW:
        executor.draw()


if __name__ == "__main__":
    main()
