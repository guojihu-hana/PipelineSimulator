import random
import cProfile
import pstats
from simulator.abstract.mutils import TrainingConfig
from simulator.abstract.variables import Schedule
from simulator.abstract.context import (
    global_context as gpc,
    apply_stage_time_profile,
    apply_device_config,
    refresh_run_output_paths,
)
from Executor import Executor
import math

def preprocess_head_times(times, arch=None, *, vocab_parallel: bool = False, device_num: int = None, scale: int = 100):
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
    filled = [math.ceil(t*scale) for t in filled]
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
    new_times = [math.ceil(t*scale) for t in new_times]
    extra_h = [math.ceil(t*scale) for t in extra_h]
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

    vocab_parallel = False if schedule_method == Schedule.OctoPipe else False
    arch = gpc.get("ARCH")
    scale = 10
    f_times = preprocess_head_times(gpc["F_TIMES"], arch, vocab_parallel=vocab_parallel, scale=scale)
    b_times = preprocess_head_times(gpc["B_TIMES"], arch, vocab_parallel=vocab_parallel, scale=scale)
    w_times = preprocess_head_times(gpc["W_TIMES"], arch, vocab_parallel=vocab_parallel, scale=scale)
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
        discrete_event_time=False,
        schedule_method=schedule_method,
        tc=tc,
        nmb_per_dp=[4 * gpc["DEVICE_NUM"]],
        device_comp_power=[[1 for _ in range(gpc["DEVICE_NUM"])] for _ in range(1)]
    )
    iter_tuning = True if schedule_method == Schedule.OctoPipe else False
    gpc["PROFILE_GENERATION"] = True
    if iter_tuning:
        # ------------------------------------------------------------------
        # OctoPipe placement search — all knobs for Executor.iterative_tuning
        # (Phase 1–3). Larger search ≈ longer runtime.
        # ------------------------------------------------------------------
        # Phase 1: candidate generation (fast estimator + balance heuristics)
        # Micro-batch count for the fast estimator is tc.micro_batch_num (TrainingConfig above).
        tune_beam_width = 4096
        # Caps random strategies B/D/E inside tuning.solve_placement_guided (min with 200/300).
        tune_top_n = 2048
        # After ranking, keep this many distinct placements for Phase 2 list head.
        tune_load_relax_for_layout = 0.06
        # Strategy F: allow max-device static compute up to (1+this)×global min to add layout diversity.
        tune_balance_break_tries = 400
        # Total random swap attempts from near–load-optimal bases (spread across bases).

        # Phase 2: full simulator on first tune_sim_k placements from Phase 1.
        tune_sim_k = 64

        # Phase 3: neighbour pool / local search
        tune_local_search_rounds = 3
        # Hill-climb rounds per _local_search call.
        tune_ls_neighbor_cap = None
        # Max neighbours scored with fast estimator per round; None = use all (slowest, broadest).
        tune_ls_full_sim_top_k_min = 3
        tune_ls_full_sim_top_k_max = 15
        tune_ls_full_sim_top_k_divisor = 10
        # Full sim on top_k neighbours: top_k = max(min, min(max, len//divisor)).
        tune_bubble_heuristic_static_scale = 0.25
        # Bubble layout score: pressure = bubble_ratio - scale×(static_load/max_static).

        # Phase 3: adaptive sim truncation (also after each new global best)
        tune_sim_cap_multiplier = 5
        # sim_cap = best_T × multiplier; placements worse than cap can stop early.

        # Phase 3: ILS kicks & random restarts
        tune_ils_kick_multiplier = 5
        # Number of main kicks ≈ tune_local_search_rounds × this.
        tune_ils_patience = 5
        # Stop kicking after this many consecutive kicks without improving best_T.
        tune_ils_random_restarts = None
        # Random adjacency restarts; None = min(5, tune_local_search_rounds).
        tune_ils_random_restart_kicks = 3
        # Extra kicks after each random restart basin.

        tune_perturb_loose_balance_slack = 0.07
        # Alternate kick: random swaps may raise max-device static load by up to this fraction.
        tune_perturb_swap_min = 3
        tune_perturb_swap_max = None
        # If None: max swaps = max(tune_perturb_swap_max_floor, num_stages // tune_perturb_swap_stage_divisor).
        tune_perturb_swap_max_floor = 5
        tune_perturb_swap_stage_divisor = 4

        # Misc
        tune_time_limit = gpc["TIME_LIMIT"]
        # Single full-simulation step limit (Phase 2 and final one_step_tuning).
        tune_iter_limit = 100
        # Reserved (legacy API); new three-phase tuner does not loop on this.

        tune_kw = dict(
            iter_limit=tune_iter_limit,
            time_limit=tune_time_limit,
            beam_width=tune_beam_width,
            top_n=tune_top_n,
            local_search_rounds=tune_local_search_rounds,
            sim_k=tune_sim_k,
            ls_neighbor_cap=tune_ls_neighbor_cap,
            load_relax_for_layout=tune_load_relax_for_layout,
            balance_break_tries=tune_balance_break_tries,
            sim_cap_multiplier=tune_sim_cap_multiplier,
            ls_full_sim_top_k_min=tune_ls_full_sim_top_k_min,
            ls_full_sim_top_k_max=tune_ls_full_sim_top_k_max,
            ls_full_sim_top_k_divisor=tune_ls_full_sim_top_k_divisor,
            bubble_heuristic_static_scale=tune_bubble_heuristic_static_scale,
            ils_kick_multiplier=tune_ils_kick_multiplier,
            ils_patience=tune_ils_patience,
            ils_random_restarts=tune_ils_random_restarts,
            ils_random_restart_kicks=tune_ils_random_restart_kicks,
            perturb_loose_balance_slack=tune_perturb_loose_balance_slack,
            perturb_swap_min=tune_perturb_swap_min,
            perturb_swap_max=tune_perturb_swap_max,
            perturb_swap_max_floor=tune_perturb_swap_max_floor,
            perturb_swap_stage_divisor=tune_perturb_swap_stage_divisor,
        )

        if schedule_method == Schedule.OctoPipe:
            partition = [1 for _ in range(gpc["LAYER_NUM"])]
            placement = [[i + j * gpc["DEVICE_NUM"] for j in range(gpc["LAYER_NUM"]//gpc["DEVICE_NUM"]) ] for i in range(gpc["DEVICE_NUM"])]
            if gpc["PROFILE_GENERATION"]:
                profiler = cProfile.Profile()
                profiler.enable()
                executor.iterative_tuning(
                    placement=placement, partition=partition, verbose=True, **tune_kw)
                profiler.disable()

                stats = pstats.Stats(profiler).sort_stats("cumtime")
                stats.print_stats(20)  # 打印前 10 个耗时函数
            else:
                # tune_strategy 0: octopipe, 1:random, 2:dfs
                executor.iterative_tuning(
                    placement=placement, partition=partition, verbose=True, **tune_kw)
        else:
            executor.run_all_dp()
    else:
        partition = [1 for _ in range(gpc["LAYER_NUM"])]
        placement = [[i + j * gpc["DEVICE_NUM"] for j in range(gpc["LAYER_NUM"]//gpc["DEVICE_NUM"]) ] for i in range(gpc["DEVICE_NUM"])]
        # partition = [16, 16, 16, 8]
        # partition = [10, 10, 10, 10]
        # placement = [[i] for i in range(gpc["DEVICE_NUM"])]
        if gpc["PROFILE_GENERATION"]:
            profiler = cProfile.Profile()
            profiler.enable()
            executor.one_step_tuning(time_limit=100000, placement=placement, partition=partition, verbose=True)
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats("cumtime")
            stats.print_stats(20)  # 打印前 10 个耗时函数
        else:
            executor.one_step_tuning(time_limit=100000, placement=placement, partition=partition, verbose=True)

    DRAW = True
    if DRAW:
        executor.draw()


if __name__ == "__main__":
    main()
