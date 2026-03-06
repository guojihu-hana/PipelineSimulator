import random
import cProfile
import pstats
from simulator.abstract.mutils import TrainingConfig
from simulator.abstract.variables import Schedule
from simulator.abstract.context import global_context as gpc
from Executor import Executor


def preprocess_head_times(times, vocab_parallel=False):
    if not vocab_parallel:
        return times
    head_plus_last_layer = times[-1]
    last_layer = times[-2]
    head_time = head_plus_last_layer - last_layer
    times.pop()
    times.append(last_layer)
    times.append(head_time / gpc["DEVICE_NUM"])
    return times


def main():
    random.seed(1024)
    chunk_num = gpc["LAYER_NUM"] // gpc["DEVICE_NUM"]
    chunk_num = 1
    schedule_method = Schedule.OctoPipe

    bwd_split = True
    if schedule_method == Schedule.STANDARD_ZBH:
        bwd_split = True
    if bwd_split:
        gpc["B_TIMES"] = [time * 0.5 for time in gpc["B_TIMES"]]
        gpc["W_TIMES"] = [time for time in gpc["B_TIMES"]]
    if schedule_method == Schedule.STANDARD_INTERLEAVED:
        chunk_num = gpc["LAYER_NUM"] // gpc["DEVICE_NUM"]

    vocab_parallel = True
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
        layer_num=gpc["LAYER_NUM"],
        chunk_num=chunk_num,
        device_num=gpc["DEVICE_NUM"],
        micro_batch_num=8,
        micro_batch_size=gpc["MICRO_BATCH_SIZE"],
        layer_f_times=preprocess_head_times(gpc["F_TIMES"], vocab_parallel),
        layer_b_times=preprocess_head_times(gpc["B_TIMES"], vocab_parallel),
        layer_w_times=preprocess_head_times(gpc["W_TIMES"], vocab_parallel),
    )
    executor = Executor(
        schedule_method=schedule_method,
        tc=tc,
        nmb_per_dp=[4 * gpc["DEVICE_NUM"]],
        device_comp_power=[[1 for _ in range(gpc["DEVICE_NUM"])] for _ in range(1)]
    )
    iter_tuning = False

    if iter_tuning:
        if schedule_method == Schedule.OctoPipe:
            if gpc["PROFILE_GENERATION"]:
                profiler = cProfile.Profile()
                profiler.enable()
                executor.iterative_tuning(iter_limit=100)
                profiler.disable()

                stats = pstats.Stats(profiler).sort_stats("cumtime")
                stats.print_stats(20)  # 打印前 10 个耗时函数
            else:
                # tune_strategy 0: octopipe, 1:random, 2:dfs
                executor.iterative_tuning(iter_limit=50, tune_strategy=0)
        else:
            executor.run_all_dp()
    else:
        partition = [1 for _ in range(gpc["LAYER_NUM"])]
        placement = [[i + j * gpc["DEVICE_NUM"] for j in range(gpc["LAYER_NUM"]//gpc["DEVICE_NUM"]) ] for i in range(gpc["DEVICE_NUM"])]

        executor.one_step_tuning(time_limit=100000, placement=placement, partition=partition)

    executor.draw()


if __name__ == "__main__":
    main()
