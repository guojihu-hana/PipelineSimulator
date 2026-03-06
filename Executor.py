from simulator.abstract.Pipeline import PipelineScheduler
from simulator.abstract.mutils import TrainingConfig, dict_to_2d_list
from simulator.abstract.Device import Device
from simulator.abstract.variables import Schedule, WorkloadType
from simulator.abstract.context import global_context as gpc
from simulator.painter import MultiPipelinePainter as MPP
from tuning import (
    time_line, time_recorder_decorator,
    check_placement_convergence, check_partition_convergence,
    balanced_transpose, serialize,
    tune_placement, tune_partition_random, tune_placement_random,
    tune_partition_dfs, tune_placement_dfs, tune_partition,
)
import random


class Executor:

    def __init__(self, schedule_method, tc: TrainingConfig, model_name: str = None, nmb_per_dp: list = None, device_comp_power: list[list] = None, device_mem: list[list] = None) -> None:
        self.time           = 0
        self.schedule_method = schedule_method
        self.model_name     = model_name,
        self.tc             = tc
        self.bwd_split      = tc.bwd_split
        self.finish_flag    = False
        self.dp_size        = tc.dp_size
        self.chunk_num      = tc.chunk_num
        self.device_num     = tc.device_num
        self.layer_num      = tc.layer_num
        self.stage_num      = self.chunk_num * self.device_num
        self.device_comp_power  = device_comp_power if device_comp_power else [[gpc["COMP_POWER"] for _ in range(self.device_num)] for _ in range(self.dp_size)]
        self.device_mem         = device_mem if device_mem else [[gpc["GPU_MAX_MEM"] for _ in range(self.device_num)] for _ in range(self.dp_size)]
        self.nmb_per_dp     = [tc.micro_batch_num for _ in range(self.dp_size)] if not nmb_per_dp else nmb_per_dp
        self.mid_offsets    = [0] + [sum(self.nmb_per_dp[:i]) for i in range(1, self.dp_size+1)]
        self.micro_batch_size = tc.micro_batch_size
        self.batch_size = sum(self.nmb_per_dp) * self.micro_batch_size
        self.best_pipelines = None

    def _init_pipelines(self, placement = None, partition=None):
        self.pipelines      = [
            PipelineScheduler(
                pipeline_idx=dp_idx,
                schedule_method=self.schedule_method,
                training_config=self.tc,
                placement=placement,
                partition=partition,
                mid_offset=self.mid_offsets[dp_idx],
                comp_power=self.device_comp_power[dp_idx],
                max_mem=self.device_mem[dp_idx],
                executor=self
            ) for dp_idx in range(self.dp_size)
        ]

    def update_time(self):
        self.time += 1

    def reset_time(self):
        self.time = 0

    def get_time(self):
        return self.time

    def get_total_workload_count(self):
        count = 0
        for pipeline in self.pipelines:
            count += pipeline.total_workload
        return count

    def update_constraints_across_dp(self, time):
        for pipeline in self.pipelines:
            for device in pipeline.devices:
                if device.current_workload and time >= device.current_workload.end_time:
                    finished_mid = device.current_workload.mid
                    for p in self.pipelines:
                        for d in p.devices:
                            if d.did == device.did or p.pid == pipeline.pid:
                                continue # only update constraints on other devices or pipelines
                            if finished_mid in d.held_mids:
                                d.update_constraints_within_device(time, constraint=device.current_workload)

    def partition_to_placement(self, partition):
        placement = []
        start = 0
        for size in partition:
            end = start + size
            placement.append(list(range(start, end)))
            start = end
        return placement

    def get_layer_comp_times(self, vocab_parallel=False):
        layer_comp_time = [f + b + w if not gpc["RECOMP"] else f + f + b + w for f, b, w in zip(gpc["F_TIMES"], gpc["B_TIMES"], gpc["W_TIMES"])]
        if vocab_parallel:
            layer_comp_time = layer_comp_time[:-1]
        layer_comp_time[-1] += gpc["HEAD_F_TIME"] + gpc["HEAD_B_TIME"] + gpc["HEAD_W_TIME"]
        if gpc["RECOMP"]:
            layer_comp_time[-1] += gpc["HEAD_F_TIME"]
        return  layer_comp_time

    @time_recorder_decorator(time_line)
    def one_step_tuning(self, time_limit, placement=None, partition=None):
        self._init_pipelines(placement=placement, partition=partition)
        self.reset_time()
        self.finish_flag = False
        while self.get_time() <= time_limit and not self.finish_flag:
            finish_count = 0
            for pipeline in self.pipelines:
                pipeline.check_workload_status(time=self.time)
                self.update_constraints_across_dp(time=self.time)
                pipeline.execute_workload(time=self.time)
                pipeline.check_device_status(time=self.time)
                finish_count += pipeline.num_finished_microbatch
            self.finish_flag = True if finish_count == self.get_total_workload_count() else False
            self.update_time()

        for pipeline in self.pipelines:
            pipeline.print_device_utilization(self.get_time())

        for pipeline in self.pipelines:
            pipeline.save_partition()

    def iterative_tuning(self, iter_limit=100, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=True, show_success=True, show_partition=True, show_placement=True, tune_strategy=0):
        it = 0
        if self.chunk_num > 1:
            placement = [[lid for lid in range(did, self.layer_num, self.device_num)] for did in range(self.device_num)]
            placement = [[lid for lid in range(did * self.layer_num//self.device_num, (did + 1) * self.layer_num//self.device_num)] for did in range(self.device_num)]
            partition = [len(p) for p in placement]
        else:
            placement = [[lid for lid in range(did * self.layer_num//self.device_num, (did + 1) * self.layer_num//self.device_num)] for did in range(self.device_num)]
            partition = [len(p) for p in placement]

            placement = [[sid] for sid in range(self.device_num)]
            partition = [self.layer_num//self.device_num for p in placement]

        history = []
        partition_history = []
        placement_history = []
        best_time = 999999999999
        best_partition = None
        layer_comp_time = self.get_layer_comp_times(vocab_parallel=self.tc.vocab_parallel)
        window_size = 10
        partition_tuning_stage = 0
        partition_check_window = window_size
        worst_time = -1
        while it < iter_limit:
            self.one_step_tuning(time_limit=time_limit, placement=placement, partition=partition)
            if self.finish_flag:
                for pipeline in self.pipelines:
                    device_execution_times = pipeline.get_device_execution_time()
                    device_bubble_times = pipeline.get_device_bubble_time()
                    model_partition = pipeline.get_model_partition()
                    model_placement = self.partition_to_placement(model_partition)
                    pipeline_finish_time = max(device_execution_times)
                    worst_time = max(worst_time, pipeline_finish_time)
                    history.append((it, pipeline_finish_time, model_partition, model_placement))
                    partition_history.append(tuple(model_partition))
                    if best_time > pipeline_finish_time:
                        best_time = pipeline_finish_time
                        best_partition = model_partition
                    print(f"Iteration:{it}, Time:{pipeline_finish_time}, Bubbles:{round(sum(device_bubble_times)/pipeline_finish_time/len(device_bubble_times), 2):.2f}, Partition:{best_partition}, Tried:{model_partition}")
                    if tune_strategy == 0:
                        model_partition = tune_partition(best_partition, model_placement, layer_comp_time, device_execution_times, device_bubble_times, history=partition_history, stage=partition_tuning_stage)
                    elif tune_strategy == 1:
                        model_partition = tune_partition_random(best_partition, model_placement, layer_comp_time, device_execution_times, device_bubble_times, history=partition_history, stage=partition_tuning_stage)
                    elif tune_strategy == 2:
                        model_partition = tune_partition_dfs(best_partition, model_placement, layer_comp_time, device_execution_times, device_bubble_times, history=partition_history, stage=partition_tuning_stage)
            else:
                print("Fail")

            it += 1
            converged = False
            if model_partition is None:
                converged = True

            converged, _, _ = check_partition_convergence(history, window=partition_check_window)

            if converged:
                if partition_tuning_stage == 1:
                    break
                elif partition_tuning_stage == 0:
                    partition_tuning_stage += 1
                    partition_check_window += window_size

        best_placement = self.partition_to_placement(partition=best_partition)
        placement = balanced_transpose(placement=placement, layer_comp_time=layer_comp_time)
        partition = [1 for _ in range(self.layer_num)]
        while it < iter_limit:

            self.one_step_tuning(time_limit=time_limit, placement=placement, partition=partition)

            if self.finish_flag:
                for pipeline in self.pipelines:
                    device_execution_times = pipeline.get_device_execution_time()
                    device_bubble_times = pipeline.get_device_bubble_time()
                    model_partition = pipeline.get_model_partition()
                    model_placement = pipeline.get_model_placement()
                    if self.tc.vocab_parallel:
                        for row in model_placement:
                            row.pop()
                    pipeline_finish_time = max(device_execution_times)
                    worst_time = max(worst_time, pipeline_finish_time)
                    history.append((it, pipeline_finish_time, model_partition, model_placement))
                    placement_history.append(serialize(model_placement))
                    if best_time > pipeline_finish_time:
                        best_time = pipeline_finish_time
                        best_placement = model_placement
                    # print(f"Iteration:{it}, Time:{pipeline_finish_time}, Bubbles:{round(sum(device_bubble_times)/pipeline_finish_time/len(device_bubble_times), 2):.2f}, Placement:{best_placement}, Tried:{model_placement}")
                    print(f"Iteration:{it}, Time:{pipeline_finish_time}, Bubbles:{round(sum(device_bubble_times)/pipeline_finish_time/len(device_bubble_times), 2):.2f}")
                    if tune_strategy == 0:
                        placement = tune_placement(best_placement, partition, layer_comp_time, device_execution_times, device_bubble_times, history=placement_history)
                    elif tune_strategy == 1:
                        placement = tune_placement_random(best_placement, partition, layer_comp_time, device_execution_times, device_bubble_times, history=placement_history)
                    elif tune_strategy == 2:
                        placement = tune_placement_dfs(best_placement, partition, layer_comp_time, device_execution_times, device_bubble_times, history=placement_history)
            else:
                print("Fail")

            converged = check_placement_convergence(placement_history=placement_history)
            if converged:
                break
            it += 1

        best_placement = sorted(best_placement, key=lambda x : x[0])
        best_placement = [[lid for lid in range(did, self.layer_num, self.device_num)] for did in range(self.device_num)]
        print(best_placement)
        self.one_step_tuning(time_limit=time_limit, placement=best_placement, partition=partition)
        for pipeline in self.pipelines:
            pipeline.print_partition_placement()
        print(f"Time {worst_time} -> {best_time}")

    def run_all_dp(self, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=False, show_success=True, show_partition=False, show_placement=False):
        self.reset_time()
        partition = [7, 8, 8, 8, 8, 8, 8, 1]
        placement = [[0], [1], [2], [3], [4], [5], [6], [7]]
        self._init_pipelines(partition=partition,placement=placement)
        workloads = {}
        while self.get_time() <= time_limit and not self.finish_flag:
            finish_count = 0
            for pipeline in self.pipelines:
                pipeline.check_workload_status(time=self.time)
                self.update_constraints_across_dp(time=self.time)
                pipeline.execute_workload(time=self.time)
                pipeline.check_device_status(time=self.time)
                finish_count += pipeline.num_finished_microbatch
                if self.dp_size > 1 and self.schedule_method in (Schedule.OctoPipe, Schedule.ReCycle):
                    if self.get_time() == 0:
                        if pipeline.pid == 0:
                            # Pop all
                            # workloads = pipeline.pop_workload(mid_group=list(range(pipeline.mid_offset, pipeline.mid_offset + pipeline.nmb)), did_group=[2])
                            # Pop partial
                            workloads = pipeline.pop_workload(mid_group=list(range(pipeline.mid_offset, pipeline.mid_offset + pipeline.nmb - 3)), did_group=[2])
                    if self.get_time() == 0:
                        if pipeline.pid == 1:
                            pipeline.insert_workload(workloads=workloads,did_group=[2])
            self.finish_flag = True if finish_count == self.get_total_workload_count() else False
            self.update_time()
        if show_success:
            if self.finish_flag:
                if show_partition and show_placement:
                    for pipeline in self.pipelines:
                        pipeline.print_partition_placement()
                print("Success")
            else:
                print("Fail")
        if show_utilization:
            for pipeline in self.pipelines:
                pipeline.print_device_utilization(self.get_time())

    def draw(self) -> None:
        res_all_dp = {}
        res_all_dp["res"]={}
        res_all_dp["painter_conf"]={}
        all_dp_f_times = {}
        all_dp_b_times = {}
        all_dp_w_times = {}
        for dp_idx in range(self.dp_size):
            pipeline = self.pipelines[dp_idx]
            f_times, b_times, w_times = pipeline.get_workloadload_duration()
            for sid in f_times:
                if sid not in all_dp_f_times:
                    all_dp_f_times[sid] = {}
                    all_dp_b_times[sid] = {}
                    all_dp_w_times[sid] = {}

                for mid in f_times[sid]:
                    all_dp_f_times[sid][mid] = f_times[sid][mid]
                    all_dp_b_times[sid][mid] = b_times[sid][mid]
                    all_dp_w_times[sid][mid] = w_times[sid][mid]

            res = {}
            for key in pipeline.results:
                if key.startswith(("f_","b_","w_","r_")):
                    res[key] = pipeline.results[key]
            painter_conf = {
                "device_num": pipeline.device_num,
                "devices": pipeline.placement,
                "stage_num": pipeline.stage_num,
                "pp_height": gpc["PP_HEIGHT"],
                "pp_align": gpc["PP_ALIGN"],
                "pixel_base": gpc["PIXEL_BASE"],
                "nmb": pipeline.nmb,
                "mid_offset": pipeline.mid_offset,
                "comm_length": [gpc["COMM_TIME"] for _ in range(pipeline.stage_num)],
            }
            res_all_dp["res"][dp_idx]=res
            res_all_dp["painter_conf"][dp_idx]=painter_conf
        for dp_idx in range(self.dp_size):
            res_all_dp["painter_conf"][dp_idx]["f_times"] = dict_to_2d_list(all_dp_f_times)
            res_all_dp["painter_conf"][dp_idx]["b_times"] = dict_to_2d_list(all_dp_b_times)
            res_all_dp["painter_conf"][dp_idx]["w_times"] = dict_to_2d_list(all_dp_w_times)
        MPP(res_all_dp["painter_conf"]).draw(res_all_dp["res"])
