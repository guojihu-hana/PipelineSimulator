from simulator.abstract.Device import *
from simulator.abstract.mutils import *
from simulator.painter import MultiPipelinePainter as MPP
from simulator.abstract.Pipeline import PipelineScheduler
import cProfile
import pstats

class Executor:

    def __init__(self, schedule_method, dp_size, bwd_split: bool, chunk_num: int, nmb_per_dp: list = None, device_comp_power: list[list] = None, device_mem: list[list] = None) -> None:
        self.schedule_method = schedule_method
        self.bwd_split      = bwd_split
        self.time           = 0
        self.finish_flag    = False
        self.dp_size        = dp_size
        self.chunk_num      = chunk_num
        self.device_num_per_dp  = gpc["DEVICE_NUM"]
        self.device_comp_power  = device_comp_power if device_comp_power else [[gpc["COMP_POWER"] for _ in range(self.device_num_per_dp)] for _ in range(dp_size)]
        self.device_mem         = device_mem if device_mem else [[gpc["GPU_MAX_MEM"] for _ in range(self.device_num_per_dp)] for _ in range(dp_size)]
        self.nmb_per_dp     = [gpc["MICRO_BATCH_NUM"] for _ in range(dp_size)] if not nmb_per_dp else nmb_per_dp
        self.mid_offsets    = [0] + [sum(self.nmb_per_dp[:i]) for i in range(1, dp_size+1)]
        self.micro_batch_size = gpc["MICRO_BATCH_SIZE"]
        self.batch_size = sum(self.nmb_per_dp) * self.micro_batch_size
        self.best_pipelines = None

    def _init_pipelines(self, placement = None):
        self.pipelines      = [
            PipelineScheduler(
                schedule_method=self.schedule_method,
                bwd_split=self.bwd_split,
                pipeline_idx=dp_idx, 
                chunk_num=self.chunk_num,
                placement=placement,
                nmb=self.nmb_per_dp[dp_idx], 
                mid_offset=self.mid_offsets[dp_idx],
                comp_power=self.device_comp_power[dp_idx],
                max_mem=self.device_mem[dp_idx],
                mbs=self.micro_batch_size,
                bs=self.batch_size,
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
                            if d.did == device.did or p.pipeline_idx == pipeline.pipeline_idx:
                                continue # only update constraints on other devices or pipelines
                            if finished_mid in d.held_mids:
                                d.update_constraints_within_device(time, constraint=device.current_workload)
    @staticmethod
    def check_convergence(history, window=10):
        """
        history: 列表 每项为 (iteration, time_value, partition)
        window: 用最近 window 次迭代判断收敛
        返回 (converged, best_partition, best_time)
        """

        n = len(history)
        if n < window:
            return False, None, None

        recent = history[-window:]

        # 找到最近 window 中的最优
        times = [item[1] for item in recent]
        best_time = min(times)
        best_partition = None
        for it, t, part in recent:
            if t == best_time:
                best_partition = part
                break

        # 情况一 partition 模式来回震荡不再变化
        partitions = [tuple(item[2]) for item in recent]
        unique_parts = set(partitions)
        if len(unique_parts) <= 2:
            return True, best_partition, best_time

        # 情况二 全局最优不再改进
        # 这里确保不会对空列表取 min
        earlier = history[:-window]
        if earlier:
            earlier_best_time = min(item[1] for item in earlier)
            # 若最近 window 次的最优时间不比更早的时间好 说明不再改进
            if best_time >= earlier_best_time:
                return True, best_partition, best_time

        return False, None, None


    def tune_partition(self, partition, device_execution_times, device_bubble_times):
        # 计算 bubble ratio
        bubble_ratio = []
        for exec_t, bubble_t in zip(device_execution_times, device_bubble_times):
            if exec_t == 0:
                ratio = 0.0
            else:
                ratio = bubble_t / exec_t
            bubble_ratio.append(ratio)

        # 找到 bubble 最大和最小的 device
        max_idx = max(range(len(bubble_ratio)), key=lambda i: bubble_ratio[i])
        min_idx = min(range(len(bubble_ratio)), key=lambda i: bubble_ratio[i])

        # 如果最小的 device 没有 layer 可以分就不做调整
        if partition[min_idx] == 0:
            return partition

        # 从 bubble 最小的 device 分一个到 bubble 最大的 device
        new_partition = list(partition)
        new_partition[min_idx] -= 1
        new_partition[max_idx] += 1

        return new_partition

    def iterative_tuning(self, iter_limit=100, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=True, show_success=True, show_partition=True, show_placement=True):
        it = 0
        placement = [[1 for _ in range(LAYER_NUM//DEVICE_NUM)] for _ in range(DEVICE_NUM)]
        partition = [len(p) for p in placement]
        history = []
        while it < iter_limit:
            self._init_pipelines(placement=placement)
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
            it += 1

            if self.finish_flag:
                for pipeline in self.pipelines:
                    device_execution_times = pipeline.get_device_execution_time()
                    device_bubble_times = pipeline.get_device_bubble_time()
                    model_partition = pipeline.get_model_partition()
                    model_placement = pipeline.get_model_placement()
                    pipeline_finish_time = max(device_execution_times)
                    history.append((it, pipeline_finish_time, model_partition))
                    print(f"Iteration:{it}, Time:{pipeline_finish_time}, Partition:{model_partition}")
                    model_partition = self.tune_partition(model_partition, device_execution_times, device_bubble_times)
                    placement = [[1 for _ in range(model_partition[did])] for did in range(DEVICE_NUM)]
            else:
                print("Fail")

            converged, best_part, best_time = self.check_convergence(history)
            if converged:
                print("Converged at iter", it)
                print("Best time", best_time)
                print("Best partition", best_part)
                break

    def run_all_dp(self, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=True, show_success=True, show_partition=True, show_placement=True):
        self.reset_time()
        self._init_pipelines()
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
                        if pipeline.pipeline_idx == 0:
                            # Pop all
                            # workloads = pipeline.pop_workload(mid_group=list(range(pipeline.mid_offset, pipeline.mid_offset + pipeline.nmb)), did_group=[2])
                            # Pop partial
                            workloads = pipeline.pop_workload(mid_group=list(range(pipeline.mid_offset, pipeline.mid_offset + pipeline.nmb - 3)), did_group=[2])
                    if self.get_time() == 0:
                        if pipeline.pipeline_idx == 1:
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
                "stage_num": pipeline.stage_num if not gpc["HEAD_DP"] else pipeline.stage_num + 1,
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

if __name__ == "__main__":
    # Example
    # executor = Executor(dp_size=4, nmb_per_dp=[15, 12, 20, 17])
    # Device fail-slow or fail
    # executor = Executor(dp_size=2, nmb_per_dp=[22, 27], device_comp_power=[[2 for _ in range(gpc["DEVICE_NUM"])] for _ in range(2)])
    # dp_size = 1
    chunk_num = gpc["LAYER_NUM"] // gpc["DEVICE_NUM"]
    chunk_num = 1
    schedule_method = Schedule.OctoPipe
    schedule_method = Schedule.STANDARD_1F1B
    bwd_split = True
    executor = Executor(dp_size=1, nmb_per_dp=[4], chunk_num=chunk_num, schedule_method=schedule_method, bwd_split=bwd_split, device_comp_power=[[1 for _ in range(gpc["DEVICE_NUM"])] for _ in range(1)])
    
    if schedule_method == Schedule.OctoPipe:
        if gpc["PROFILE_GENERATION"]:
            profiler = cProfile.Profile()
            profiler.enable()
            executor.iterative_tuning(iter_limit=100)
            profiler.disable()

            stats = pstats.Stats(profiler).sort_stats("cumtime")
            stats.print_stats(20)  # 打印前 10 个耗时函数
        else:
            executor.iterative_tuning()
    else:
        executor.run_all_dp()
    
    # if gpc["PROFILE_GENERATION"]:
    #     profiler = cProfile.Profile()
    #     profiler.enable()
    #     executor.run_all_dp()
    #     profiler.disable()

    #     stats = pstats.Stats(profiler).sort_stats("cumtime")
    #     stats.print_stats(20)  # 打印前 10 个耗时函数
    # else:
    #     executor.run_all_dp()

    executor.draw()
    
    # test bubble
    # for sm in [Schedule.STANDARD_ZBH]:
    # for sm in [Schedule.STANDARD_1F1B, Schedule.STANDARD_INTERLEAVED, Schedule.STANDARD_ZBH, Schedule.Mist]:
    #     if sm == Schedule.STANDARD_INTERLEAVED:
    #         chunk_num = LAYER_NUM // DEVICE_NUM
    #     else:
    #         chunk_num = 1
    #     bwd_split = False
    #     if sm == Schedule.STANDARD_ZBH:
    #         bwd_split = True
        
    #     ideal_case = True
    #     flush_fbw_time(bwd_split, ideal_case)

    #     executor = Executor(dp_size=1, nmb_per_dp=[16], chunk_num=chunk_num, schedule_method=sm, bwd_split=bwd_split, device_comp_power=[[1 for _ in range(gpc["DEVICE_NUM"])] for _ in range(1)])
    #     executor.run_all_dp()
    #     executor.draw()

    