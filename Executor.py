from simulator.abstract.Device import *
from simulator.abstract.mutils import *
from simulator.painter import MultiPipelinePainter as MPP
from simulator.abstract.Pipeline import PipelineScheduler
import cProfile
import pstats

class Executor:

    def __init__(self, dp_size, nmb_per_dp: list = None, device_comp_power: list[list] = None, device_mem: list[list] = None) -> None:
        self.time           = 0
        self.finish_flag    = False
        self.dp_size        = dp_size
        self.device_num_per_dp  = gpc["DEVICE_NUM"]
        self.device_comp_power  = device_comp_power if device_comp_power else [[gpc["COMP_POWER"] for _ in range(self.device_num_per_dp)] for _ in range(dp_size)]
        self.device_mem         = device_mem if device_mem else [[gpc["GPU_MAX_MEM"] for _ in range(self.device_num_per_dp)] for _ in range(dp_size)]
        self.nmb_per_dp     = [gpc["MICRO_BATCH_NUM"] for _ in range(dp_size)] if not nmb_per_dp else nmb_per_dp
        self.mid_offsets    = [0] + [sum(self.nmb_per_dp[:i]) for i in range(1, dp_size+1)]
        self.micro_batch_size = gpc["MICRO_BATCH_SIZE"]
        self.batch_size = sum(self.nmb_per_dp) * self.micro_batch_size
        self.pipelines      = [
            PipelineScheduler(
                pipeline_idx=dp_idx, 
                nmb=self.nmb_per_dp[dp_idx], 
                mid_offset=self.mid_offsets[dp_idx],
                comp_power=self.device_comp_power[dp_idx],
                max_mem=self.device_mem[dp_idx],
                mbs=self.micro_batch_size,
                bs=self.batch_size,
                executor=self
            ) for dp_idx in range(dp_size)
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

    def run_all_dp(self, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=True, show_success=True):
        self.reset_time()
        workloads = {}
        while self.get_time() <= time_limit and not self.finish_flag:
            finish_count = 0
            for pipeline in self.pipelines:
                pipeline.check_workload_status(time=self.time)
                self.update_constraints_across_dp(time=self.time)
                pipeline.execute_workload(time=self.time)
                pipeline.check_device_status(time=self.time)
                finish_count += pipeline.num_finished_microbatch
                if self.dp_size > 1 and gpc["SCHEDULE_METHOD"] in (Schedule.OctoPipe, Schedule.ReCycle):
                    if self.get_time() == 0:
                        if pipeline.pipeline_idx == 0:
                            # Pop all
                            # workloads = pipeline.pop_workload(mid_group=list(range(pipeline.mid_offset, pipeline.mid_offset + pipeline.nmb)), did_group=[2])
                            # Pop partial
                            workloads = pipeline.pop_workload(mid_group=list(range(pipeline.mid_offset, pipeline.mid_offset + pipeline.nmb - 3)), did_group=[2])
                    if self.get_time() == 0:
                        if pipeline.pipeline_idx == 1:
                            pipeline.insert_workload(workloads=workloads,did_group=[2])
                    # if self.get_time() == 666:
                    #     if pipeline.pipeline_idx == 1:
                    #         workloads = pipeline.pop_workload(mid_group=[2],did_group=[2])
                    # if self.get_time() == 666:
                    #     if pipeline.pipeline_idx == 0:
                    #         pipeline.insert_workload(workloads=workloads,did_group=[2])
            self.finish_flag = True if finish_count == self.get_total_workload_count() else False
            self.update_time()
        if show_success:
            if self.finish_flag:
                print("Success")
            else:
                print("Fail")

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
    # executor = Executor(dp_size=2, nmb_per_dp=[9, 7], device_comp_power=[[2 for _ in range(gpc["DEVICE_NUM"])] for _ in range(2)])
    dp_size = 1
    executor = Executor(dp_size=dp_size, nmb_per_dp=[128], device_comp_power=[[2 for _ in range(gpc["DEVICE_NUM"])] for _ in range(dp_size)])

    if gpc["PROFILE_GENERATION"]:
        profiler = cProfile.Profile()
        profiler.enable()
        executor.run_all_dp()
        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(20)  # 打印前 10 个耗时函数
    else:
        executor.run_all_dp()

    executor.draw()