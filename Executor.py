from simulator.abstract.Device import *
from simulator.abstract.mutils import *
from simulator.painter import MultiPipelinePainter as MPP
from simulator.abstract.Pipeline import PipelineScheduler
import time
import cProfile
import pstats
from functools import wraps
from collections import defaultdict
import heapq

time_line = []

def time_recorder_decorator(time_line_list):
    """
    创建一个装饰器，用于记录被装饰函数的执行时间，并将其追加到指定的列表中。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 记录函数执行前的时间
            start_time = time.time()
            
            # 执行原始函数
            result = func(*args, **kwargs)
            
            # 记录函数执行后的时间
            end_time = time.time()
            
            # 计算花费的时间
            elapsed_time = end_time - start_time
            
            # 将花费的时间追加到列表中
            if time_line_list:
                time_line_list.append(elapsed_time + time_line_list[-1])
            else:
                time_line_list.append(elapsed_time)
            
            # 返回原始函数的执行结果
            return result
        return wrapper
    return decorator

def check_placement_convergence(placement_history, repeat_count=2):
    """
    placement_history: 列表，每项为序列化后的 placement
    repeat_count: 连续重复次数判定收敛
    返回 (converged: bool)
    """
    n = len(placement_history)
    if n < repeat_count:
        return False

    # 最近 repeat_count 次是否完全相同
    last_places = placement_history[-repeat_count:]
    if all(p == last_places[0] for p in last_places):
        return True

    return False

def balanced_transpose(placement, layer_comp_time):
    # 原始行的总开销
    original_cost = [sum(layer_comp_time[l] for l in row) for row in placement]
    num_rows = len(placement)

    # 分组 comp time -> list of layers
    comp_groups = defaultdict(list)
    for row in placement:
        for l in row:
            comp_groups[layer_comp_time[l]].append(l)

    # 将 comp time 从大到小排序
    comp_times = sorted(comp_groups.keys(), reverse=True)

    # 新的空 placement
    new_place = [[] for _ in range(num_rows)]
    new_cost = [0] * num_rows

    # 一个小根堆按 (当前开销, 行号) 排序
    # 这样每次将更重的 layer 放到当前最轻的行
    heap = [(0, i) for i in range(num_rows)]
    heapq.heapify(heap)

    # 按 comp time 从大到小填充
    for comp in comp_times:
        layers = sorted(comp_groups[comp])  # 保证稳定顺序
        for l in layers:
            cost, idx = heapq.heappop(heap)
            new_place[idx].append(l)
            new_cost[idx] += layer_comp_time[l]
            heapq.heappush(heap, (new_cost[idx], idx))

    # 最终按每行第一个 layer 升序排序保证干净布局
    new_place.sort(key=lambda row: row[0] if row else 1e9)

    return new_place

def serialize(p):
    return tuple(tuple(row) for row in p)

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

    def _init_pipelines(self, placement = None, partition=None, tune_partition=False):
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
    @staticmethod
    def check_partition_convergence(history, window=10):
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
        for it, t, part, place in recent:
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

    def partition_to_placement(self, partition):
        placement = []
        start = 0
        for size in partition:
            end = start + size
            placement.append(list(range(start, end)))
            start = end
        return placement

    def tune_placement(self, model_placement, model_partition, layer_comp_time, device_execution_times, device_bubble_times, history):
        num_devices = len(model_placement)
        bubble_ratio = []
        for exec_t, bubble_t in zip(device_execution_times, device_bubble_times):
            if exec_t == 0:
                ratio = 0.0
            else:
                ratio = bubble_t / exec_t
            bubble_ratio.append(ratio)

        num_devices = len(model_placement)

        # source device 從 bubble 小到大
        source_list = sorted(range(num_devices),
                                key=lambda i: (bubble_ratio[i], device_bubble_times[i]))

        # target device 從 bubble 大到小
        target_list = sorted(range(num_devices),
                                key=lambda i: (bubble_ratio[i], device_bubble_times[i]),
                                reverse=True)

        # 序列化用於檢查重複
        

        old_state = serialize(model_placement)

        # 對所有 source device 嘗試
        for src in source_list:

            if not model_placement[src]:
                continue

            # 找 source device 中最小開銷 layer
            small_layer = min(model_placement[src], key=lambda l: layer_comp_time[l])
            small_cost = layer_comp_time[small_layer]

            # 對所有 target device 嘗試
            for tgt in target_list:

                if tgt == src:
                    continue

                # 在 target device 中找 comp time 更大的 layer
                candidates = [l for l in model_placement[tgt]
                                if layer_comp_time[l] > small_cost]

                if not candidates:
                    continue

                # 選擇 comp time 最小的 candidate
                large_layer = min(candidates, key=lambda l: layer_comp_time[l])

                # 嘗試交換
                new_place = [row[:] for row in model_placement]

                new_place[src].remove(small_layer)
                new_place[tgt].remove(large_layer)
                new_place[src].append(large_layer)
                new_place[tgt].append(small_layer)

                # 排序
                for row in new_place:
                    row.sort()

                new_state = serialize(new_place)

                # 不可和歷史重複
                if new_state not in history and new_state != old_state:
                    return new_state

        # 所有交換方案都無法使用
        return model_placement

    def tune_partition_random(self, partition, placement, layer_comp_time, device_execution_times, device_bubble_times, history, stage=0):
        num_devices = len(partition)
        src = random.randrange(num_devices)
        if partition[src] == 0:
            return partition

        dst = random.randrange(num_devices)
        while dst == src:
            dst = random.randrange(num_devices)

        new_partition = list(partition)
        new_partition[src] -= 1
        new_partition[dst] += 1

        return new_partition
    
    def tune_placement_random(self, model_placement, model_partition, layer_comp_time, device_execution_times, device_bubble_times, history):
        new_partition = model_partition
        new_placement = model_placement

        num_devices = len(new_partition)

        src = random.randrange(num_devices)
        if len(new_placement[src]) == 0:
            return new_placement
        src_layer = random.choice(new_placement[src])

        dst = random.randrange(num_devices)
        while dst == src:
            dst = random.randrange(num_devices)
        dst_layer = random.choice(new_placement[dst])

        new_placement[src].remove(src_layer)
        new_placement[dst].append(src_layer)

        new_placement[src].append(dst_layer)
        new_placement[dst].remove(dst_layer)
        
        return new_placement
    
    def tune_partition_dfs(self, partition, placement, layer_comp_time, device_execution_times, device_bubble_times, history, stage=0):
        num_devices = len(partition)
        src_list = range(num_devices)
        dst_list = range(num_devices)
        for src in src_list:
            for dst in dst_list:
                if dst == src:
                    continue
                new_partition = list(partition)
                new_partition[src] -= 1
                new_partition[dst] += 1
                if tuple(new_partition) in history:
                    continue
                else:
                    return new_partition
        return partition
    
    def tune_placement_dfs(self, model_placement, model_partition, layer_comp_time, device_execution_times, device_bubble_times, history):
        num_devices = len(model_placement)
        for src in range(num_devices):
            src_layers = model_placement[src]
            for dst in range(num_devices):
                if dst == src:
                    continue
                dst_layers = model_placement[dst]
                for src_layer in src_layers:
                    for dst_layer in dst_layers:
                        new_placement = list(model_placement)
                        new_placement[src].remove(src_layer)
                        new_placement[dst].append(src_layer)
                        new_placement[src].append(dst_layer)
                        new_placement[dst].remove(dst_layer)
                        if serialize(new_placement) in history:
                            continue
                        else:
                            return new_placement
        
        return new_placement

        

    def tune_partition(self, partition, placement, layer_comp_time, device_execution_times, device_bubble_times, history, stage=0):
        n = len(partition)
        indexes = list(range(n))

        if stage == 0:
            workloads = []
            for lids in placement:
                w = 0
                for lid in lids:
                    w += layer_comp_time[lid]
                workloads.append(w)

            sorted_idx = sorted(indexes, key=lambda i: workloads[i])
            max_list = sorted_idx[:]
            min_list = sorted_idx[::-1]

        elif stage == 1:
            bubble_ratio = []
            for exec_t, bubble_t in zip(device_execution_times, device_bubble_times):
                if exec_t == 0:
                    ratio = 0.0
                else:
                    ratio = bubble_t / exec_t
                bubble_ratio.append(ratio)

            sorted_idx = sorted(indexes,
                                key=lambda i: (bubble_ratio[i], device_bubble_times[i]),
                                reverse=True)
            max_list = sorted_idx[:]
            min_list = sorted_idx[::-1]

        # Skip the same cases in history
        for max_idx in max_list:
            for min_idx in min_list:

                if max_idx == min_idx:
                    continue

                if partition[min_idx] <= 1:
                    continue

                # new partition
                new_partition = list(partition)
                new_partition[min_idx] -= 1
                new_partition[max_idx] += 1

                # check history
                if tuple(new_partition) not in history:
                    return new_partition

        return partition

    def get_layer_comp_times(self):
        layer_comp_time = [f + b + w if not gpc["RECOMP"] else f + f + b + w for f, b, w in zip(gpc["F_TIMES"], gpc["B_TIMES"], gpc["W_TIMES"])]
        layer_comp_time[-1] += gpc["HEAD_F_TIME"] + gpc["HEAD_B_TIME"] + gpc["HEAD_W_TIME"]
        if gpc["RECOMP"]:
            layer_comp_time[-1] += gpc["HEAD_F_TIME"]
        return  layer_comp_time
    
    @time_recorder_decorator(time_line)
    def one_step_tuning(self, time_limit, placement=None, partition=None, tune_partition=False):
        self._init_pipelines(placement=placement, partition=partition, tune_partition=tune_partition)
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

    def iterative_tuning(self, iter_limit=100, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=True, show_success=True, show_partition=True, show_placement=True, tune_strategy=0):
        it = 0
        if self.chunk_num > 1:
            placement = [[lid for lid in range(did, LAYER_NUM, DEVICE_NUM)] for did in range(DEVICE_NUM)]
            placement = [[lid for lid in range(did * LAYER_NUM//DEVICE_NUM, (did + 1) * LAYER_NUM//DEVICE_NUM)] for did in range(DEVICE_NUM)]
            partition = [len(p) for p in placement]
        else:
            placement = [[lid for lid in range(did * LAYER_NUM//DEVICE_NUM, (did + 1) * LAYER_NUM//DEVICE_NUM)] for did in range(DEVICE_NUM)]
            partition = [len(p) for p in placement]

            placement = [[sid] for sid in range(DEVICE_NUM)]
            partition = [LAYER_NUM//DEVICE_NUM for p in placement]

        # worst
        # placement = [
        #     [0], 
        #     [1], 
        #     [2], 
        #     [3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        # ]
        history = []
        partition_history = []
        placement_history = []
        best_time = 999999999999
        best_partition = None
        layer_comp_time = self.get_layer_comp_times()
        window_size = 10
        partition_tuning_stage = 0
        partition_check_window = window_size
        worst_time = -1
        while it < iter_limit:
            self.one_step_tuning(time_limit=time_limit, placement=placement, partition=partition, tune_partition=True)
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
                        model_partition = self.tune_partition(best_partition, model_placement, layer_comp_time, device_execution_times, device_bubble_times, history=partition_history, stage=partition_tuning_stage)
                    elif tune_strategy == 1:
                        model_partition = self.tune_partition_random(best_partition, model_placement, layer_comp_time, device_execution_times, device_bubble_times, history=partition_history, stage=partition_tuning_stage)
                    elif tune_strategy == 2:
                        model_partition = self.tune_partition_dfs(best_partition, model_placement, layer_comp_time, device_execution_times, device_bubble_times, history=partition_history, stage=partition_tuning_stage)
            else:
                print("Fail")
            
            it += 1
            converged = False
            if model_partition is None:
                converged = True
            
            converged, _, _ = self.check_partition_convergence(history,window=partition_check_window)
            
            if converged:
                if partition_tuning_stage == 1:
                    break
                elif partition_tuning_stage == 0:
                    partition_tuning_stage += 1
                    partition_check_window += window_size
        
        placement_check_window = partition_check_window + window_size * 10
        best_placement = self.partition_to_placement(partition=best_partition)
        placement = balanced_transpose(placement=placement, layer_comp_time=layer_comp_time)
        while it < iter_limit:

            self.one_step_tuning(time_limit=time_limit, placement=placement)

            if self.finish_flag:
                for pipeline in self.pipelines:
                    device_execution_times = pipeline.get_device_execution_time()
                    device_bubble_times = pipeline.get_device_bubble_time()
                    model_partition = pipeline.get_model_partition()
                    model_placement = pipeline.get_model_placement()
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
                        placement = self.tune_placement(best_placement, best_partition, layer_comp_time, device_execution_times, device_bubble_times, history=placement_history)
                    elif tune_strategy == 1:
                        placement = self.tune_placement_random(best_placement, best_partition, layer_comp_time, device_execution_times, device_bubble_times, history=placement_history)
                    elif tune_strategy == 2:
                        placement = self.tune_placement_dfs(best_placement, best_partition, layer_comp_time, device_execution_times, device_bubble_times, history=placement_history)
            else:
                print("Fail")

            converged = check_placement_convergence(placement_history=placement_history)
            if converged:
                break
            it += 1
        
        best_placement = sorted(best_placement, key=lambda x : x[0])
        best_placement = [[lid for lid in range(did, LAYER_NUM, self.device_num)] for did in range(self.device_num)]
        print(best_placement)
        self.one_step_tuning(time_limit=time_limit, placement=best_placement)
        for pipeline in self.pipelines:
            pipeline.save_partition()
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
    random.seed(1024)
    # Example
    chunk_num = gpc["LAYER_NUM"] // gpc["DEVICE_NUM"]
    chunk_num = 1
    schedule_method = Schedule.OctoPipe
    # schedule_method = Schedule.STANDARD_1F1B
    # schedule_method = Schedule.STANDARD_ZBH
    # schedule_method = Schedule.STANDARD_INTERLEAVED
    bwd_split = False
    if schedule_method == Schedule.STANDARD_ZBH:
        bwd_split = True
    if schedule_method == Schedule.STANDARD_INTERLEAVED:
        chunk_num = gpc["LAYER_NUM"] // gpc["DEVICE_NUM"]
        # chunk_num = 2
        # bwd_split = False
    
    tc = TrainingConfig(
        pp_size=gpc["DEVICE_NUM"],
        tp_size=1,
        dp_size=1,
        bwd_split=bwd_split,
        vocab_parallel  =False,
        overlap_aware   =   gpc["OVERLAP_AWARE_SCHEDULE"],
        save_memory     =   gpc["SAVE_MEMORY"],
        constrain_warmup    =   gpc["CONSTRAIN_WARMUP"],
        swith_workload_type =   gpc["SWITCH_WORKLOAD_TYPE"],
        layer_num=gpc["LAYER_NUM"],
        chunk_num=chunk_num,
        device_num=gpc["DEVICE_NUM"],
        micro_batch_num=gpc["MICRO_BATCH_NUM"],
        micro_batch_size=gpc["MICRO_BATCH_SIZE"],
        layer_f_times=gpc["F_TIMES"],
        layer_b_times=gpc["B_TIMES"],
        layer_w_times=gpc["W_TIMES"],
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
                executor.iterative_tuning(iter_limit=50,tune_strategy=0)
        else:
            executor.run_all_dp()
    else:
        partition = [7, 8, 8, 8, 8, 8, 8, 1]
        placement = [[0], [1], [2], [3], [4], [5], [6], [7]]
        partition = [16, 18, 17, 5]
        placement = [[0], [1], [2], [3]]
        partition = [8, 8, 8, 7, 8, 8, 8, 1]
        partition = [6, 6, 7, 6, 10, 10, 10, 1]
        placement = [[0,4], [1,5], [2,6], [3,7]]
        partition = [7, 8, 10, 10, 9, 8, 4]
        placement = [[0,3], [1,4], [2,5], [6]]

        partition = [8, 7, 7, 6]
        placement = [[0], [1], [2], [3]]
        # partition = None
        # placement = None
        executor.one_step_tuning(time_limit=100000, placement=placement, partition=partition)
        # executor.one_step_tuning(time_limit=10000, placement=[[lid for lid in range(did, LAYER_NUM, DEVICE_NUM)] for did in range(DEVICE_NUM)])
        # executor.one_step_tuning(time_limit=10000)


    executor.draw()

    