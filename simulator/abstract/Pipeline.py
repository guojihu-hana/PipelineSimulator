from .Device import *
from .Stage import Stage
from .Workload import Workload
from .mutils import *
from ..painter import SchedulingPainter as SP
from ..LayerwisePainter import LayerwiseSchedulingPainter as LSP
from ..utils import save_to_file
from .Placement import PipelinePlacement
from ..solver.predefined_model_partition_placement import get_octopipe_predefined_partition_placement, get_mist_predefined_partition_placement
from ..solver.ordered_model_partition_placement import solve_ordered
from ..solver.unordered_model_partition_placement import solve_unordered
import itertools
import json
import os
import copy
workload_type_mapping = {
    'f':WorkloadType.F,
    'b':WorkloadType.B,
    'w':WorkloadType.W,
    'r':WorkloadType.R,
}

class PipelineScheduler:

    def __init__(self, pipeline_idx, schedule_method, training_config: TrainingConfig, time=0, mid_offset=None, placement=None, partition=None, run_schedule=False, comp_power:list=None, max_mem:list=None, executor=None) -> None:
        self.pid = pipeline_idx
        self.schedule_method = schedule_method

        self.tc         = training_config
        self.bwd_split  = self.tc.bwd_split
        self.chunk_num  = self.tc.chunk_num
        self.device_num = self.tc.device_num
        self.layer_num  = self.tc.layer_num
        self.bs         = self.tc.batch_size
        self.nmb        = self.tc.micro_batch_num
        self.mbs        = self.tc.micro_batch_size

        self.executor = executor
        self.time = time
        
        self.results = {}
        self.devices: list[Device] = []
        self.mid_offset = pipeline_idx * self.nmb if not mid_offset else mid_offset
        self.solver_results = None
        self.comp_power = comp_power if comp_power else [1 for _ in range(self.device_num)]
        self.max_mem = max_mem if max_mem else [gpc["GPU_MAX_MEM"] for _ in range(self.device_num)]

        self._init_placement_and_partition(schedule_method=schedule_method, placement=placement, partition=partition)
        
        self.total_workload = self.stage_num * self.nmb
        self.layer_wise = gpc["LAYERWISE"]
        self.head_dp = gpc["HEAD_DP"]
        self.finish_flag = False
        self.num_finished_microbatch = 0
        self.run_schedule = run_schedule
        self.manual_recomp_set = []
        self.workload_execute_record: list[list[Workload]] = [[] for _ in range(self.device_num)]
        self._init_device()
        self.schedule = [[] for _ in range(self.device_num)]
        self.generate_workload_schedule()
        self.set_workload_schedule()
        self.temp_results = {}
        self.last_workload: Workload = None
        if run_schedule:
            print("Read schedule generated before...")
            self.file2result()
            self.result2schedule()
            self.set_workload_schedule()

        for device in self.devices:
            workloads = device.get_initial_executable_workload(time=time)
            for workload in workloads:
                device.executable_workloads.push(workload)

    def _init_placement_and_partition(self, schedule_method, placement, partition):
        if schedule_method in (Schedule.STANDARD_ZBH, Schedule.STANDARD_1F1B, Schedule.STANDARD_INTERLEAVED, Schedule.STANDARD_AFAB, Schedule.ReCycle):
            self.stage_num = self.device_num * self.chunk_num
            assert self.layer_num % self.stage_num == 0, f"Layer num {self.layer_num} should be divisible by stage num {self.stage_num} for standard schedules."
            self.partition = [self.layer_num // self.stage_num] * self.stage_num
            self.placement = [[i + self.device_num * j for j in range(self.chunk_num)] for i in range(self.device_num)]
        elif schedule_method == Schedule.Mist:
            assert partition, "Mist schedule requires predefined partition."
            self.partition = partition
            self.placement = [[i] for i in range(self.device_num)]
            if placement != self.placement:
                print("Warning: Mist schedule should use partition-based placement, provided placement will be ignored.")
            self.schedule_method = Schedule.STANDARD_1F1B
        elif placement and partition:
            assert self.layer_num == sum(partition), f"Layer num should be equal to the sum of partition ({self.layer_num} != {sum(partition)})."
            assert len(placement) == self.device_num, f"Placement length should be equal to device num ({len(placement)} != {self.device_num})."
            assert len(partition) == sum([len(p) for p in placement]), f"Partition length should be equal to the total number of stages in placement ({len(partition)} != {sum([len(p) for p in placement])})."
            self.stage_num = len(partition)
            self.partition = partition
            self.placement = placement
        else:
            assert placement is None and partition is None, "Either both placement and partition should be provided, or neither of them should be provided."
        print(f"-- Initialize {self.schedule_method.name} placement: {self.placement} and partition : {self.partition} successfully.")

    def set_model_partition_and_placement(self):
        self.partition = get_octopipe_predefined_partition_placement(seq_len=SEQ_LEN, device_num=self.device_num, layer_num=self.layer_num)
        if self.schedule_method in (Schedule.STANDARD_ZBH, Schedule.STANDARD_1F1B, Schedule.STANDARD_AFAB, Schedule.ReCycle):
            self.partition = [self.layer_num // self.device_num] * self.device_num
        if self.schedule_method == Schedule.Mist:
            self.partition = get_mist_predefined_partition_placement(seq_len=SEQ_LEN, device_num=self.device_num, layer_num=self.layer_num)
            self.schedule_method = Schedule.STANDARD_1F1B

        self.placement = [
            [1 for _ in range(layer_num)] for layer_num in self.partition
        ]

        if self.schedule_method == Schedule.OctoPipe:
            computation_times = [f + b + w for f, b, w in zip(gpc["F_TIMES"], gpc["B_TIMES"], gpc["W_TIMES"])]
            computation_times[-1] += gpc["HEAD_F_TIME"] + gpc["HEAD_B_TIME"] + gpc["HEAD_W_TIME"]
            if self.stage_num == self.layer_num:
                solver_results = solve_unordered(
                    times=computation_times, 
                    D=self.device_num,
                    mems=[1 for _ in range(self.layer_num)], 
                    mem_limits=[144 for _ in range(self.device_num)]
                )
            else:
                solver_results = solve_ordered(
                    times=computation_times, 
                    D=self.device_num,
                    mems=[1 for _ in range(self.layer_num)], 
                    mem_limits=[144 for _ in range(self.device_num)]
                )
            self.solver_results = solver_results
            self.placement = solver_results["assignments"]
            self.partition = [len(p) for p in self.placement]

        self.save_partition()

    def save_partition(self):
        with open("schedule_results/partition.txt", 'w') as f:
            f.write(str(self.partition))
            f.flush()

    def sid2did(self, sid):
        for did, sids in enumerate(self.placement):
            if sid in sids:
                return did
            
    def result2schedule(self):
        for key in self.results.keys():
            if not key.startswith(("f_","b_","w_",)):
                continue
            k, mid, sid = key.split("_")[:3]
            sid = int(sid)
            mid = int(mid)
            did = self.sid2did(sid=sid)
            t = self.results[key]
            self.schedule[did].append((workload_type_mapping[k], mid, sid, t))
        print("Result to schedule successfully.")

    def result2file(self, filepath=None):
        if filepath is None:
            filepath = 'data.txt'
        with open(filepath, 'w') as file:
            json.dump(self.results, file)
        print("Result to file successfully.")

    def file2result(self, filepath=None):
        if filepath is None:
            filepath = 'data.txt'
        with open(filepath, 'r') as file:
            results = json.load(file)
            self.results = {}
            for k in results.keys():
                if str(k).startswith(("f","b","w")):
                    self.results[k] = results[k]

    # NOTE _reset_workload_type is efficient but 
    # lead to random order of W in some cases
    def resort_w(self):
        w_times = [[] for _ in range(self.layer_num + 3)]
        for res in self.results:
            if res.startswith("w"):
                w,mid,sid = res.split("_")
                sid = int(sid)
                w_times[sid].append(self.results[res])

        for sid in range(self.layer_num + 3):
            w_times_in_sid = sorted(w_times[sid])
            for mid in range(len(w_times_in_sid)):
                w_key = f"w_{mid}_{sid}"
                self.results[w_key] = w_times_in_sid[mid]

    def record_recomputation_config(self):
        for idx, r in enumerate(self.recomp_set):
            self.recomp_set[idx] = 1 if r else 0
            self.results[f"theta_{idx}"] = r

    def _init_device(self):
        for did in range(self.device_num):
            device = Device(
                device_idx = did,
                schedule_method = self.schedule_method,
                training_config = self.tc,
                stage_num   = self.stage_num,
                mid_offset  = self.mid_offset,
                max_mem     = self.max_mem[did],
                comp_power  = self.comp_power[did],
                pipeline    = self,
            )
            self.devices.append(device)
        self.set_recomputation_config()
        assert self.placement and self.partition, "Placement and partition should be provided for initializing devices."
        layer_idx_start = 0
        for did in range(self.device_num):
            for sid in self.placement[did]:
                layer_num = self.partition[sid]
                self.devices[did].add_stage(stage_id=sid, layer_num=layer_num, layer_idx_start=layer_idx_start, recomp=self.recomp_set[did])
                layer_idx_start += layer_num
        
        if self.head_dp:
            for device in self.devices:
                device.add_stage(self.stage_num, False, False, 0)

        # Launch MemoryMonitors
        for device in self.devices:
            device.init_memory_monitor()

        self.placement = []
        for did in range(self.device_num):
            self.placement.append(list(self.devices[did].stages.keys()))

        # print(self.placement)

        save_to_file(gpc["TEMP_PLA_PATH"], str(self.placement), 'w')
        save_to_file(gpc["PLA_FILE_PATH"], str(self.placement), 'w')

    def set_recomputation_config(self):
        self.recomp_set = [1 if gpc["RECOMP"] else 0 for _ in range(self.stage_num)]
        if self.manual_recomp_set:
            print("Use provided recomputation config.")
            self.recomp_set = self.manual_recomp_set
        
    def set_workload_schedule(self):
        for did in range(self.device_num):
            self.devices[did].static_schedule = self.schedule[did]

    def generate_workload_schedule(self):
        if self.schedule_method == Schedule.STANDARD_1F1B:
            self.generate_1f1b_schedule()
            # print("Generate 1F1B Schedule.")
        elif self.schedule_method == Schedule.STANDARD_AFAB:
            self.generate_afab_schedule()
            # print("Generate AFAB Schedule.")
        elif self.schedule_method == Schedule.STANDARD_ZBH:
            self.generate_zbh_schedule()
            # print("Generate ZBH Schedule.")
        elif self.schedule_method == Schedule.STANDARD_INTERLEAVED:
            self.generate_interleaved_1f1b_schedule()
            # print("Generate I-1F1B Schedule.")
        else:
            # print(f"Generate {self.schedule_method.name} Schedule.")
            pass

    def generate_afab_schedule(self):
        assert self.chunk_num == 1
        workload_type_order = [WorkloadType.F, WorkloadType.B]
        if self.bwd_split:
            workload_type_order.append(WorkloadType.W)
        workload_type_num = len(workload_type_order)
        mid_offset = self.mid_offset
        for did in range(self.device_num):
            mids = [0 for _ in range(workload_type_num)]
            for i in range(workload_type_num):
                while mids[i] < self.nmb:
                    self.schedule[did].append((workload_type_order[i], mids[i] + mid_offset, did))
                    mids[i]+=1

    def generate_1f1b_schedule(self):
        assert self.chunk_num == 1
        workload_type_order = [WorkloadType.B, WorkloadType.W, WorkloadType.F] if self.bwd_split else [WorkloadType.B, WorkloadType.F]
        workload_type_num = len(workload_type_order)
        workload_idx_in_mids = {WorkloadType.F: 0, WorkloadType.B : 1, WorkloadType.W : 2}
        mid_offset = self.mid_offset
        for did in range(self.device_num):
            mids = [0 for _ in range(workload_type_num)]
            # warmup
            while mids[0] < min(self.device_num - did, self.nmb):
                self.schedule[did].append((WorkloadType.F, mids[0] + mid_offset, did))
                mids[0] += 1
            
            iter = 0
            finish_flag = [0 for _ in range(workload_type_num)]
            while sum(finish_flag) < workload_type_num:
                next_workload_type = workload_type_order[iter % workload_type_num]
                next_mid = mids[workload_idx_in_mids[next_workload_type]]
                if next_mid < self.nmb:
                    self.schedule[did].append((next_workload_type, next_mid + mid_offset, did))
                    mids[workload_idx_in_mids[next_workload_type]] += 1
                else:
                    finish_flag[workload_idx_in_mids[next_workload_type]] = 1
                iter+=1

    def generate_zbh_schedule(self):
        assert self.bwd_split

        workload_type_order = [WorkloadType.B, WorkloadType.W, WorkloadType.F]
        workload_type_num = len(workload_type_order)
        workload_idx_in_mids = {WorkloadType.F: 0, WorkloadType.B : 1, WorkloadType.W : 2}
        mid_offset = self.mid_offset
        for did in range(self.device_num):
            accumulated_act_num = min(self.nmb, (self.device_num - did - 1) * gpc["MAX_ACT"] + 1)
            mids = [0 for _ in range(workload_type_num)]
            # warmup, should not be simplified
            while mids[0] < min(accumulated_act_num, self.nmb):
                self.schedule[did].append((WorkloadType.F, mids[0] + mid_offset, did))
                mids[0] += 1
            
            # steady + cooldown
            iter = 0
            finish_flag = [0 for _ in range(workload_type_num)]
            while sum(finish_flag) < workload_type_num:
                next_workload_type = workload_type_order[iter % workload_type_num]
                next_mid = mids[workload_idx_in_mids[next_workload_type]]
                if mids[0] < min(self.nmb, self.stage_num * gpc["MAX_ACT"]):
                    if next_workload_type == WorkloadType.W:
                        iter += 1
                        continue 
                if next_mid < self.nmb:
                    self.schedule[did].append((next_workload_type, next_mid+mid_offset, did))
                    mids[workload_idx_in_mids[next_workload_type]] += 1
                else:
                    finish_flag[workload_idx_in_mids[next_workload_type]] = 1
                iter+=1

    def generate_interleaved_1f1b_schedule(self):
        workload_type_num = 2
        mid_offset = self.mid_offset
        for did in range(self.device_num):
            sids = list(self.placement[did])
            
            mids = [0 for _ in range(workload_type_num * self.chunk_num)]
            f_mid_count = 0
            
            f_next_sid_idx = 0
            f_next_sid = sids[f_next_sid_idx]
            idx_in_f_mids = f_next_sid_idx * workload_type_num
        
            warmup_f_num = (self.chunk_num - 1) * self.device_num + (self.device_num - did - 1) * 2
            while mids[idx_in_f_mids] < self.nmb and f_mid_count < warmup_f_num:
                self.schedule[did].append((WorkloadType.F ,mids[idx_in_f_mids] + mid_offset, f_next_sid))
                mids[idx_in_f_mids] += 1
                f_mid_count += 1
                if f_mid_count % self.device_num == 0:
                    f_next_sid_idx = (f_next_sid_idx + 1) % len(sids)
                    f_next_sid = sids[f_next_sid_idx]
                    idx_in_f_mids = f_next_sid_idx * workload_type_num

            b_mid_count = 0
            bsids = list(reversed(sids))
            b_next_sid_idx = 0
            b_next_sid = bsids[b_next_sid_idx]
            idx_in_b_mids = 1 + b_next_sid_idx * workload_type_num

            # Start 1f1b with F operation
            operation_flag = 'f'
            while b_mid_count + f_mid_count < self.nmb * self.chunk_num * workload_type_num:
                if operation_flag == 'f':
                    if mids[idx_in_f_mids] < self.nmb:
                        self.schedule[did].append((WorkloadType.F ,mids[idx_in_f_mids] + mid_offset, f_next_sid))
                        mids[idx_in_f_mids] += 1
                        f_mid_count += 1
                        if f_mid_count % self.device_num == 0:
                            f_next_sid_idx = (f_next_sid_idx + 1) % len(sids)
                            f_next_sid = sids[f_next_sid_idx]
                            idx_in_f_mids = f_next_sid_idx * workload_type_num
                    operation_flag = 'b'
                elif operation_flag == 'b':
                    if mids[idx_in_b_mids] < self.nmb:
                        if gpc["RECOMP"] and gpc["SPLIT_RECOMP"]:
                            self.schedule[did].append((WorkloadType.R ,mids[idx_in_b_mids] + mid_offset, b_next_sid))
                        self.schedule[did].append((WorkloadType.B ,mids[idx_in_b_mids] + mid_offset, b_next_sid))
                        if self.bwd_split:
                            self.schedule[did].append((WorkloadType.W ,mids[idx_in_b_mids] + mid_offset, b_next_sid))
                        mids[idx_in_b_mids] += 1
                        b_mid_count += 1
                        if b_mid_count % self.device_num == 0:
                            b_next_sid_idx = (b_next_sid_idx + 1) % len(bsids)
                            b_next_sid = bsids[b_next_sid_idx]
                            idx_in_b_mids = 1 + b_next_sid_idx * workload_type_num
                    operation_flag = 'f'
                else:
                    raise("UNKOWN OPERATION FLAG")
        # self.print_schedule()

    def print_schedule(self):
        for s in self.schedule:
            for w in s:
                print(f"{w[0].name}{w[1]}", end=" ")
            print()

    def print_stages(self):
        for device in self.devices:
            print("Device ID:{}".format(device.did))
            if device.did == 7:
                device.show_stages(detail_info=True)

    def print_workload_schedule(self):
        for k in self.results:
            print(k, self.results[k])

    def get_device_execution_time(self):
        res = [
            device.finish_time for device in self.devices
        ]
        return res

    def get_device_bubble_time(self):
        res = [
            device.idle_time for device in self.devices
        ]
        return res

    def get_model_partition(self):
        return self.partition
    
    def get_model_placement(self):
        return self.placement
    
    def print_partition_placement(self):
        for did, p in enumerate(self.placement):
            print(f"Device {did} has {len(p)} layers, {p}")
        if self.solver_results:
            print(f"Workloads on each Device:{self.solver_results['loads']}, var:{self.solver_results['variance']}")
            print(f"Layer assignments:{self.solver_results['assignments']}")

    def print_device_utilization(self, total_time):
        avg_bubble = 0
        bubble_ratios = []
        for device in self.devices:
            bubble_ratio = round(device.idle_time / total_time, 4)
            bubble_ratios.append(round(bubble_ratio*100,2))
            avg_bubble += bubble_ratio
        print(f"{self.schedule_method.name}:\tTime = {total_time}\t Avg bubble ratio = {round(avg_bubble/len(self.devices)*100,2)}\tDevice bubble ratios = {bubble_ratios} ")

    def print_memory_footprint(self, device_id=(0,), show_mem=True):
        peak_mem_usages = [0 for _ in range(len(self.devices))]
        for device in self.devices:
            aim_file_path = "schedule_results/memory/device{}.txt".format(device.did)
            save_to_file(aim_file_path, "Device {} mem usage:\n".format(device.did), mode='w')
            last_mem_record = 0
            for t, mem_record in device.mem_usage_record.items():
                (peak_mem, wtype, sid, mid) = device.peak_mem_usage_record[t]
                oom_flag = "" if mem_record <= device.max_memory else "OOM"
                save_to_file(aim_file_path, "Time {} {}, mem = {}, {}, peak = {}, {}.\n".format(t, (wtype, sid, mid), round(mem_record,2), round((mem_record - last_mem_record), 2), round(peak_mem, 2), oom_flag), 'a')
                last_mem_record = mem_record
                peak_mem_usages[device.did] = round(max(peak_mem_usages[device.did], peak_mem), 3)
        
        oom = False
        for did, device in enumerate(self.devices):
            if device.max_memory < peak_mem_usages[did]:
                if show_mem:
                    print(f"Out of Memory in Device {device.did}. ({peak_mem_usages[did]} > {device.max_memory})")
                oom = True

        if show_mem:
            print(peak_mem_usages)
        return not oom
    
    def update_constraints_within_pipeline(self, time, constraint: Workload):
        for device in self.devices:
            device.update_constraints_within_device(time, constraint=constraint)

    def record_workload(self, workload: Workload):
        if workload is None:
            return

        wlt = workload.wtype_str  # 在 Workload 中预缓存小写字符串
        mid = workload.mid
        sid = workload.sid
        did = workload.did

        # 用 + 拼接比 format 快约 25–35%
        k = wlt + '_' + str(mid) + '_' + str(sid) + '_' + str(did)

        self.results[k] = workload.start_time

        end_time = workload.start_time + workload.duration
        last = self.last_workload
        if last is None or end_time > (last.start_time + last.duration):
            self.last_workload = workload

        if gpc["HEAD_DP"]:
            if sid == self.stage_num:
                for d in self.devices:
                    if d.did == did: continue
                    for s in d.stages.keys():
                        if s == sid:
                            if mid in d.stages[s].workloads.keys():
                                d.stages[s].workloads.pop(mid)

    def update_workload_execution_record(self):
        for device in self.devices:
            device.workload_execute_record = self.workload_execute_record 

    def check_workload_status(self, time):
        for device in self.devices:
            if device.state == Device.BUSY and time >= device.current_workload.end_time:
                if device.current_workload.wtype == WorkloadType.W:
                    self.num_finished_microbatch += 1
                elif not self.bwd_split and device.current_workload.wtype == WorkloadType.B:
                    if device.current_workload.duration > 0:
                        self.num_finished_microbatch += 1
                self.workload_execute_record[device.current_workload.did].append(device.current_workload)

                device.current_workload.complete(time=time)
                device.finish_time = time
                self.update_constraints_within_pipeline(time=time, constraint=device.current_workload)
                device.update_memory_usage()
                device.state = Device.IDLE

    def execute_workload(self, time):
        for device in self.devices:
            processing_workload = device.execute_workload(run_schedule=self.run_schedule,time=time)
            self.record_workload(processing_workload)

    def update_time(self):
        self.time += 1

    def reset_time(self):
        self.time = 0

    def get_time(self):
        return self.time

    def check_device_status(self, time):
        idle_num = 0
        for device in self.devices:
            if device.state == Device.IDLE:
                device.idle_time += 1
                idle_num += 1
        
        # NOTE: Add missed finished cases caused by transferring micro-batches to other pipelines
        if time > 0 and not self.finish_flag and idle_num == len(self.devices):
            self.finish_flag = True

    def run_pipeline_parallelism(self, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=True, show_success=True):
        # self.run_schedule = False
        self.reset_time()
        while self.get_time() <= time_limit and not self.finish_flag:
            self.check_workload_status(time=self.time)
            self.execute_workload(time=self.time)
            self.check_device_status(time=self.time)
            self.update_time()
        if self.finish_flag:
            if show_success:
                print("Success")
            if show_utilization:
                self.print_device_utilization()
            self.record_recomputation_config()
            if not self.print_memory_footprint(show_mem=show_mem):
                if show_mem:
                    print("Fail due to OOM")
            else:
                self.temp_results = copy.deepcopy(self.results)
        else:
            if show_success:
                print("Fail")

        return self.last_workload.end_time

    def pop_workload(self, mid_group = None, did_group = None, pop_wtypes = [WorkloadType.F, WorkloadType.B, WorkloadType.W]):
        workloads = []
        assert mid_group is None or type(mid_group) is list
        assert did_group is None or type(did_group) is list
        did_group = range(self.device_num) if did_group is None else did_group
        for did in did_group:
            device = self.devices[did]
            for sid, stage in device.stages.items():
                for wid, workload in stage.workloads.items():
                    if wid in mid_group:
                        for wtype in pop_wtypes:
                            workloads.append(workload[wtype])
                for mid in mid_group:
                    stage.workloads.pop(mid)
            for mid in mid_group:
                device.held_mids.discard(mid)
        return workloads

    def insert_workload(self, workloads:list[Workload], did_group=None):
        assert did_group is None or type(did_group) is list
        self.nmb += 1
        for workload in workloads:
            wtype = workload.wtype
            device = self.devices[workload.did]
            if device.did in did_group:
                stage = device.stages[workload.sid]
                mid = workload.mid
                if mid not in stage.workloads:
                    stage.workloads[mid] = {}
                stage.workloads[mid][wtype] = workload
                workload.duration = workload.duration * workload.comp_power / device.comp_power
                device.held_mids.add(mid)

    def get_completed_workload_count(self):
        count = 0
        for device in self.devices:
            count += device.get_completed_workload_count_by_type(wtype=WorkloadType.W)
        return count

    def get_workloadload_duration(self):
        f_times = {}
        b_times = {}
        w_times = {}
        for device in self.devices:
            for sid in device.stages:
                if sid not in f_times:
                    f_times[sid] = {}
                if sid not in b_times:
                    b_times[sid] = {}
                if sid not in w_times:
                    w_times[sid] = {}
                for mid in range(self.bs):
                    if mid not in device.stages[sid].workloads: continue
                    f_times[sid][mid] = device.stages[sid].workloads[mid][WorkloadType.F].duration
                    b_times[sid][mid] = device.stages[sid].workloads[mid][WorkloadType.B].duration
                    if self.bwd_split:
                        w_times[sid][mid] = device.stages[sid].workloads[mid][WorkloadType.W].duration
                    else:
                        w_times[sid][mid] = 0
        return f_times, b_times, w_times
         
    def draw(self) -> None:
        # 绘制结果的逻辑
        f_times, b_times, w_times = self.get_workloadload_duration()
        res = {}
        for key in self.results:
            if str(key).startswith(("f_","b_","w_","r_")):
                res[key] = self.results[key]
        painter_conf = {
            "device_num": self.device_num,
            "devices": self.placement,
            "stage_num": self.stage_num if not gpc["HEAD_DP"] else self.stage_num + 1,
            "pp_height": gpc["PP_HEIGHT"],
            "pp_align": gpc["PP_ALIGN"],
            "pixel_base": gpc["PIXEL_BASE"],
            "nmb": self.nmb,
            "mid_offset": self.mid_offset,
            "f_times": dict_to_2d_list(f_times),
            "b_times": dict_to_2d_list(b_times),
            "w_times": dict_to_2d_list(w_times),
            "comm_length": [gpc["COMM_TIME"] for _ in range(self.stage_num)],
        }
        SP(painter_conf).draw(res)