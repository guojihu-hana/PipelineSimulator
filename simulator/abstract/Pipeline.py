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

    def __init__(self, pipeline_idx, nmb, mbs, bs, time=0, mid_offset=None, placement=None, run_schedule=False, comp_power:list=None, max_mem:list=None, executor=None) -> None:
        self.executor = executor
        self.time = time
        self.mbs = mbs
        self.bs = bs
        self.pipeline_idx = pipeline_idx # A flag for identifying each pipeline
        self.results = {}
        self.device_num = gpc["DEVICE_NUM"]
        self.comp_power = comp_power if comp_power else [1 for _ in range(self.device_num)]
        self.max_mem = max_mem if max_mem else [gpc["GPU_MAX_MEM"] for _ in range(self.device_num)]
        self.layer_num = gpc["LAYER_NUM"]
        self.devices: list[Device] = []
        self.nmb = nmb
        self.mid_offset = pipeline_idx * self.nmb if not mid_offset else mid_offset
        
        self.placement = [] if not placement else placement
        self.layer_assignment = []
        self.set_model_partition_and_placement()
        
        print(self.placement)
        self.stage_num = gpc["STAGE_NUM"]
        self.total_workload = self.stage_num * self.nmb
        self.schedule_method = gpc["SCHEDULE_METHOD"]
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

    def set_model_partition_and_placement(self):
        self.layer_assignment = get_octopipe_predefined_partition_placement(seq_len=SEQ_LEN, device_num=self.device_num, layer_num=self.layer_num)
        if gpc["SCHEDULE_METHOD"] in (Schedule.STANDARD_ZBH, Schedule.STANDARD_1F1B, Schedule.STANDARD_AFAB, Schedule.ReCycle):
            self.layer_assignment = [self.layer_num // self.device_num] * self.device_num
        if gpc["SCHEDULE_METHOD"] == Schedule.Mist:
            self.layer_assignment = get_mist_predefined_partition_placement(seq_len=SEQ_LEN, device_num=self.device_num, layer_num=self.layer_num)
            gpc["SCHEDULE_METHOD"] = Schedule.STANDARD_1F1B
        
        self.placement = [
            [1 for _ in range(layer_num)] for layer_num in self.layer_assignment
        ]

        if gpc["SCHEDULE_METHOD"] == Schedule.OctoPipe:
            computation_times = [f + b + w for f, b, w in zip(F_TIMES, B_TIMES, W_TIMES)]
            computation_times[-1] += HEAD_F_TIME + HEAD_B_TIME + HEAD_W_TIME
            if gpc["STAGE_NUM"] == self.layer_num:
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
            self.placement = solver_results["assignments"]
            self.layer_assignment = [len(p) for p in self.placement]

        with open("schedule_results/partition.txt", 'w') as f:
            f.write(str(self.layer_assignment))
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
    # which will break solver constraint (not affect the correctness)
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
        layer_num = self.layer_num // self.stage_num
        for did in range(self.device_num):
            device = Device(
                        did=did,
                        nmb=self.nmb,
                        mid_offset=self.mid_offset,
                        mbs=self.mbs,
                        bs=self.bs,
                        max_mem=self.max_mem[did],
                        comp_power=self.comp_power[did],
                        pipeline=self,
                    )
            self.devices.append(device)
        self.set_recomputation_config()
        if not self.placement and self.schedule_method not in (Schedule.STANDARD_INTERLEAVED, Schedule.STANDARD_1F1B, Schedule.ZBV, Schedule.STANDARD_ZBH):
            layer_computation_cost = [F_TIMES[i]+B_TIMES[i]+W_TIMES[i] for i in range(self.layer_num)]
            head_total = gpc["HEAD_F_TIME"] + gpc["HEAD_B_TIME"] + gpc["HEAD_W_TIME"]
            ce_total = gpc["CE_F_TIME"] + gpc["CE_B_TIME"] + gpc["CE_W_TIME"]
            layer_computation_cost[-1] += head_total + ce_total
            total_layer = self.layer_num
            self.pipeline_placement_solver = PipelinePlacement(
                layer_num=total_layer,
                layer_computation_cost=layer_computation_cost,
                layer_para=[1 for _ in range(total_layer)],
                chunk_num=gpc["CHUNK_NUM"],
                dev_num=self.device_num,
                dev_max_memory=[100000 for _ in range(total_layer)],
                dev_compute_power=self.comp_power,
            )
            if not self.placement:
                self.placement = self.pipeline_placement_solver.get_placements()
        if self.placement and self.schedule_method in (Schedule.STANDARD_1F1B, Schedule.ReCycle, Schedule.STANDARD_ZBH, Schedule.STANDARD_AFAB, Schedule.Mist):
            assert self.placement is not None
            layer_idx_start = 0
            for did in range(self.device_num):
                self.devices[did].add_stage(did, layer_num = len(self.placement[did]), layer_idx_start=layer_idx_start, recomp=self.recomp_set[did])
                layer_idx_start += len(self.placement[did])
        elif self.placement and self.schedule_method == Schedule.OctoPipe and CHUNK_NUM == 1:
            layer_idx_start = 0
            for did in range(self.device_num):
                self.devices[did].add_stage(did, layer_num = len(self.placement[did]), layer_idx_start=layer_idx_start, recomp=self.recomp_set[did])
                layer_idx_start += len(self.placement[did])
        elif self.schedule_method == Schedule.STANDARD_INTERLEAVED:
            layer_idx_start = 0
            for pid in range(self.stage_num):
                self.devices[pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid],layer_idx_start=layer_idx_start, layer_num = layer_num)
                layer_idx_start += layer_num
        elif self.placement:
            layer_idx_start = 0
            for did in range(self.device_num):
                for pid in self.placement[did]:
                    self.devices[did].add_stage(pid, layer_num = layer_num, recomp=self.recomp_set[pid],layer_idx_start=layer_idx_start)
                    layer_idx_start += layer_num
        elif self.layer_wise:
            if gpc["STAGE_PLACEMENT"] == Placement.INTERLEAVED:
                print("Use Interleaved placement")
                for pid in range(self.layer_num):
                    self.devices[pid % self.device_num].add_stage(pid + 1, layer_num = layer_num, recomp=self.recomp_set[pid])
            elif gpc["STAGE_PLACEMENT"] == Placement.RECURRENT:
                print("Use Recurrent placement")
                unit = range(self.device_num)
                orders = []
                while len(orders) < self.layer_num:
                    unit = list(unit)
                    orders += unit[:-1]
                    unit = reversed(unit)

                for pid in range(self.layer_num - 1):
                    self.devices[orders[pid]].add_stage(pid + 1, layer_num = layer_num, recomp=self.recomp_set[pid])
                self.devices[-1].add_stage(self.layer_num, layer_num = layer_num, recomp=self.recomp_set[pid])
            elif gpc["STAGE_PLACEMENT"] == Placement.CROSS:
                print("Use V+I placement")
                for pid in range(self.layer_num):
                    if (pid // (self.device_num)) == self.layer_num // (self.device_num) - 1:
                        self.devices[self.device_num - 1 - pid % self.device_num].add_stage(pid + 1, layer_num = layer_num, recomp=self.recomp_set[pid + 1])
                    else:
                        self.devices[pid % self.device_num].add_stage(pid + 1, layer_num = layer_num, recomp=self.recomp_set[pid + 1])
            else:   
                print("Use Wavelike placement")
                offset = self.device_num if gpc["REVERSE_LAST_STAGES"] else 0
                print(f"Reverse last {offset} stages.")
                for pid in range(self.layer_num - offset):
                    if (pid // self.device_num) % 2 == 0:
                        self.devices[pid % self.device_num].add_stage(pid + 1, layer_num = layer_num, recomp=self.recomp_set[pid + 1])
                    else:
                        self.devices[self.device_num - 1 - pid % self.device_num].add_stage(pid + 1, layer_num = layer_num, recomp=self.recomp_set[pid + 1])
                for pid in range(self.layer_num - offset, self.layer_num):
                    self.devices[pid % self.device_num].add_stage(pid + 1, layer_num = layer_num, recomp=self.recomp_set[pid + 1])

            if gpc["STAGE_PLACEMENT"] != Placement.RECURRENT:
                self.devices[-1].add_stage(0, layer_num = layer_num)
                self.devices[0].add_stage(self.layer_num+1, layer_num = layer_num)
                self.devices[1].add_stage(self.layer_num+2, layer_num = layer_num)
            else:
                self.devices[-1].add_stage(0, layer_num = layer_num)
                self.devices[0].add_stage(self.layer_num+1, layer_num = layer_num)
                self.devices[1].add_stage(self.layer_num+2, layer_num = layer_num)
        else:
            layer_idx_start = 0
            if self.schedule_method == Schedule.STANDARD_INTERLEAVED:
                for pid in range(self.stage_num):
                    self.devices[pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid],layer_idx_start=layer_idx_start, layer_num = layer_num)
                    layer_idx_start += layer_num
            elif self.schedule_method == Schedule.STANDARD_ZBH:
                layer_num = self.layer_num // self.device_num
                assert layer_num == int(layer_num)
                for pid in range(self.device_num):
                    self.devices[did].add_stage(pid, recomp=self.recomp_set[pid],layer_idx_start=layer_idx_start, layer_num = layer_num)
                    layer_idx_start += layer_num
            elif self.schedule_method in (Schedule.STANDARD_1F1B, Schedule.STANDARD_INTERLEAVED):
                for pid in range(self.stage_num):
                    self.devices[pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid],layer_idx_start=layer_idx_start, layer_num = layer_num)
                    layer_idx_start += layer_num
            elif gpc["STAGE_PLACEMENT"] == Placement.SEARCHED:
                print("Use Searched placement")
                for pid in range(self.device_num):
                    self.devices[pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)
                for pid in range(self.device_num, self.device_num * 2):
                    self.devices[self.device_num - (pid % self.device_num) - 1].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)
                for pid in range(self.device_num * 2, self.stage_num):
                    self.devices[pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)
            elif gpc["STAGE_PLACEMENT"] == Placement.INTERLEAVED:
                print("Use Interleaved placement")
                offset = self.device_num if gpc["REVERSE_LAST_STAGES"] else 0
                for pid in range(self.stage_num - offset):
                    self.devices[pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)
                for pid in range(self.stage_num - offset, self.stage_num):
                    self.devices[self.device_num - (pid % self.device_num) - 1].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)
            else:
                assert self.stage_num <= self.layer_num, f"STAGE should be less than LAYER ({self.stage_num} >= {self.layer_num})"                
                offset = self.device_num if gpc["REVERSE_LAST_STAGES"] else 0
                for pid in range(self.stage_num - offset):
                    if (pid // self.device_num) % 2 == 0:
                        self.devices[pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)
                    else:
                        self.devices[self.device_num - 1 - pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)
                for pid in range(self.stage_num - offset, self.stage_num):
                    self.devices[pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)

        if self.head_dp:
            for device in self.devices:
                device.add_stage(self.stage_num, False, False, 0)

        # Launch MemoryMonitors
        for device in self.devices:
            device.init_memory_monitor()

        self.placement = []
        for did in range(self.device_num):
            self.placement.append(list(self.devices[did].stages.keys()))

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
            print("Generate 1F1B Schedule.")
        elif self.schedule_method == Schedule.STANDARD_AFAB:
            self.generate_afab_schedule()
            print("Generate AFAB Schedule.")
        elif self.schedule_method == Schedule.STANDARD_ZBH:
            self.generate_zbh_schedule()
            print("Generate ZBH Schedule.")
        elif self.schedule_method == Schedule.STANDARD_INTERLEAVED:
            self.generate_interleaved_1f1b_schedule()
            print("Generate I-1F1B Schedule.")
        else:
            print(f"Generate {SCHEDULE_METHOD.name} Schedule.")

    def generate_afab_schedule(self):
        assert gpc["CHUNK_NUM"] == 1
        workload_type_order = [WorkloadType.F, WorkloadType.B]
        if gpc["SPLIT_BACKPROP"]:
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
        assert gpc["CHUNK_NUM"] == 1
        workload_type_order = [WorkloadType.B, WorkloadType.W, WorkloadType.F] if gpc["SPLIT_BACKPROP"] else [WorkloadType.B, WorkloadType.F]
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
        assert gpc["SPLIT_BACKPROP"]

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
                if mids[0] < min(self.nmb, gpc["STAGE_NUM"] * gpc["MAX_ACT"]):
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
            
            mids = [0 for _ in range(workload_type_num * gpc["CHUNK_NUM"])]
            f_mid_count = 0
            
            f_next_sid_idx = 0
            f_next_sid = sids[f_next_sid_idx]
            idx_in_f_mids = f_next_sid_idx * workload_type_num
        
            warmup_f_num = (gpc["CHUNK_NUM"] - 1) * self.device_num + (self.device_num - did - 1) * 2
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
            while b_mid_count + f_mid_count < self.nmb * gpc["CHUNK_NUM"] * workload_type_num:
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
                        if gpc["RECOMP"]:
                            self.schedule[did].append((WorkloadType.R ,mids[idx_in_b_mids] + mid_offset, b_next_sid))
                        self.schedule[did].append((WorkloadType.B ,mids[idx_in_b_mids] + mid_offset, b_next_sid))
                        if gpc["SPLIT_BACKPROP"]:
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
    
    def print_stages(self):
        for device in self.devices:
            print("Device ID:{}".format(device.did))
            if device.did == 7:
                device.show_stages(detail_info=True)

    def print_workload_schedule(self):
        for k in self.results:
            print(k, self.results[k])

    def print_device_utilization(self):
        avg_bubble = 0
        for device in self.devices:
            bubble_ratio = round(device.idle_time / self.get_time(), 4)
            print(f"Device {device.did} idle time : {device.idle_time}, idle ratio: {bubble_ratio*100}")
            avg_bubble += bubble_ratio
        print(f"Avg bubble ratio: {avg_bubble/len(self.devices)*100}")

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
            if sid == gpc["STAGE_NUM"]:
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
                elif not gpc["SPLIT_BACKPROP"] and device.current_workload.wtype == WorkloadType.B:
                    if device.current_workload.duration > 0:
                        self.num_finished_microbatch += 1
                self.workload_execute_record[device.current_workload.did].append(device.current_workload)

                device.current_workload.complete(time=time)
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
            print(f"{time}: All devices are idle, set pipeline {self.pipeline_idx} as finished.")

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
                    w_times[sid][mid] = device.stages[sid].workloads[mid][WorkloadType.W].duration
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