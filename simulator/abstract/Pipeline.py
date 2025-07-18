from .Device import *
from .Stage import Stage
from .Workload import Workload
from .mutils import *
from ..painter import SchedulingPainter as SP
from ..LayerwisePainter import LayerwiseSchedulingPainter as LSP
from ..utils import save_to_file
from .Placement import PipelinePlacement
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

    def __init__(self, pipeline_idx, placement=None, run_schedule=False) -> None:
        
        self.time = 0
        self.pipeline_idx = pipeline_idx # A flag for identifying each pipeline
        self.results = {}
        self.device_num = gpc["DEVICE_NUM"]
        self.layer_num = gpc["LAYER_NUM"]
        self.devices: list[Device] = []
        self.placement = [] if not placement else placement
        # 222 0
        # 145 A
        # 152 AB
        # 130 ABC
        # Standard 1F1B 4034 666 
        # Standard I1F1B 3662
        # ZBH 3638 616
        # Alpa gpt3
        # self.placement = [
        #     [1,1,1,1,1,1,1,1],
        #     [1,1,1,1,1,1,1,1],
        #     [1,1,1,1,1,1,1,1],
        #     [1,1,1,1,1,1,1,1],
        # ]
        # Alpa ds
        # self.placement = [
        #     [1,1,1,1,1,1,1,1,1,1],
        #     [1,1,1,1,1,1,1,1],
        #     [1,1,1,1,1,1,1,1],
        #     [1,1,1,1,1,1],
        # ]
        # Alpa llama3
        # self.placement = [
        #     [1,1,1,1,1,1,1,1],
        #     [1,1,1,1,1,1,1,1,1],
        #     [1,1,1,1,1,1,1,1,1],
        #     [1,1,1,1,1,1],
        # ]
        # Alpa gemma
        self.placement = [
            [1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1],
        ]
        # UPP 127 148 172 221
        #     1.74 1.49 1.28
        # self.placement = [ # 3002 475
        #     [0, 4, 7, 11, 14, 18, 21, 25, 28],
        #     [1, 5, 8, 12, 15, 19, 22, 26, 29],
        #     [2, 6, 9, 13, 16, 20, 23, 27, 30],
        #     [3, 10, 17, 24, 31],
        # ]
        # self.placement = [ # 466
        #     [0, 4, 8, 12, 16, 19, 22, 25, 28],
        #     [1, 5, 9, 13, 17, 20, 23, 26, 29],
        #     [2, 6, 10, 14, 18, 21, 24, 27, 30],
        #     [3, 7, 11, 15, 31],
        # ]
        # self.placement = [ # 464 Gemma
        #     [0, 4, 8, 12, 16, 19, 22, 25, 28],
        #     [1, 6, 9, 13, 17, 20, 23, 26, 29],
        #     [2, 7, 10, 14, 18, 21, 24, 27, 30],
        #     [3, 5, 11, 15, 31],
        # ]
        # self.placement = [ # 464 llama3
        #     [0, 4, 8, 12, 16, 20, 23, 26],
        #     [1, 5, 9, 13, 17, 21, 24, 27, 29],
        #     [2, 6, 10, 14, 18, 22, 25, 28, 30],
        #     [3, 7, 11, 15, 19, 31],
        # ]
        # self.placement = [ # 464 ds
        #     [0, 4, 8, 12, 16, 20, 23, 26],
        #     [1, 5, 9, 13, 17, 21, 24, 27, 29],
        #     [2, 6, 10, 14, 18, 22, 25, 28, 30],
        #     [3, 7, 11, 15, 19, 31],
        # ]
        # self.placement = [ # 464 gpt3
        #     [0, 4, 8, 12, 16, 20, 24, 31],
        #     [1, 5, 9, 13, 17, 21, 25, 30],
        #     [2, 6, 10, 14, 18, 22, 26, 29],
        #     [3, 7, 11, 15, 19, 23, 27, 28],
        # ]
        # self.placement = [ # 3002 483
        #     [0, 4, 8, 12, 31],
        #     [1, 5, 9, 13, 16, 19, 22, 25, 28],
        #     [2, 6, 10, 14, 17, 20, 23, 26, 29],
        #     [3, 7, 11, 15, 18, 21, 24, 27, 30],
        # ]
        # self.placement = [[0, 3, 6, 9, 12, 15, 19, 23, 27], [1, 4, 7, 10, 13, 16, 20, 24, 28], [2, 5, 8, 11, 14, 17, 21, 25, 29], [18, 22, 26, 30, 31]]
        # self.placement = [ # 2901 FBW 2798 BFW! 485 BFW
        #     [0, 7, 14, 21, 31],
        #     [1, 4, 8, 11, 15, 18, 22, 25, 28],
        #     [2, 5, 9, 12, 16, 19, 23, 26, 29],
        #     [3, 6, 10, 13, 17, 20, 24, 27, 30],
        # ]

        # self.placement = [ # 1991
        #     [0, 2, 6, 15],
        #     [1, 3, 7, 10, 11, 14],
        #     [4, 8, 12],
        #     [5, 9, 13],
        # ]
        # self.placement = [ # 1991
        #     [0, 1, 2, 3,4,5,6,7,8],
        #     [0, 1, 2, 3,4,5,6,7,8,9],
        #     [0, 1, 2, 3,4,5,6,7,8,9],
        #     [0, 1, 2],
        # ]
        # self.placement = [ # 2132
        #     [0, 2, 15],
        #     [1, 3, 6, 7, 10, 11, 14],
        #     [4, 8, 12],
        #     [5, 9, 13],
        # ]
        # self.placement = [ # 2080
        #     [0, 2, 6, 8, 9, 12],
        #     [1, 3, 7, 15],
        #     [4, 10, 13],
        #     [5, 11, 14],
        # ]
        self.nmb = gpc["MICRO_BATCH_NUM"]
        self.stage_num = gpc["STAGE_NUM"]
        self.schedule_method = gpc["SCHEDULE_METHOD"]
        self.layer_wise = gpc["LAYERWISE"]
        self.head_dp = gpc["HEAD_DP"]
        self.microbatch_schedule_range = range(0,min(gpc["SCHEDULE_UNIT"], self.nmb))
        self.acc_finished_mb = 0
        self.finish_flag = False
        self.num_finished_microbatch = 0
        self.run_schedule = run_schedule
        self.manual_recomp_set = []
        # self.manual_recomp_set = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1]
        
        self.fail_indexes = set()
        # pp4 tp4 zero4 I1F1B recomp set
        # self.manual_recomp_set = [0 for _ in range(self.layer_num)]
        # self.manual_recomp_set[2] = 1
        # self.manual_recomp_set[3] = 1
        # self.manual_recomp_set[6] = 1
        # self.manual_recomp_set[7] = 1
        # self.manual_recomp_set[11] = 1
        # self.manual_recomp_set[15] = 1

        min_value = min(list(reversed(range(self.layer_num))))
        max_value = max(list(reversed(range(self.layer_num))))

        # 缩放到 DENSITY_MIN 和 DENSITY_MAX 之间
        self.layer_density = [
            gpc["DENSITY_MIN"] + (value - min_value) * (gpc["DENSITY_MAX"] - gpc["DENSITY_MIN"]) / (max_value - min_value)
            for value in list(reversed(range(self.layer_num)))
        ]
        self._init_stage()
        self.set_microbatch_schedule_range(microbatch_schedule_range=self.microbatch_schedule_range)
        self.schedule = [[] for _ in range(self.device_num)]
        self.generate_schedule()
        self.set_schedule()
        self.temp_results = {}
        self.recomp_set_traverser = self.generate_binary_combinations()
        self.last_workload: Workload = None
        self.workload_execute_record: list[list[Workload]] = [[] for _ in range(self.device_num)]
        if run_schedule:
            print("Read schedule generated before...")
            self.file2result()
            self.result2schedule()
            self.set_schedule()

    def _sid2did(self, sid):
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
            did = self._sid2did(sid=sid)
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


    def show_detail_info(self):
        for device in self.devices:
            print("Device ID:{}".format(device.did))
            if device.did == 7:
                device.show_stages(detail_info=True)

    def set_microbatch_schedule_range(self, microbatch_schedule_range):
        for device in self.devices:
            device.mid_traverse_order = microbatch_schedule_range

    def record_recomp_set(self):
        for idx, r in enumerate(self.recomp_set):
            self.recomp_set[idx] = 1 if r else 0
            self.results[f"theta_{idx}"] = r

    def _init_stage(self):
        dev_compute_power = []
        layer_num = self.layer_num // self.stage_num
        for did in range(self.device_num):
            max_mem = gpc["GPU_MAX_MEM"]
            comp_power = 2
            if gpc["HETER_DEVICE"]:
                if did >= self.device_num // 2:
                    max_mem = gpc["GPU_MAX_MEM"] / gpc["HETER_RATIO"]
                    comp_power = comp_power / gpc["HETER_RATIO"]
            device = Device(
                        device_id = did, 
                        max_activation_counts=gpc["MAX_ACTIVATION_COUNTS"], 
                        nmb=self.nmb,
                        memory_usage_constrain_rate=0.85,
                        max_mem=max_mem,
                        comp_power=comp_power,
                        layer_density=self.layer_density,
                    )
            dev_compute_power.append(comp_power)
            self.devices.append(device)
        self.set_recomp()
        if gpc["HETER_DEVICE"] and not self.placement and self.schedule_method not in (Schedule.STANDARD_INTERLEAVED, Schedule.STANDARD_1F1B, Schedule.ZBV, Schedule.STANDARD_ZBH1):
            layer_computation_cost = [1 for _ in range(self.layer_num)]
            if self.layer_wise:
                layer_computation_cost.append(gpc["HEAD_F_TIME"] / gpc["F_TIME"])
                layer_computation_cost.append(gpc["CE_F_TIME"] / gpc["F_TIME"])
            else:
                layer_computation_cost[-1] += (gpc["HEAD_F_TIME"] + gpc["CE_F_TIME"]) / gpc["F_TIME"] * 0
            total_layer = self.layer_num if not self.layer_wise else self.layer_num + 2
            self.pipeline_placement_solver = PipelinePlacement(
                layer_num=total_layer,
                layer_computation_cost=layer_computation_cost,
                layer_para=[1 for _ in range(total_layer)],
                dev_num=self.device_num,
                dev_max_memory=[100000 for _ in range(total_layer)],
                dev_compute_power=dev_compute_power,
            )
            if not self.placement:
                self.placement = self.pipeline_placement_solver.get_placements()
        if self.placement and (self.schedule_method == Schedule.STANDARD_1F1B or self.schedule_method == Schedule.STANDARD_ZBH1):
            assert self.placement is not None
            for did in range(self.device_num):
                self.devices[did].add_stage(did, layer_num = len(self.placement[did]), recomp=self.recomp_set[did])
            print("Alpa")
        elif self.placement and self.schedule_method == Schedule.UnifiedPP and CHUNK_NUM == 1:
            for did in range(self.device_num):
                self.devices[did].add_stage(did, layer_num = len(self.placement[did]), recomp=self.recomp_set[did])
            print("S1F1B + model partition + workload scheduling")
        elif self.placement:
            offset = 0
            if self.layer_wise:
                offset = 1
            for did in range(self.device_num):
                for pid in self.placement[did]:
                    self.devices[did].add_stage(pid + offset, layer_num = layer_num, recomp=self.recomp_set[pid + offset])
            if self.layer_wise:
                self.devices[self.device_num - 1].add_stage(0, layer_num = layer_num, recomp=self.recomp_set[0])
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
            if self.schedule_method in (Schedule.STANDARD_1F1B, Schedule.STANDARD_INTERLEAVED):
                for pid in range(self.stage_num):
                    self.devices[pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)
            elif gpc["STAGE_PLACEMENT"] == Placement.SEARCHED:
                print("Use Searched placement")
                # for pid in range(self.stage_num - self.device_num):
                #     self.devices[pid % self.device_num].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)
                # for pid in range(self.stage_num - self.device_num, self.stage_num):
                #     self.devices[self.device_num - (pid % self.device_num) - 1].add_stage(pid, recomp=self.recomp_set[pid], layer_num = layer_num)
                
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
            print(list(self.devices[did].stages.keys()))
            self.placement.append(list(self.devices[did].stages.keys()))

        save_to_file(gpc["TEMP_PLA_PATH"], str(self.placement), 'w')
        save_to_file(gpc["PLA_FILE_PATH"], str(self.placement), 'w')

    def set_recomp(self):
        if self.layer_wise:
            self.recomp_set = [1 if gpc["RECOMP"] else 0 for _ in range(self.layer_num + 3)]
        else:
            self.recomp_set = [1 if gpc["RECOMP"] else 0 for _ in range(self.stage_num)]
        if self.manual_recomp_set:
            print("Use manual recomp set")
            self.recomp_set = self.manual_recomp_set
            return
        print("Set recomputation")
    def set_schedule(self):
        for did in range(self.device_num):
            self.devices[did].static_schedule = self.schedule[did]

    def generate_schedule(self):
        if self.schedule_method == Schedule.STANDARD_1F1B:
            self.generate_1f1b_schedule()
            print("Generate STANDARD_1F1B Schedule.")

        elif self.schedule_method == Schedule.STANDARD_AFAB:
            self.generate_afab_schedule()
            print("Generate STANDARD_AFAB Schedule.")

        elif self.schedule_method == Schedule.STANDARD_ZBH1:
            self.generate_zbh1_schedule()
            print("Generate STANDARD_ZBH1 Schedule.")

        elif self.schedule_method == Schedule.STANDARD_INTERLEAVED and not self.layer_wise:
            self.generate_interleaved_1f1b_schedule()
            print("Generate STANDARD_INTERLEAVED Schedule.")
        else:
            print("Using UPP Schedule.")

    def generate_afab_schedule(self):
        assert gpc["CHUNK_NUM"] == 1
        print("Generate standard AFAB schedule...")
        workload_type_order = [WorkloadType.F, WorkloadType.B]
        if gpc["SPLIT_BACKPROP"]:
            workload_type_order.append(WorkloadType.W)

        for did in range(self.device_num):
            mids = [0 for _ in range(gpc["WORKLOAD_TYPE_NUM"])]
            for i in range(gpc["WORKLOAD_TYPE_NUM"]):
                while mids[i] < self.nmb:
                    self.schedule[did].append((workload_type_order[i], mids[i], did))
                    mids[i]+=1

    def generate_1f1b_schedule(self):
        assert gpc["CHUNK_NUM"] == 1
        workload_type_order = [WorkloadType.B, WorkloadType.W, WorkloadType.F] if SPLIT_BACKPROP else [WorkloadType.B, WorkloadType.F]
        workload_idx_in_mids = {WorkloadType.F: 0, WorkloadType.B : 1, WorkloadType.W : 2}
        for did in range(self.device_num):
            mids = [0 for _ in range(gpc["WORKLOAD_TYPE_NUM"])]
            # warmup
            while mids[0] < self.device_num - did:
                self.schedule[did].append((WorkloadType.F, mids[0], did))
                mids[0] += 1
            
            iter = 0
            finish_flag = [0 for _ in range(gpc["WORKLOAD_TYPE_NUM"])]
            while sum(finish_flag) < gpc["WORKLOAD_TYPE_NUM"]:
                next_workload_type = workload_type_order[iter % gpc["WORKLOAD_TYPE_NUM"]]
                next_mid = mids[workload_idx_in_mids[next_workload_type]]
                if next_mid < self.nmb:
                    self.schedule[did].append((next_workload_type, next_mid, did))
                    mids[workload_idx_in_mids[next_workload_type]] += 1
                else:
                    finish_flag[workload_idx_in_mids[next_workload_type]] = 1
                iter+=1
        for schedule in self.schedule:
            for w in schedule:
                print(w[0].name, end="")
            print()

    def generate_zbh1_schedule(self):
        assert gpc["WORKLOAD_TYPE_NUM"] == 3

        workload_type_order = [WorkloadType.B, WorkloadType.W, WorkloadType.F]
        workload_idx_in_mids = {WorkloadType.F: 0, WorkloadType.B : 1, WorkloadType.W : 2}
        for did in range(self.device_num):
            mids = [0 for _ in range(gpc["WORKLOAD_TYPE_NUM"])]
            # warmup, should not be simplified
            while mids[0] < self.device_num - did:
                self.schedule[did].append((WorkloadType.F, mids[0], did))
                mids[0] += 1
            # Inject as much as possible F with limited max activation counts
            if gpc["MAX_ACTIVATION_COUNTS"] > self.stage_num * gpc["CHUNK_NUM"]:
                comm_delay = (self.device_num - did - 1) * 2 * gpc["COMM_TIME"]
                compute_delay = (self.device_num - did - 1) * gpc["B_TIME"]
                additional_f_num = min(gpc["MAX_ACTIVATION_COUNTS"] - mids[0] - did, (comm_delay + compute_delay) // gpc["F_TIME"])
                while mids[0] < min(gpc["MAX_ACTIVATION_COUNTS"], self.nmb) and additional_f_num:
                    self.schedule[did].append((WorkloadType.F ,mids[0], did))
                    mids[0] += 1
                    additional_f_num -= 1

            # steady + cooldown
            iter = 0
            finish_flag = [0 for _ in range(gpc["WORKLOAD_TYPE_NUM"])]
            while sum(finish_flag) < gpc["WORKLOAD_TYPE_NUM"]:
                next_workload_type = workload_type_order[iter % gpc["WORKLOAD_TYPE_NUM"]]
                next_mid = mids[workload_idx_in_mids[next_workload_type]]
                if mids[0] < min(self.nmb, gpc["MAX_ACTIVATION_COUNTS"]):
                    if next_workload_type == WorkloadType.W:
                        iter += 1
                        continue 
                if next_mid < self.nmb:
                    self.schedule[did].append((next_workload_type, next_mid, did))
                    mids[workload_idx_in_mids[next_workload_type]] += 1
                else:
                    finish_flag[workload_idx_in_mids[next_workload_type]] = 1
                iter+=1

    def generate_interleaved_1f1b_schedule(self):
        workload_type_num = 2
        for did in range(self.device_num):
            sids = list(self.placement[did])
            
            mids = [0 for _ in range(workload_type_num * gpc["CHUNK_NUM"])]
            f_mid_count = 0
            
            f_next_sid_idx = 0
            f_next_sid = sids[f_next_sid_idx]
            idx_in_f_mids = f_next_sid_idx * workload_type_num
        
            # warmup, inject as much microbatches as possible
            warmup_f_num = (gpc["CHUNK_NUM"] - 1) * self.device_num + (self.device_num - did - 1) * 2
            while mids[idx_in_f_mids] < self.nmb and f_mid_count < warmup_f_num:
                self.schedule[did].append((WorkloadType.F ,mids[idx_in_f_mids], f_next_sid))
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
                        self.schedule[did].append((WorkloadType.F ,mids[idx_in_f_mids], f_next_sid))
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
                            self.schedule[did].append((WorkloadType.R ,mids[idx_in_b_mids], b_next_sid))
                        self.schedule[did].append((WorkloadType.B ,mids[idx_in_b_mids], b_next_sid))
                        if gpc["WORKLOAD_TYPE_NUM"] == 3:
                            self.schedule[did].append((WorkloadType.W ,mids[idx_in_b_mids], b_next_sid))
                        mids[idx_in_b_mids] += 1
                        b_mid_count += 1
                        if b_mid_count % self.device_num == 0:
                            b_next_sid_idx = (b_next_sid_idx + 1) % len(bsids)
                            b_next_sid = bsids[b_next_sid_idx]
                            idx_in_b_mids = 1 + b_next_sid_idx * workload_type_num
                    operation_flag = 'f'
                else:
                    raise("UNKOWN OPERATION FLAG")
    
    def show_record(self):
        for k in self.results:
            print(k, self.results[k])

    def update_constraints(self, time, constraint):
        for device in self.devices:
            device.update_constraints(time, constraint=constraint)

    def record_workload(self, workload: Workload):
        if workload:
            wlt = workload.wtype.value.lower()
            mid = workload.mid
            sid = workload.sid
            did = workload.did
            k = '{}_{}_{}_{}'.format(wlt,mid,sid,did)
            self.results[k] = workload.start_time
            if self.last_workload is None or workload.start_time + workload.duration > self.last_workload.start_time + self.last_workload.duration:
                self.last_workload = workload
            if workload.wtype is not WorkloadType.F:
                for device in self.devices:
                    device.overlap_flag = True
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

    def change_mid_traverse_order(self, workload: Workload):
        if workload:
            for device in self.devices:
                device.overlap_aware_executable_workload_reorder(workload=workload)         

    def check_workload_status(self, time):
        for device in self.devices:
            if device._finish_proc_workload(time=time):
                if device.proc_workload.wtype == WorkloadType.W:
                    self.num_finished_microbatch += 1
                    self.acc_finished_mb += 1
                    if self.layer_wise:
                        if self.acc_finished_mb == (1 + self.layer_num) * self.nmb:
                            self.finish_flag = True
                    elif gpc["HEAD_DP"]:
                        if self.acc_finished_mb == (1 + self.stage_num) * self.nmb:
                            self.finish_flag = True
                    else:
                        if self.acc_finished_mb == self.stage_num * self.nmb:
                            self.finish_flag = True
                if not gpc["SPLIT_BACKPROP"] and device.proc_workload.wtype == WorkloadType.B:
                    if device.proc_workload.duration > 0:
                        self.num_finished_microbatch += 1
                        self.acc_finished_mb += 1
                    if self.layer_wise:
                        if self.acc_finished_mb == (1 + self.layer_num) * self.nmb:
                            self.finish_flag = True
                    else:
                        if self.acc_finished_mb == self.stage_num * self.nmb:
                            self.finish_flag = True 
                self.workload_execute_record[device.proc_workload.did].append(device.proc_workload)
                self.update_workload_execution_record()

                device.proc_workload.complete(time=time)
                self.update_constraints(time=time, constraint=device.proc_workload)
                device.update_memory_usage()
                device.state = Device.IDLE

        if self.num_finished_microbatch == (1 + self.layer_num) * len(self.microbatch_schedule_range):
            self.num_finished_microbatch = 0
            self.microbatch_schedule_range = [n + len(self.microbatch_schedule_range) for n in self.microbatch_schedule_range if n + len(self.microbatch_schedule_range) < self.nmb]
            self.set_microbatch_schedule_range(microbatch_schedule_range=self.microbatch_schedule_range)

    def execute_workload(self, time):
        for device in self.devices:
            processing_workload = device.execute_workload(run_schedule=self.run_schedule,time=time)
            self.record_workload(processing_workload)
            
    def reduce_recomp_degree(self):
        
        self.manual_recomp_set = self.recomp_set
        for index, value in reversed(list(enumerate(self.manual_recomp_set))):
            if value:
                if index not in self.fail_indexes:
                    self.manual_recomp_set[index] = 0
                    print("Index {}".format(index))
                    return index
        print("Already the best recomp config.")
        return -1
    
    def add_recomp_degree(self):
        recomp_set = self.recomp_set
        for index, value in list(enumerate(recomp_set)):
            if not value:
                self.recomp_set[index] = 1
                print("Try the added the recomp degree.")
                return True
        print("Set all stage to recomputing.")
        return False
    
    def reset_run_para(self):
        self.results = {}
        self.devices: list[Device] = []
        self.placement = []
        self.acc_finished_mb = 0
        self.finish_flag = False
        self.num_finished_microbatch = 0
        self._init_stage()
        self.reset_time()
    
    def generate_binary_combinations(self):
        """
        生成所有长度为 layer_num 的二进制组合0 或 1。
        每次调用 next() 返回一个未返回过的情况。
        """
        # 使用 itertools.product 生成所有可能的组合
        combinations = itertools.product([1, 0], repeat=self.layer_num)
        return combinations

    def recomp_set_check(self, recomp_set:list):
        # [43.37751770019531, 43.25251770019531, 43.12751770019531, 43.00251770019531, 42.87751770019531, 42.75251770019531, 42.62751770019531, 72.66658020019531]
        # Activation Layer=6.0625,Activation Input=0.0625,Activation Loss=4.640625
        # Gradient Input=1.5001983642578125,Gradient Parameters=1.5001983642578125,Gradient Head=2.3203125
        # LOSS=4.640625,VOC=152064
        # LAYER_MEM:1.5001983642578125
        # MODEL MEM:15.0
        # OPT MEM:11.0
        if self.schedule_method == Schedule.STANDARD_INTERLEAVED:
            if sum(recomp_set[self.device_num-1::self.device_num]) < 8:
                return False
            # cut branch for 1st device
            if sum(recomp_set[0::self.device_num]) != 7 and recomp_set[0] != 1:
                return False
            
            last_recomp_num = self.layer_num + 1
            for did in range(self.device_num):
                layer_num_wo_recomp = recomp_set[did::self.device_num]
                if sum(layer_num_wo_recomp) <= 7:
                    return False
                if sum(layer_num_wo_recomp) > last_recomp_num:
                    return False
                last_recomp_num = sum(layer_num_wo_recomp)
        return True

    def update_time(self):
        self.time += 1

    def reset_time(self):
        self.time = 0

    def get_time(self):
        return self.time

    def check_device_states(self):
        for device in self.devices:
            if device.state == Device.IDLE:
                # if device.exe_num_w != device.nmb * len(device.stages):
                device.idle_time += 1
        #     print(device.state, end=" ")
        # print()

    def print_device_utilization(self):
        avg_bubble = 0
        for device in self.devices:
            bubble_ratio = round(device.idle_time / self.get_time(), 4)
            print(f"Device {device.did} idle time : {device.idle_time}, idle ratio: {bubble_ratio*100}")
            avg_bubble += bubble_ratio
        print(f"Avg bubble ratio: {avg_bubble/len(self.devices)*100}")

    def run_pipeline_parallelism(self, time_limit = gpc["TIME_LIMIT"]):
        # self.run_schedule = False
        self.reset_time()
        while self.get_time() <= time_limit and not self.finish_flag and not gpc["TERMINAL_FLAG"]:
            self.check_workload_status(time=self.time)
            self.execute_workload(time=self.time)
            self.check_device_states()
            self.update_time()
        if self.finish_flag:
            print("Success")
            self.print_device_utilization()
            self.record_recomp_set()
            if not self.show_mem_usage():
                print("Fail due to OOM")
            else:
                self.temp_results = copy.deepcopy(self.results)
        else:
            print("Fail")

        if AUTO_RECOMP_SEARCH:
            # self.record_recomp_set()
            # self.result2file()
            self.run_schedule = True
            fail_times = 0
            while fail_times < self.device_num:
                idx = self.reduce_recomp_degree()
                if idx == -1:
                    break
                self.reset_run_para()
                print("Read schedule generated before...")
                # self.file2result()
                # self.result2schedule()
                self.set_schedule()

                print(self.recomp_set)
                
                while self.get_time() <= time_limit and not self.finish_flag:
                    self.check_workload_status(time=self.time)
                    self.execute_workload(time=self.time)
                    self.update_time()
                if self.finish_flag:
                    self.record_recomp_set()
                    if not self.show_mem_usage():
                        print("Fail OOM")
                        fail_times += 1
                        self.fail_indexes.add(idx)
                        self.recomp_set[idx] = 1
                    else:
                        print("Success")
                        self.temp_results = copy.deepcopy(self.results)
                        print("Reset fail times to 0.")
                        fail_times = 0
                else:
                    print("Fail")
                    self.results = self.temp_results
                    break
            
            self.reset_run_para()
            self.results = self.temp_results
            # self.result2schedule()
            self.set_schedule()
            while self.get_time() <= time_limit and not self.finish_flag:
                self.check_workload_status(time=self.time)
                self.execute_workload(time=self.time)
                self.update_time()
            if self.finish_flag:
                print("Success")
                self.record_recomp_set()
                self.result2file()
            else:
                print("Wrong answer!")
        return self.last_workload.end_time

    def show_mem_usage(self, device_id=(0,), show_all=False):
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
                print(f"Out of Memory in Device {device.did}. ({peak_mem_usages[did]} > {device.max_memory})")
                oom = True

        print(peak_mem_usages)
        return not oom
    
    def get_workloadload_duration(self):
        fwd_time = [gpc["F_TIME"] for _ in range(self.layer_num+3)]
        iwd_time = [gpc["B_TIME"] for _ in range(self.layer_num+3)]
        pwd_time = [gpc["W_TIME"] for _ in range(self.layer_num+3)]
        for device in self.devices:
            for sid in device.stages:
                for mid in range(self.nmb):
                    if mid not in device.stages[sid].workloads: continue
                    fwd_time[sid] = device.stages[sid].workloads[mid][WorkloadType.F].duration
                    if WorkloadType.B in device.stages[sid].workloads[mid]:
                        iwd_time[sid] = device.stages[sid].workloads[mid][WorkloadType.B].duration
                    if WorkloadType.W in device.stages[sid].workloads[mid]:
                        pwd_time[sid] = device.stages[sid].workloads[mid][WorkloadType.W].duration
        return fwd_time, iwd_time, pwd_time
    
    def get_workload_len(self, key):
        workload_type, mid, lid = key.split("_")
        mid = int(mid)
        lid = int(lid)
        if self.layer_wise:
            layers = 1
        else:
            layers = self.layer_num // self.stage_num

        if workload_type == "f":
            workload_len = gpc["F_TIME"] * layers
            if self.layer_wise:
                if lid == 0:
                    workload_len = gpc["EMB_F_TIME"]
                elif lid == self.layer_num - 1:
                    workload_len = gpc["CE_F_TIME"]
                elif lid == self.layer_num - 2:
                    workload_len = gpc["HEAD_F_TIME"]
            else:
                if lid == 0:
                    workload_len += gpc["EMB_F_TIME"]
                elif lid == self.stage_num - 1:
                    workload_len += gpc["CE_F_TIME"] + gpc["HEAD_F_TIME"]
        elif workload_type == "b":
            workload_len = gpc["B_TIME"] * layers
            if self.layer_wise:
                if lid == self.layer_num - 1:
                    workload_len = gpc["CE_B_TIME"]
                elif lid == self.layer_num - 2:
                    workload_len = gpc["HEAD_B_TIME"]
            else:
                if lid == self.stage_num - 1:
                    workload_len += gpc["CE_B_TIME"] + gpc["HEAD_B_TIME"]
        elif workload_type == "w":
            workload_len = gpc["W_TIME"] * layers
            if self.layer_wise:
                if lid == self.layer_num - 1:
                    workload_len = gpc["CE_W_TIME"]
                elif lid == self.layer_num - 2:
                    workload_len = gpc["HEAD_W_TIME"]
            else:
                if lid == self.stage_num - 1:
                    workload_len += gpc["CE_W_TIME"] + gpc["HEAD_W_TIME"]
        return workload_len 
         
    def draw(self) -> None:
        # 绘制结果的逻辑
        # self.resort_w()
        fwd_time, iwd_time, pwd_time = self.get_workloadload_duration()
        if self.layer_wise:
            painter_conf = {
                "device_num": self.device_num,
                "devices": self.placement,
                "num_layer": self.layer_num+3,
                "stage_num": self.layer_num+3,
                "pp_height": gpc["PP_HEIGHT"],
                "pp_align": gpc["PP_ALIGN"],
                "pixel_base": gpc["PIXEL_BASE"],
                "nmb": self.nmb,
                "forward_length": fwd_time,
                "backward_length": iwd_time,
                "backward_length2": pwd_time,
                "comm_length": [gpc["COMM_TIME"] for _ in range(self.stage_num)],
            }
            LSP(painter_conf).draw(self.results)
        else:
            res = {}
            for key in self.results:
                if key.startswith(("f_","b_","w_","r_")):
                    res[key] = self.results[key]
            painter_conf = {
                "device_num": self.device_num,
                "devices": self.placement,
                "stage_num": self.stage_num if not gpc["HEAD_DP"] else self.stage_num + 1,
                "pp_height": gpc["PP_HEIGHT"],
                "pp_align": gpc["PP_ALIGN"],
                "pixel_base": gpc["PIXEL_BASE"],
                "nmb": self.nmb,
                "forward_length": fwd_time,
                "backward_length": iwd_time,
                "backward_length2": pwd_time,
                "comm_length": [gpc["COMM_TIME"] for _ in range(self.stage_num)],
            }
            SP(painter_conf).draw(res)