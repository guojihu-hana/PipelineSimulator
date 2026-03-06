import time
from gurobipy import Model, GRB, quicksum
from ..painter import SchedulingPainter, MultiPipelinePainter as MPP
from ..abstract.mutils import *
from .chimera_utils import print_to_file
import re

workload_type_mapping = {
    'f':WorkloadType.F,
    'b':WorkloadType.B,
    'w':WorkloadType.W,
}

def duplicate_last_part(text):
        return re.sub(r'(_\d+)$', r'\1\1', text)

class ReCyclePipeline:

    def __init__(self, config: dict, new_comm_length=None) -> None:
        self._num_pipelines = config['pipeline_num']
        self._num_devices = config["device_num"]
        self._num_microbatches = config["microbatch_num"]
        self._mid_offset = [sum(self._num_microbatches[: idx]) for idx, nmb in enumerate(self._num_microbatches)]
        self._num_stages = config["stage_num"]
        self._fail_pipelines_stages = config['fail_pipelines_stages']
        self._file_path = config["file_path"]
        self._time_limit = config["time_limit"]
        self._f_time = config["f_time"]
        self._b_time = config["b_time"]
        self._w_time = config["w_time"]
        self._comp_time_ratio = config["comp_time_ratio"]
        self._comm_time = config["comm_time"] if not new_comm_length else new_comm_length
        # 检查输入参数
        assert isinstance(self._f_time, (list, tuple))
        assert isinstance(self._b_time, (list, tuple))
        assert isinstance(self._w_time, (list, tuple))

        self.estimated_time_cost = self.estimate_time_cost()
        self.M = self.set_M_value(self.estimated_time_cost)
        # 创建 Gurobi 模型
        self.model = Model("Simulator")

        # 变量初始化
        self._stage_f_time  = [[] for _ in range(self._num_pipelines)]
        self._stage_b_time = [[] for _ in range(self._num_pipelines)]
        self._stage_w_time = [[] for _ in range(self._num_pipelines)]

        self._stage_f_offsets = [[[] for _ in range(self._num_stages)] for _ in range(self._num_pipelines)]
        self._stage_b_offsets = [[[] for _ in range(self._num_stages)] for _ in range(self._num_pipelines)]
        self._stage_w_offsets = [[[] for _ in range(self._num_stages)] for _ in range(self._num_pipelines)]

        self._devices = [[] for _ in range(self._num_devices)]
        self._construct_device_stage_mappings()
        self.model_result = None

    def estimate_time_cost(self):
        res = (max(self._num_microbatches) + self._num_devices) * (sum(self._f_time) + sum(self._b_time) + sum(self._w_time)) / self._num_stages
        assert res > 0, "Time cost should be greater than 0"
        print_to_file(self._file_path, "Estimated time cost {}.\n".format(res))
        return res

    def set_M_value(self, estimated_cost):
        import math
        n = math.floor(math.log10(estimated_cost)) + 2 # 计算位数
        print("Set M to {}.\n".format(10 ** (n + 1)))
        return 10 ** (n + 1)  # 返回 10 的 (n + 1) 次方

    def show_device_stage_mapping(self):
        for did, ds in enumerate(self._devices):
            print_to_file(self._file_path, "Device {}: {}.\n".format(did, ds))

    def show_solution_detail(self):
        prefixes = ('f_', 'b_', 'w_')
        for key in self.model_result:
            if str(key).startswith(prefixes):
                print_to_file(self._file_path, "{},{}.\n".format(str(key), self.model_result[key]))
            if str(key) == "max_start_offset":
                print_to_file(self._file_path, "MinExeTime:{}.\n".format(self.model_result[key]))

    def _construct_device_stage_mappings(self):
        for pid in range(self._num_stages):
            self._devices[pid % self._num_devices].append(pid)
        
    def _reset_comm_length(self, dsa):
        new_comm_length = [[COMM_TIME if i != j else 0 for j in range(self._num_stages)] for i in range(self._num_stages)]
        for d in dsa:
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    new_comm_length[d[i]][d[j]] = 0
                    new_comm_length[d[j]][d[i]] = 0
        return new_comm_length

    def _build_constraints(self) -> None:
        for pid in range(self._num_pipelines):
            for sid in range(self._num_stages):
                for mid in range(self._num_microbatches[pid]):
                    if pid in self._fail_pipelines_stages and sid in self._fail_pipelines_stages[pid]:
                        d_pid = self._real_pid(pid=pid, sid=sid, mid=mid)
                        self._stage_f_offsets[pid][sid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"f_{mid + self._mid_offset[pid]}_{sid + self._num_stages * d_pid}", lb=0))
                        self._stage_b_offsets[pid][sid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"b_{mid + self._mid_offset[pid]}_{sid + self._num_stages * d_pid}", lb=0))
                        self._stage_w_offsets[pid][sid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"w_{mid + self._mid_offset[pid]}_{sid + self._num_stages * d_pid}", lb=0))
                    else:
                        self._stage_f_offsets[pid][sid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"f_{mid + self._mid_offset[pid]}_{sid + self._num_stages * pid}", lb=0))
                        self._stage_b_offsets[pid][sid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"b_{mid + self._mid_offset[pid]}_{sid + self._num_stages * pid}", lb=0))
                        self._stage_w_offsets[pid][sid].append(self.model.addVar(vtype=GRB.INTEGER, name=f"w_{mid + self._mid_offset[pid]}_{sid + self._num_stages * pid}", lb=0))
                
                self._stage_f_time[pid].append(int(self._f_time[sid] * self._comp_time_ratio[pid][sid]))
                self._stage_b_time[pid].append(int(self._b_time[sid] * self._comp_time_ratio[pid][sid]))
                self._stage_w_time[pid].append(int(self._w_time[sid] * self._comp_time_ratio[pid][sid]))
        
        self._comm_time = self._reset_comm_length(self._devices)
        self._construct_pipeline_data_dependencies()
        self._construct_recycle_constraints()
        # self._serial_computation_within_device_constraint()
        # self._add_memory_constraints()

    def freeze_schedule_by_mid(self, mid):
        if BASE_SOLUTION:
            self.model.update()
            for var in self.model.getVars():
                if not var.VarName.startswith(("f_","b_","w_")):
                    continue
                _mid = int(var.VarName.split("_")[1])
                if _mid == mid:
                    self.model.addConstr(var == self.pipeline_scheduler.results[var.VarName])

    def _real_pid(self, pid, sid, mid):
        if pid in self._fail_pipelines_stages and sid in self._fail_pipelines_stages[pid]:
            health_pipeline_idxs = [i for i in range(self._num_pipelines) if i not in self._fail_pipelines_stages or ( i in self._fail_pipelines_stages and sid not in self._fail_pipelines_stages[i])]
            nhpp = len(health_pipeline_idxs)
            r_pid = health_pipeline_idxs[mid % nhpp]
            return r_pid
        return pid

    def _construct_pipeline_data_dependencies(self):
        for pid in range(self._num_pipelines):
            for mid in range(self._num_microbatches[pid]):
                for sid in range(1, self._num_stages):
                    self.model.addConstr(self._stage_f_offsets[pid][sid][mid] >= self._stage_f_offsets[pid][sid - 1][mid] +
                                        self._stage_f_time[self._real_pid(pid, sid - 1, mid)][sid - 1] + self._comm_time[sid - 1][sid], name=f"f_{pid}_{sid}_{mid} > f_{pid}_{sid - 1}_{mid}")

                for sid in range(self._num_stages - 1, 0, -1):
                    self.model.addConstr(self._stage_b_offsets[pid][sid - 1][mid] >= self._stage_b_offsets[pid][sid][mid] +
                                        self._stage_b_time[self._real_pid(pid, sid, mid)][sid] + self._comm_time[sid][sid - 1], name=f"b_{pid}_{sid - 1}_{mid} > b_{pid}_{sid}_{mid}")

                self.model.addConstr(self._stage_b_offsets[pid][self._num_stages - 1][mid] >= self._stage_f_offsets[pid][self._num_stages - 1][mid] +
                                    self._stage_f_time[self._real_pid(pid, self._num_stages - 1, mid)][self._num_stages - 1], name=f"b_{pid}_{self._num_stages - 1}_{mid} > f_{pid}_{self._num_stages - 1}_{mid}")

                if SPLIT_BACKPROP:
                    for sid in range(self._num_stages):
                        if pid in self._fail_pipelines_stages and sid in self._fail_pipelines_stages[pid]:
                            self.model.addConstr(self._stage_w_offsets[pid][sid][mid] >= self._stage_b_offsets[pid][sid][mid] +
                                            self._stage_b_time[self._real_pid(pid, sid, mid)][sid], name=f"w_{sid} > b_{sid}")
                        else:
                            self.model.addConstr(self._stage_w_offsets[pid][sid][mid] == self._stage_b_offsets[pid][sid][mid] +
                                            self._stage_b_time[self._real_pid(pid, sid, mid)][sid], name=f"w_{sid} = b_{sid}")

                if mid > 0:
                    for sid in range(self._num_stages):
                        self.model.addConstr(self._stage_f_offsets[pid][sid][mid] >= self._stage_f_offsets[pid][sid][mid - 1] +
                                            self._stage_f_time[self._real_pid(pid, sid, mid - 1)][sid], name=f"f_mid{mid} > f_mid{mid-1}")
                        self.model.addConstr(self._stage_b_offsets[pid][sid][mid] >= self._stage_b_offsets[pid][sid][mid - 1] +
                                            self._stage_b_time[self._real_pid(pid, sid, mid - 1)][sid], name=f"b_mid{mid} > b_mid{mid-1}")
                        if SPLIT_BACKPROP:
                            self.model.addConstr(self._stage_w_offsets[pid][sid][mid] >= self._stage_w_offsets[pid][sid][mid - 1] +
                                                self._stage_w_time[self._real_pid(pid, sid, mid - 1)][sid], name=f"w_mid{mid} > w_mid{mid-1}")
            if pid not in self._fail_pipelines_stages or (0 not in self._fail_pipelines_stages[pid]):
                self.model.addConstr(self._stage_f_offsets[pid][0][0] == 0, name=f"f_start_at_0")

    def _construct_constraints_of_schedule(self, pipeline_schedules):
        type2offset = {
            WorkloadType.F: self._stage_f_offsets,
            WorkloadType.B : self._stage_b_offsets, 
            WorkloadType.W : self._stage_w_offsets,
        }
        type2time = {
            WorkloadType.F: self._stage_f_time,
            WorkloadType.B : self._stage_b_time, 
            WorkloadType.W : self._stage_w_time,
        }
        for pid in range(self._num_pipelines):
            pipeline_schedule = pipeline_schedules[pid]
            for stage_schedule in pipeline_schedule:
                for idx in range(len(stage_schedule) - 1):
                    wtype, mid, sid = stage_schedule[idx]
                    nwtype, nmid, nsid = stage_schedule[idx + 1]
                    # if pid in self._fail_pipelines_stages and (sid in self._fail_pipelines_stages[pid] or nsid in self._fail_pipelines_stages[pid]):
                    #     continue
                    self.model.addConstr(type2offset[nwtype][pid][nsid][nmid] >= type2offset[wtype][pid][sid][mid] + type2time[wtype][self._real_pid(pid, sid, mid)][sid], name=f"1f1b constr")

    def _construct_recycle_constraints(self):
        #NOTE Maintain 1F1B schedule on healthy devices and ILP solved schedule on failed devices
        wtype_order = [WorkloadType.B, WorkloadType.W, WorkloadType.F]
        wtype_num = len(wtype_order)
        workload_idx_in_mids = {WorkloadType.F: 0, WorkloadType.B : 1, WorkloadType.W : 2}
        schedules = [[[] for _ in range(self._num_stages)] for _ in range(self._num_pipelines)]
        mid_offset = 0
        for pid in range(self._num_pipelines):
            for sid in range(self._num_stages):
                mids = [0 for _ in range(wtype_num)]
                # warmup
                while mids[0] < min(self._num_stages - sid, self._num_microbatches[pid]):
                    schedules[pid][sid].append((WorkloadType.F, mids[0] + mid_offset, sid))
                    mids[0] += 1
                # 1F1B
                iter = 0
                finish_flag = [0 for _ in range(wtype_num)]
                while sum(finish_flag) < wtype_num:
                    next_workload_type = wtype_order[iter % wtype_num]
                    next_mid = mids[workload_idx_in_mids[next_workload_type]]
                    if next_mid < self._num_microbatches[pid]:
                        schedules[pid][sid].append((next_workload_type, next_mid + mid_offset, sid))
                        mids[workload_idx_in_mids[next_workload_type]] += 1
                    else:
                        finish_flag[workload_idx_in_mids[next_workload_type]] = 1
                    iter+=1
        self._construct_constraints_of_schedule(pipeline_schedules=schedules)
        self._construct_across_device_serial_evenly_distributed_constraints()

    def _get_stage_fbw_offsets(self, pid, sid):
        f = self._stage_f_offsets[pid][sid]
        b = self._stage_b_offsets[pid][sid]
        w = self._stage_w_offsets[pid][sid]
        stage_fbw_offsets = [x for triple in zip(f, b, w) for x in triple]
        return stage_fbw_offsets

    def _construct_across_device_serial_evenly_distributed_constraints(self):
        M = self.M
        for pid in self._fail_pipelines_stages:
            for sid in self._fail_pipelines_stages[pid]:
                failed_stage_fbw_offsets = self._get_stage_fbw_offsets(pid=pid, sid=sid)
                for idx, fbw in enumerate(failed_stage_fbw_offsets):
                    fbw_mid = idx // 3
                    fbw_type = idx % 3
                    stage_fbw_time = [self._stage_f_time, self._stage_b_time, self._stage_w_time]
                    stage_time = stage_fbw_time[fbw_type]
                    d_pid = self._real_pid(pid, sid, fbw_mid)
                    d_sid = sid

                    for d_mid, d_fbw in enumerate(self._stage_f_offsets[d_pid][d_sid]):
                        y = self.model.addVar(vtype=GRB.BINARY)
                        self.model.addConstr(fbw >= d_fbw + self._stage_f_time[d_pid][d_sid] - (1 - y) * M) 
                        self.model.addConstr(fbw + stage_time[d_pid][d_sid] <= d_fbw + y * M)

                    for d_mid, d_fbw in enumerate(self._stage_b_offsets[d_pid][d_sid]):
                        y = self.model.addVar(vtype=GRB.BINARY)
                        self.model.addConstr(fbw >= d_fbw + self._stage_b_time[d_pid][d_sid] - (1 - y) * M) 
                        self.model.addConstr(fbw + stage_time[d_pid][d_sid] <= d_fbw + y * M)

                    for d_mid, d_fbw in enumerate(self._stage_w_offsets[d_pid][d_sid]):
                        y = self.model.addVar(vtype=GRB.BINARY)
                        self.model.addConstr(fbw >= d_fbw + self._stage_w_time[d_pid][d_sid] - (1 - y) * M) 
                        self.model.addConstr(fbw + stage_time[d_pid][d_sid] <= d_fbw + y * M)

                    for f_pid in self._fail_pipelines_stages:
                        if f_pid != pid:
                            for f_sid in self._fail_pipelines_stages[f_pid]:
                                if f_sid == sid:
                                    for d_mid, d_fbw in enumerate(self._stage_f_offsets[f_pid][f_sid]):
                                        y = self.model.addVar(vtype=GRB.BINARY)
                                        self.model.addConstr(fbw >= d_fbw + self._stage_f_time[d_pid][d_sid] - (1 - y) * M) 
                                        self.model.addConstr(fbw + stage_time[d_pid][d_sid] <= d_fbw + y * M)

                                    for d_mid, d_fbw in enumerate(self._stage_b_offsets[f_pid][f_sid]):
                                        y = self.model.addVar(vtype=GRB.BINARY)
                                        self.model.addConstr(fbw >= d_fbw + self._stage_b_time[d_pid][d_sid] - (1 - y) * M) 
                                        self.model.addConstr(fbw + stage_time[d_pid][d_sid] <= d_fbw + y * M)

                                    for d_mid, d_fbw in enumerate(self._stage_w_offsets[f_pid][f_sid]):
                                        y = self.model.addVar(vtype=GRB.BINARY)
                                        self.model.addConstr(fbw >= d_fbw + self._stage_w_time[d_pid][d_sid] - (1 - y) * M) 
                                        self.model.addConstr(fbw + stage_time[d_pid][d_sid] <= d_fbw + y * M)
                

    
    def _construct_serial_constraints(self):
        M = self.M
        for pid in range(self._num_pipelines):
            for sid in range(self._num_stages):
                if pid not in self._fail_pipelines_stages:
                    continue
                if sid not in self._fail_pipelines_stages[pid]:
                    continue
                for f_mid, f in enumerate(self._stage_f_offsets[pid][sid]):
                    for b_mid, b in enumerate(self._stage_b_offsets[pid][sid]):
                        y = self.model.addVar(vtype=GRB.BINARY, name=f"Dofb{sid}_{f_mid}_{b_mid}")
                        self.model.addConstr(f >= b + self._stage_b_time[pid][sid] - (1 - y) * M, name=f"Dofb{sid}_{f_mid}_{b_mid}1") 
                        self.model.addConstr(f + self._stage_f_time[pid][sid] <= b + y * M, name=f"Dofb{sid}_{f_mid}_{b_mid}2")
                    for w_mid, w in enumerate(self._stage_w_offsets[pid][sid]):
                        y = self.model.addVar(vtype=GRB.BINARY, name=f"Dofw{sid}_{f_mid}_{w_mid}")
                        self.model.addConstr(f >= w + self._stage_w_time[pid][sid] - (1 - y) * M, name=f"Dofw{sid}_{f_mid}_{w_mid}1") 
                        self.model.addConstr(f + self._stage_f_time[pid][sid] <= w + y * M, name=f"Dofw{sid}_{f_mid}_{w_mid}2")

                for f_mid, f in enumerate(self._stage_b_offsets[pid][sid]):
                    for b_mid, b in enumerate(self._stage_f_offsets[pid][sid]):
                        y = self.model.addVar(vtype=GRB.BINARY, name=f"Dobf{sid}_{f_mid}_{b_mid}")
                        self.model.addConstr(f >= b + self._stage_b_time[pid][sid] - (1 - y) * M, name=f"Dobf{sid}_{f_mid}_{b_mid}1") 
                        self.model.addConstr(f + self._stage_f_time[pid][sid] <= b + y * M, name=f"Dobf{sid}_{f_mid}_{b_mid}2")
                    for w_mid, w in enumerate(self._stage_w_offsets[pid][sid]):
                        y = self.model.addVar(vtype=GRB.BINARY, name=f"Dobw{sid}_{f_mid}_{w_mid}")
                        self.model.addConstr(f >= w + self._stage_w_time[pid][sid] - (1 - y) * M, name=f"Dobw{sid}_{f_mid}_{w_mid}1") 
                        self.model.addConstr(f + self._stage_f_time[pid][sid] <= w + y * M, name=f"Dobw{sid}_{f_mid}_{w_mid}2")

                for f_mid, f in enumerate(self._stage_w_offsets[pid][sid]):
                    for b_mid, b in enumerate(self._stage_f_offsets[pid][sid]):
                        y = self.model.addVar(vtype=GRB.BINARY, name=f"Dowf{sid}_{f_mid}_{b_mid}")
                        self.model.addConstr(f >= b + self._stage_b_time[pid][sid] - (1 - y) * M, name=f"Dowf{sid}_{f_mid}_{b_mid}1") 
                        self.model.addConstr(f + self._stage_f_time[pid][sid] <= b + y * M, name=f"Dowf{sid}_{f_mid}_{b_mid}2")
                    for w_mid, w in enumerate(self._stage_b_offsets[pid][sid]):
                        y = self.model.addVar(vtype=GRB.BINARY, name=f"Dowb{sid}_{f_mid}_{w_mid}")
                        self.model.addConstr(f >= w + self._stage_w_time[pid][sid] - (1 - y) * M, name=f"Dowb{sid}_{f_mid}_{w_mid}1") 
                        self.model.addConstr(f + self._stage_f_time[pid][sid] <= w + y * M, name=f"Dowb{sid}_{f_mid}_{w_mid}2")

    def _build_optimize_objectives(self) -> None:
        max_var = self.model.addVar(vtype=GRB.INTEGER, name="max_start_offset")
        for pid in range(self._num_pipelines):
            for sid in range(self._num_stages):
                self.model.addConstr(max_var >= self._stage_w_offsets[pid][sid][-1] + self._stage_w_time[pid][sid])
        self.model.setObjective(max_var, GRB.MINIMIZE)

    def run(self, draw=False) -> None:
        """run simulation"""
        self._build_constraints()        
        self._build_optimize_objectives()

        self.model.setParam('TimeLimit', self._time_limit)
        # self.model.setParam('MIPGap', 0.00)

        # for mid in range(MICRO_BATCH_NUM):
        #     self.freeze_schedule_by_mid(mid=mid)

        start_time = time.time()
        self.model.optimize()
        end_time = time.time()
        if self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")

            # 计算不一致子集
            self.model.computeIIS()

            # 输出导致不可行的约束
            print("The following constraints are infeasible:")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"{c.ConstrName}")
                    print(f"{c.IISConstr}")
                    input("Press any key to continue...")

            # 可选：输出变量
            # print("The following variables are part of the IIS:")
            # for v in self.model.getVars():
            #     if v.IISVar:
            #         print(f"{v.VarName}")

        if self.model.status == GRB.OPTIMAL:
            print_to_file(self._file_path, f"Optimal Result Cost: {end_time - start_time:.2f}.\n")
            # tranforms the result to a dictionary.
        elif self.model.status != GRB.INFEASIBLE:
            print_to_file(self._file_path, f"Result Found Cost: {end_time - start_time:.2f}.\n")
        else:
            print_to_file(self._file_path, "No Solution Found.\n")
            return {"max_start_offset": 999999999999}
        
        results = {var.varName: var.x for var in self.model.getVars()}
        self.model_result = results
        print_to_file(self._file_path, "MinExeTime:{}.\n".format(results["max_start_offset"]))
        if draw:
            # 4. draws the result.
            results = {duplicate_last_part(str(key)) : self.model_result[key] for key in self.model_result if str(key)[0:2] in ["f_","b_","w_"]}
            self._draw(results)
        return results

    def _draw(self, results: dict) -> None:
        painter_conf = {
            "pipeline_num": self._num_pipelines,
            "device_num": self._num_devices,
            "devices": self._devices * self._num_pipelines,
            "stage_num": self._num_stages,
            "pp_height": PP_HEIGHT,
            "pp_align": PP_ALIGN,
            "pixel_base": PIXEL_BASE,
            "nmb": self._num_microbatches * self._num_pipelines,
            "f_times": self._stage_f_time,
            "b_times": self._stage_b_time,
            "w_times": self._stage_w_time,
            "comm_length": self._comm_time,
            "file_path": self._file_path,
            "max_time": self.model_result['max_start_offset'],
        }
        print(results, len(results))
        SchedulingPainter(painter_conf).draw(results)