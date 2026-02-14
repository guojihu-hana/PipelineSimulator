from .Stage import *
from ..utils import save_to_file
# TODO rethinking the head memory cost
def get_required_memory(stage_id, layer_num, wtype, recomp, stage_num, bwd_split):
    if wtype ==WorkloadType.F:
        required_memory = Activation.FULL * layer_num
        if recomp:
            required_memory = layer_num * (Activation.FULL * (1 - recomp) + Activation.INPUT * recomp)
        if stage_id == stage_num - 1:
            required_memory += Activation.LOSS
    elif wtype == WorkloadType.R:
        required_memory = layer_num * (Activation.FULL - Activation.INPUT)
    else:
        if bwd_split:
            if wtype == WorkloadType.B:
                required_memory = (Gradient.INPUT - Activation.FULL * ACT_B_RATIO) * layer_num
            elif wtype == WorkloadType.W:
                required_memory = Gradient.PARAMETER * layer_num
        else:
            if wtype == WorkloadType.B:
                required_memory = (Gradient.INPUT + Gradient.PARAMETER - Activation.FULL) * layer_num
    return required_memory

class Device:
    
    BUSY = 1
    IDLE = 2

    def __init__(self, 
                schedule_method, 
                bwd_split, 
                did: int, 
                nmb:int, 
                mid_offset:int, 
                mbs:int, 
                bs:int, 
                layer_num: int, 
                device_num: int, 
                chunk_num: int, 
                stage_num: int,
                execute_strategy: ExecuteStrategy,
                static_schedule: list = None, 
                max_mem: int = 80, 
                comp_power: float = 1, 
                pipeline = None):
        
        self.schedule_method = schedule_method
        self.bwd_split = bwd_split
        self.did = did
        self.execute_strategy = execute_strategy
        self.layer_num = layer_num
        self.device_num = device_num
        self.chunk_num = chunk_num
        self.stage_num = stage_num
        self.warmup_end_flag = False
        self.warmup_diff = 1 if self.did != DEVICE_NUM - 1 else 0
        self.begin_warmup_num = (self.chunk_num - 1) * self.device_num + 1 + self.device_num - 1 - self.did + self.warmup_diff
        self.begin_warmup_num = (self.chunk_num - 1) * self.device_num + 1 + (self.device_num - 1 - self.did) * 2 + 1 + self.warmup_diff
        self.begin_warmup_num = (4 - self.did)
        # self.begin_warmup_num = (self.chunk_num - 1) * self.device_num + 1 + (self.device_num - 1 - self.did) * 2
        self.idle_time = 0
        self.steady_start_flag = False
        self.stages: dict[int, Stage] = {}  # 存放各阶段的字典
        self.state: int = Device.IDLE
        self.current_workload: Workload = None
        self.current_mem_usage: int = 0
        self.peak_memory_usage: int = 0
        self.nmb: int = nmb
        self.mid_offset: int = mid_offset
        self.mbs : int = mbs
        self.bs : int = bs
        self.held_mids: set = set(range(self.mid_offset, self.mid_offset + self.nmb))
        self.mem_usage_record: dict[int, int] = {}
        self.peak_mem_usage_record: dict[int, int] = {}
        self.static_schedule: list[str] = static_schedule
        self.next_workload_idx: int = 0
        self.workload_type_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
        self.last_wtype = None
        self.total_layers = 0
        self.mid_traverse_order:list[int] = list(range(0,self.nmb))
        self.warmup_num_f = 0
        self.warmup_num_b = 0
        self.warmup_num_w = 0
        self.exe_num_f = 0
        self.exe_num_b = 0
        self.exe_num_w = 0
        self.exe_num_r = 0
        self.overlap_flag = False
        self.order_balance = True
        self.next_workload_type = None
        # self.executable_workloads:list[Workload] = []
        self.executable_workloads:OrderedQueue = OrderedQueue(self.workload_type_priority_order)
        self.pipeline = pipeline

        self.next_mid = 0
        self.released_workloads = []
        self.available_f_num = 0
        self.executing_mid_idx = 0
        self.finish_time = -1

        self.memory_monitor = None
        self.wait_for_schedule = 0

        self.situations = 1
        self.max_memory = max_mem
        self.comp_power = comp_power

        self.workload_execute_record = self.pipeline.workload_execute_record

    def get_max_mem_did(self):
        # 找到显存最大的设备
        max_device = max(
            self.pipeline.devices,
            key=lambda d: d.current_mem_usage
        )
        return max_device.did


    def get_required_memory_by_workload(self, workload:Workload):
        # wrapper of get required memory
        return self.stages[workload.sid].update_memory_usage(workload=workload, sim=True)
    
    def init_memory_monitor(self):
        pass

    def is_bottleneck_device(self):
        return self.stage_num - 1 in list(self.stages.keys())

    def has_direct_dependency(self, time, workload:Workload):
        if len(self.workload_execute_record[self.did]) == 0:
            return False
        if self.workload_execute_record[self.did][-1].end_time < time: # Already idle
            return False
        for did, executed_workloads in enumerate(self.workload_execute_record):
            if did == self.did or len(executed_workloads) == 0:
                continue
            pivot_workload = executed_workloads[-1]
            if pivot_workload.mid == workload.mid: # Only micro-batches with same the mid have dependency
                if pivot_workload.sid == workload.sid - 1 and pivot_workload.wtype == workload.wtype == WorkloadType.F:
                    if pivot_workload.end_time <= self.workload_execute_record[self.did][-1].start_time:
                        continue
                    return True
                if pivot_workload.sid == workload.sid + 1 and pivot_workload.wtype == workload.wtype == WorkloadType.B:
                    if pivot_workload.end_time <= self.workload_execute_record[self.did][-1].start_time:
                        continue
                    return True
        return False

    def get_initial_executable_workload(self, time)->list[Workload]:
        executable_workoads = []
        workload_type_order = [WorkloadType.B,WorkloadType.F,WorkloadType.W]
        if self.execute_strategy.switch_workload_type:
            if self.last_wtype == WorkloadType.B:
                workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]
            if self.last_wtype == WorkloadType.F:
                workload_type_order = [WorkloadType.B,WorkloadType.F,WorkloadType.W]

        if self.execute_strategy.save_memory and self.get_max_mem_did() == self.did and self.current_mem_usage > self.peak_memory_usage / 5 * 4:
            if self.last_wtype == WorkloadType.B:
                workload_type_order = [WorkloadType.W,WorkloadType.F,WorkloadType.B]
            if self.last_wtype == WorkloadType.W:
                workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]
        if self.execute_strategy.save_memory:
            head_ce_workloads = []
            for wtype in [WorkloadType.W]:
                for mid in range(self.mid_offset, self.mid_offset + self.nmb):
                    for sid in self.stages:
                            workloads = self.stages[sid].workloads
                            if mid not in workloads: continue
                            if wtype in workloads[mid] and workloads[mid][wtype].is_executable(time=time):
                                head_ce_workloads.append(workloads[mid][wtype])                          
            # ensure head to be executed as quickly as possible
            executable_workoads += head_ce_workloads

        delayed_workload = []
        canceled_workload = []
        for wtype in workload_type_order:
            for mid in range(self.bs // self.mbs):
                for sid in self.stages:
                    workloads = self.stages[sid].workloads
                    if mid in workloads and wtype in workloads[mid] and workloads[mid][wtype].is_executable(time=time):
                        workload = workloads[mid][wtype]
                        if self.execute_strategy.overlap_aware and self.has_direct_dependency(time=time, workload=workload):
                            if self.is_bottleneck_device():
                                delayed_workload.append(workloads[mid][wtype])
                            else:
                                canceled_workload.append(workloads[mid][wtype])
                        else:
                            executable_workoads.append(workloads[mid][wtype])
        executable_workoads += delayed_workload
        if len(executable_workoads) == 0:
            executable_workoads += canceled_workload

        return executable_workoads
    
    def get_executable_overlap_aware_workload(self, time)->list[Workload]:
        if len(self.executable_workloads) == 0:
            return []
        
        workload_type_order = [WorkloadType.B,WorkloadType.F,WorkloadType.W]
        if self.execute_strategy.switch_workload_type:
            if self.last_wtype == WorkloadType.B:
                workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]
            if self.last_wtype == WorkloadType.F:
                workload_type_order = [WorkloadType.B,WorkloadType.F,WorkloadType.W]

        if self.execute_strategy.save_memory:
            if self.last_wtype == WorkloadType.B:
                workload_type_order = [WorkloadType.W,WorkloadType.F,WorkloadType.B]
            if self.last_wtype == WorkloadType.W:
                workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]
        
        self.executable_workloads.set_type_order(workload_type_order)
        workload = self.executable_workloads.pop()
        delayed_workloads = []        
        while len(self.executable_workloads) and self.execute_strategy.overlap_aware and self.has_direct_dependency(time=time, workload=workload):
            delayed_workloads.append(workload)
            workload = self.executable_workloads.pop()
        
        for delayed_workload in delayed_workloads:
            self.executable_workloads.push(delayed_workload)
        
        if not workload:
            return [self.executable_workloads.pop()]
        return [workload]

    def show_stages(self, detail_info=False):
        for sid in self.stages:
            # print("Stage {} recomp={} on Device {}".format(sid, self.stages[sid].recomp, self.device_id))
            if detail_info:
                for mid in self.stages[sid].workloads:
                    if mid == 10:
                        for wlt in self.stages[sid].workloads[mid]:
                            print(self.stages[sid].workloads[mid][wlt])

    def add_stage(self, stage_id: int, layer_num:int, layer_idx_start: int,
                  recomp:bool = False, 
                  layerwise:bool = False,
                  basic_memory = 0) -> None:
        stage_type = StageType.LAYERS
        para_num = 0
        if layerwise:
            assert layer_num == 1 and self.chunk_num == self.layer_num // self.device_num, f"LAYERWISE require 1 layer per stage (CHUNK_NUM == LAYER_NUM // PP_SIZE) but got {layer_num} per stage"
            if stage_id == 0:
                stage_type = StageType.EMBD
                basic_memory = StateMemory.EMB
                para_num = Parameter.EMB
            elif stage_id == self.layer_num + 1:
                stage_type = StageType.HEAD
                basic_memory = StateMemory.HEAD
                para_num = Parameter.HEAD
            elif stage_id == self.layer_num + 2:
                stage_type = StageType.CE
            else:
                stage_type = StageType.LAYER
                basic_memory = StateMemory.LAYER
                para_num = Parameter.LAYER
        else:
            basic_memory = StateMemory.LAYER * layer_num
            para_num = Parameter.LAYER * layer_num
            if stage_id == 0:
                basic_memory += StateMemory.EMB
                para_num += Parameter.EMB
            elif stage_id == self.stage_num - 1 and not gpc["HEAD_DP"]:
                basic_memory += StateMemory.HEAD
                para_num += Parameter.HEAD
            elif stage_id == self.stage_num and gpc["HEAD_DP"]:
                para_num += Parameter.HEAD

        stage = Stage(
                bwd_split = self.bwd_split,
                schedule_method=self.schedule_method,
                device_id=self.did, 
                stage_id=stage_id,
                stage_num=self.stage_num,
                para_num=para_num,
                stage_type=stage_type,
                nmb=self.nmb,
                mid_offset=self.mid_offset,
                recomp=recomp,
                layerwise=layerwise,
                layer_num=layer_num,
                layer_idx_start=layer_idx_start,
                comp_power=self.comp_power,
            )
        self.stages[stage.sid] = stage
        self.total_layers+=layer_num

    def count_wtype_num(self, did : int, wtype : WorkloadType):
        count = 0
        for w in self.workload_execute_record[did]:
            count += 1 if w.wtype == wtype else 0
        return count

    def update_constraints_within_device(self, time, constraint: Workload):
        for sid in self.stages:
            if constraint.mid not in self.held_mids:
                continue
            updated_workload = self.stages[sid].update_constraints_within_stage(time, constraint=constraint)
            if updated_workload and updated_workload.is_executable(time):
                self.executable_workloads.push(updated_workload)
    
    def update_mid_traverse_order(self,mid=None):
        if type(self.mid_traverse_order) is not list:
            self.mid_traverse_order = list(self.mid_traverse_order)
        self.mid_traverse_order.sort()
        if mid:
            self.mid_traverse_order.remove(mid)
            self.mid_traverse_order.append(mid)
    
    def get_completed_workload_count_by_type(self, wtype:WorkloadType):
        workload_num = 0
        for workload in self.workload_execute_record[self.did]:
            if workload.wtype == wtype:
                workload_num += 1
        return workload_num

    def get_executable_workload_num_by_type(self, wtype:WorkloadType):
        workload_num = 0
        for workload in self.executable_workloads:
            if workload.wtype == wtype:
                workload_num += 1
        return workload_num

    def execute_workload(self, time, run_schedule=False) -> None:
        assert time >= 0, f"Time should be non-negative (but got {time})."
        if self.state == Device.IDLE:
            if self.schedule_method == Schedule.OctoPipe:
                workload_list = self.get_executable_overlap_aware_workload(time=time)

                for workload in workload_list:
                    mid = workload.mid
                    sid = workload.sid
                    wtype = workload.wtype
                    
                    if self.execute_strategy.constrain_warmup:
                        if self.exe_num_f < self.begin_warmup_num:
                            if wtype != WorkloadType.F:
                                self.executable_workloads.push(workload)
                                continue
                        else:
                            if not self.warmup_end_flag:
                                if wtype != WorkloadType.B:
                                    self.executable_workloads.push(workload)
                                    continue
                    
                    proc_workload = self.stages[sid].execute_workload(time, mid=mid,workload_type=wtype)

                    if proc_workload:
                        self.last_wtype = wtype
                        if wtype == WorkloadType.F:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_f += 1
                        elif wtype == WorkloadType.B:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_b += 1
                            self.warmup_end_flag = True
                        elif wtype == WorkloadType.W:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_w += 1
                        
                        self.current_workload = proc_workload
                        self.update_memory_usage()
                        self.state = Device.BUSY
                        return proc_workload
            elif self.schedule_method == Schedule.STANDARD_INTERLEAVED:
                if self.next_workload_idx == len(self.static_schedule):
                    return None
                (workload_type, workload_mid, workload_sid) = self.static_schedule[self.next_workload_idx]
                
                if self.did < self.device_num - 1:
                    if self.exe_num_f > (self.chunk_num - 1) * self.device_num + (self.device_num - self.did - 1) * 2 - 1 and self.exe_num_f < self.chunk_num * self.nmb:
                        if workload_type == WorkloadType.F and self.count_wtype_num(self.did, WorkloadType.B) < self.count_wtype_num(self.did + 1, WorkloadType.B) and self.workload_execute_record[self.did + 1][-1].wtype == WorkloadType.B:
                            proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)  
                        elif workload_type in (WorkloadType.B, WorkloadType.W, WorkloadType.R):
                            proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)  
                        else:
                            return None
                    else:
                        proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)
                else:
                    proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)
                
                if proc_workload:
                    self.current_workload = proc_workload
                    self.update_memory_usage()
                    self.state = Device.BUSY
                    self.last_wtype = workload_type
                    if workload_type == WorkloadType.F:
                        self.exe_num_f += 1
                    elif workload_type == WorkloadType.B:
                            self.exe_num_b += 1
                    elif workload_type == WorkloadType.W:
                            self.exe_num_w += 1
                    elif workload_type == WorkloadType.R:
                            self.exe_num_r += 1
                    else:
                        raise Exception("Error workload type.")
                    self.next_workload_idx += 1
                    return proc_workload
            elif run_schedule or self.schedule_method in (Schedule.STANDARD_AFAB, Schedule.STANDARD_ZBH):
                if self.next_workload_idx == len(self.static_schedule):
                    return None
                try:
                    (workload_type, workload_mid, workload_sid) = self.static_schedule[self.next_workload_idx][:3]
                except Exception as e:
                    print((workload_type, workload_mid, workload_sid))
                    input()
                proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)
                if proc_workload:
                    self.current_workload = proc_workload
                    self.update_memory_usage()
                    self.state = Device.BUSY
                    self.next_workload_idx += 1
                    return proc_workload
            elif run_schedule or self.schedule_method == Schedule.STANDARD_1F1B:
                if self.next_workload_idx == len(self.static_schedule):
                    return None
                try:
                    (wtype, mid, sid) = self.static_schedule[self.next_workload_idx][:3]
                except Exception as e:
                    print((wtype, mid, sid))
                    input()

                proc_workload = self.stages[sid].execute_workload(time=time, mid=mid, workload_type=wtype)
                if proc_workload:
                    if wtype == WorkloadType.F:
                        if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                            self.exe_num_f += 1
                    elif wtype == WorkloadType.B:
                        if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                            self.exe_num_b += 1
                    elif wtype == WorkloadType.W:
                        if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                            self.exe_num_w += 1
                    self.current_workload = proc_workload
                    self.update_memory_usage()
                    self.state = Device.BUSY
                    self.next_workload_idx += 1
                    return proc_workload
            elif self.schedule_method in (Schedule.STANDARD_ZBV, Schedule.ZBV):
                if self.last_wtype == WorkloadType.F:
                    workload_type = WorkloadType.B
                elif self.last_wtype == WorkloadType.B:
                    workload_type = WorkloadType.W
                elif self.last_wtype == WorkloadType.W:
                    workload_type = WorkloadType.F
                else:
                    workload_type = WorkloadType.F

                if self.warmup_num_f < self.device_num * 2:
                    workload_type = WorkloadType.F
                if self.warmup_num_f == self.nmb * 2:
                    workload_type = WorkloadType.B
                if self.warmup_num_b == self.nmb * 2:
                    workload_type = WorkloadType.W

                for mid in range(self.nmb):
                    for sid in self.stages:
                        required_memory = get_required_memory(
                            stage_id=sid, 
                            layer_num=self.layer_num//self.stage_num,
                            wtype=workload_type,
                            workload_type_num=3, 
                            layer_wise=True,
                            recomp=self.stages[sid].recomp,
                        )

                        workload_type = self._reset_workload_type(
                            workload_type=workload_type,
                            required_memory=required_memory,
                            current_mem_usage=self.current_mem_usage,
                            max_memory=self.max_memory,
                        )

                        proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
                        if proc_workload:
                            if workload_type == WorkloadType.F:
                                self.warmup_num_f += 1
                            elif workload_type == WorkloadType.B:
                                self.warmup_num_b += 1
                            elif workload_type == WorkloadType.W:
                                self.warmup_num_w += 1
                            self.last_wtype = workload_type
                            self.current_workload = proc_workload
                            self.update_memory_usage()
                            self.state = Device.BUSY
                            return proc_workload
                
                now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                for workload_type in now_workload_priority_order:
                    for mid in range(self.nmb):
                        for sid in self.stages:
                            proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
                            if proc_workload:
                                if workload_type == WorkloadType.F:
                                    self.warmup_num_f += 1
                                elif workload_type == WorkloadType.B:
                                    self.warmup_num_b += 1
                                elif workload_type == WorkloadType.W:
                                    self.warmup_num_w += 1
                                self.last_wtype = workload_type
                                self.current_workload = proc_workload
                                self.update_memory_usage()
                                self.state = Device.BUSY
                                return proc_workload             
            elif self.schedule_method in (Schedule.ReCycle,):
                workload_list = self.get_executable_overlap_aware_workload(time=time)

                for workload in workload_list:
                    mid = workload.mid
                    sid = workload.sid
                    wtype = workload.wtype
                    proc_workload = self.stages[sid].execute_workload(time, mid=mid,workload_type=wtype)

                    if proc_workload:
                        self.last_wtype = wtype
                        if wtype == WorkloadType.F:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_f += 1
                        elif wtype == WorkloadType.B:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_b += 1
                            self.warmup_end_flag = True
                        elif wtype == WorkloadType.W:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_w += 1
                        
                        self.current_workload = proc_workload
                        self.update_memory_usage()
                        self.state = Device.BUSY
                        return proc_workload
            else:
                print("Schedule Not Supported")
        return None

    def _reset_workload_type(self, workload_type, required_memory, current_mem_usage, max_memory):
        #TODO 不同个Wave情况下，处于Wave不同边上的memory预留的值应该不同
        if current_mem_usage + required_memory >= max_memory - Gradient.INPUT - Gradient.PARAMETER:
            workload_type = WorkloadType.B
            if not self.bwd_split:
                return workload_type
        if current_mem_usage + required_memory >= max_memory - Gradient.PARAMETER:
            workload_type = WorkloadType.W
        return workload_type

    def update_memory_usage(self) -> int:
        if self.current_workload.state == Workload.IN_PROGRESS and self.current_workload.wtype in (WorkloadType.F, WorkloadType.R, WorkloadType.B):
            self.stages[self.current_workload.sid].update_memory_usage(workload=self.current_workload)
        elif self.current_workload.state == Workload.COMPLETED and self.current_workload.wtype == WorkloadType.W:
            self.stages[self.current_workload.sid].update_memory_usage(workload=self.current_workload)
        self.peak_memory_usage = sum(stage.peak_memory_usage for stage in self.stages.values())    
        self.current_mem_usage = sum(stage.memory_usage for stage in self.stages.values())
        self.mem_usage_record[(self.current_workload.start_time,self.current_workload.end_time)] = self.current_mem_usage
        self.peak_mem_usage_record[(self.current_workload.start_time,self.current_workload.end_time)] = (self.peak_memory_usage, self.current_workload.wtype.name, self.current_workload.sid, self.current_workload.mid)
        for stage in self.stages.values(): # recover peak memory usage to current memory usage
            if stage.sid != self.current_workload.sid:
                stage.peak_memory_usage = stage.memory_usage

    def get_memory_usage(self) -> int:
        return self.current_mem_usage

    def __repr__(self) -> str:
        return f"DeviceClass(stages={self.stages.keys()}),state={self.state}"