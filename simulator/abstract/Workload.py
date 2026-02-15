from .mutils import *
from .context import global_context as gpc
from .context import flush_fbw_time

class Workload:
    # 定义状态常量
    not_started = 1
    in_progress = 2
    finished = 3

    def __init__(self, schedule_method, device_idx:int, microbatch_idx: int, stage_idx: int, bwd_split: bool, duration: int, total_stage_num:int, wtype: WorkloadType, recomp:bool, split_recomp:bool, layer_idxs:list, comp_power:float):
        self.schedule_method = schedule_method
        self.bwd_split = bwd_split
        self.did = device_idx
        self.mid: int = microbatch_idx  # 微批次编号
        self.sid: int = stage_idx              # 阶段编号
        self.duration: int = duration               # 任务所需时间
        self.start_time: float = None               # 初始开始时间为None
        self.end_time: float = None                 # 结束时间
        self.state: int = Workload.not_started      # 初始状态为未开始
        self.ready_time: int = -1
        self.total_stage_num: int = total_stage_num
        self.recomp:bool = recomp
        self.split_recomp:bool = split_recomp
        self.layer_idxs:list = layer_idxs
        self.comp_power = comp_power
        if self.mid == 0 and self.sid == 0:
            self.ready_time = 0
        self.wtype: WorkloadType = wtype  # 工作负载类型
        self.wtype_str = wtype.name.lower()
        self.constraints: set = set()               # {(i1, j1, C1), ...}表示Stage i1 上的Microbatch j1 的 C1 操作是前置约束
        self._generate_constraints()

    def _generate_constraints(self):
        if self.wtype == WorkloadType.F:
            if self.sid > 0:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did,
                        microbatch_id = self.mid,
                        stage_id = self.sid - 1,
                        workload_type = WorkloadType.F)
                )
        elif self.wtype == WorkloadType.R:
            self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did,
                        microbatch_id = self.mid,
                        stage_id = self.sid,
                        workload_type = WorkloadType.F)
                )
        elif self.wtype == WorkloadType.B:
            if self.sid + 1 < self.total_stage_num:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did+1,
                        stage_id = self.sid+1, 
                        microbatch_id= self.mid, 
                        workload_type = WorkloadType.B if not (self.schedule_method in (Schedule.STANDARD_1F1B, Schedule.STANDARD_INTERLEAVED) and self.bwd_split == True) else WorkloadType.W)
                )
            else:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did,
                        stage_id=self.total_stage_num - 1, 
                        microbatch_id=self.mid, 
                        workload_type = WorkloadType.F)
                )
            if self.recomp and self.split_recomp:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did,
                        stage_id=self.sid, 
                        microbatch_id=self.mid, 
                        workload_type = WorkloadType.R)
                )
        elif self.wtype == WorkloadType.W:
            self.constraints.add(
                WorkloadConstraint(
                    device_id = self.did,
                    stage_id=self.sid, 
                    microbatch_id=self.mid, 
                    workload_type=WorkloadType.B)
            )

    def _generate_communication(self, time, constraint: WorkloadConstraint):
        if constraint.did != self.did:
            self.ready_time = max(self.ready_time, time + gpc["COMM_TIME"])
        else:
            self.ready_time = max(self.ready_time, time)

    def update_constraints(self, time, constraint: WorkloadConstraint):
        origin_len = len(self.constraints)
        self.constraints.discard(constraint)
        now_len = len(self.constraints)

        if origin_len != now_len:
            self._generate_communication(time, constraint)

    def is_executable(self, time):
        return len(self.constraints) == 0 and self.ready_time <= time and self.state == Workload.not_started
    
    def execute(self, time) -> bool:
        if self.state == Workload.not_started:
            if self.is_executable(time=time):
                self.state = Workload.in_progress
                self.start_time = time
                self.end_time = self.start_time + self.duration
                return True
        return False

    def complete(self, time) -> None:
        """完成任务并更新状态"""
        if self.state == Workload.in_progress and self.end_time <= time:
            self.state = Workload.finished

    @property
    def is_w(self):
        return self.wtype == WorkloadType.W
    
    @property
    def is_b(self):
        return self.wtype == WorkloadType.B
    
    @property
    def is_f(self):
        return self.wtype == WorkloadType.F
    
    @property
    def is_r(self):
        return self.wtype == WorkloadType.R
    
    def __repr__(self):
        return (f"did={self.did}, mid={self.mid}, sid={self.sid}, wtype={self.wtype.name}, "
            f"duration={self.duration}, start_time={self.start_time}, end_time={self.end_time}, state={self.state}, "
            f"ready_time={self.ready_time}, total_stages={self.total_stage_num}, "
            f"constraints={self.constraints}\n")
