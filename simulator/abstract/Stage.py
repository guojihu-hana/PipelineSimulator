from .Workload import *
import copy

@dataclass
class StageType:
    EMBD = 1
    LAYER = 2
    HEAD = 3
    CE = 4
    LAYERS = 5

def calculate_total_time(wtype:WorkloadType, recomp, layer_idxs:list)->float:
    if layer_idxs == None:
        return 0
    total_time = 0
    type_time_map = {
        WorkloadType.F:"F_TIMES",
        WorkloadType.B:"B_TIMES",
        WorkloadType.W:"W_TIMES",
    }
    times = gpc[type_time_map[wtype]]
    for idx in layer_idxs:
        total_time += times[idx]

    if recomp and wtype == WorkloadType.B:
        for idx in layer_idxs:
            total_time += gpc["F_TIMES"][idx]
    return total_time

def get_workload_duration(mid:int, sid:int, layer_wise:bool, bwd_split: bool, layer_num:int, stage_num: int, wtype:WorkloadType, recomp, layer_idxs:list=None, comp_power:float = 1)->float:
    if layer_wise:
        assert f"LAYERWISE not enabled."
    
    duration = calculate_total_time(wtype=wtype, recomp=recomp, layer_idxs=layer_idxs)
    # print(f"Sid:{sid}, Duration:{duration}.")
    if wtype in (WorkloadType.F, WorkloadType.R):
        if sid == 0:
            duration += gpc["EMB_F_TIME"]
        elif sid == stage_num - 1 and not HEAD_DP:
            duration += gpc["HEAD_F_TIME"] + gpc["CE_F_TIME"]
        elif sid == stage_num and HEAD_DP:
            duration = gpc["HEAD_F_TIME"] + gpc["CE_F_TIME"]
    elif wtype == WorkloadType.B: # Consider recomputation
        recomp = 0 #DEBUG seperate R and B
        if sid == stage_num - 1 and not HEAD_DP:
            duration += gpc["HEAD_B_TIME"] + gpc["CE_B_TIME"] + (gpc["HEAD_F_TIME"] + gpc["CE_F_TIME"]) * recomp
            if not bwd_split:
                duration += gpc["HEAD_W_TIME"]
        elif sid == stage_num and HEAD_DP:
            duration = gpc["HEAD_B_TIME"] + gpc["CE_B_TIME"] + (gpc["HEAD_F_TIME"] + gpc["CE_F_TIME"]) * recomp
            if not bwd_split:
                duration += gpc["HEAD_W_TIME"]
        elif sid == 0:
            duration += gpc["EMB_B_TIME"]
            if not bwd_split:
                duration += gpc["EMB_W_TIME"]
    elif wtype == WorkloadType.W:
        if sid == stage_num - 1 and not HEAD_DP:
            duration += gpc["HEAD_W_TIME"]
        elif sid == stage_num and HEAD_DP:
            duration = gpc["HEAD_W_TIME"]
        elif sid == 0:
            duration += gpc["EMB_W_TIME"]
    else:
        raise ValueError(f"Wrong workload type: {wtype}.")
    # return int(duration / comp_power) * gpc["MICRO_BATCH_TIME"][mid]
    return int(duration / comp_power)

class Stage:
    
    INTERLEAVED = 1
    VSHAPE = 2
    WAVELIKE = 3

    def __init__(self,schedule_method, bwd_split: bool, device_id:int, stage_id: int, stage_num: int, para_num:int, stage_type: StageType, nmb:int, mid_offset:int, layer_num: int, layer_idx_start: int, layerwise:bool = False, recomp: bool = False, comp_power: float = 1):
        self.schedule_method = schedule_method
        self.bwd_split = bwd_split
        self.did: int = device_id
        self.sid: int = stage_id
        self.stage_num: int = stage_num
        self.nmb: int = nmb
        self.mid_offset: int = mid_offset
        self.para_num: int = para_num / gpc["G"]
        self.model_memory_usage = self.para_num * gpc["FP16"] / gpc["TP_SIZE"]
        self.grad_memory_usage = 0
        self.emb_memory_gradient_usage = 0
        self.opt_memory_usage = self.para_num * 3 * gpc["FP32"] / gpc["TP_SIZE"] / gpc["ZERO_SIZE"]
        self.memory_generated_by_mb : int = [0] * self.nmb
        self.memory_usage: int = self.model_memory_usage + self.grad_memory_usage + self.opt_memory_usage
        self.peak_memory_usage: int = self.model_memory_usage + self.grad_memory_usage + self.opt_memory_usage
        self.workloads: dict[int, dict[WorkloadType, Workload]] = {}  
        self.stage_type: StageType = stage_type
        self.recomp = recomp
        self.layerwise = layerwise
        self.layer_num = layer_num
        self.layer_idx_start = layer_idx_start
        self.layer_idxs = list(range(layer_idx_start, layer_idx_start + layer_num))
        self.comp_power = comp_power
        if layerwise: 
            assert layer_num == 1, f"LAYERWISE require 1 layer per stage but got {layer_num}"
        self._add_workload()
        
    def _add_workload(self) -> None:
        total_stages = gpc["LAYER_NUM"]+3 if self.layerwise else self.stage_num
        if gpc["HEAD_DP"]:
            total_stages = self.stage_num + 1
        for mid in range(self.mid_offset, self.mid_offset + self.nmb):
            if gpc["HEAD_DP"]:
                if self.sid == self.stage_num and self.did == 0:
                    # pass
                    continue
                if self.sid == self.stage_num:
                    if mid == 0:
                        if self.did != gpc["PP_SIZE"] - 1:
                            continue
                    if mid == 8:
                        if self.did != gpc["PP_SIZE"] - 3:
                            continue
                    if mid == 9:
                        if self.did != gpc["PP_SIZE"] - 2:
                            continue
            self.workloads[mid] = {}
            fpw = Workload(
                schedule_method = self.schedule_method,
                bwd_split=self.bwd_split,
                device_id=self.did,
                stage_id=self.sid,
                microbatch_id=mid,
                wtype=WorkloadType.F,
                duration=get_workload_duration(
                    bwd_split=self.bwd_split,
                    mid=mid,
                    sid=self.sid,
                    layer_wise=self.layerwise,
                    layer_num=self.layer_num,
                    stage_num=self.stage_num,
                    wtype=WorkloadType.F,
                    recomp=self.recomp,
                    layer_idxs=self.layer_idxs,
                    comp_power=self.comp_power,
                ),
                recomp=self.recomp,
                total_stages=total_stages,
                layer_idxs=self.layer_idxs,
                comp_power=self.comp_power,
            )
            self.workloads[mid][WorkloadType.F] = fpw
            if self.sid == 0 and self.layerwise: # Embedding layer only has F
                continue

            igw = Workload(
                schedule_method = self.schedule_method,
                bwd_split=self.bwd_split,
                device_id=self.did,
                stage_id=self.sid,
                microbatch_id=mid,
                wtype=WorkloadType.B,
                duration=get_workload_duration(
                    bwd_split=self.bwd_split,
                    mid=mid,
                    sid=self.sid,
                    layer_wise=self.layerwise,
                    layer_num=self.layer_num,
                    stage_num=self.stage_num,
                    wtype=WorkloadType.B,
                    recomp=self.recomp,
                    layer_idxs=self.layer_idxs,
                    comp_power=self.comp_power,
                ), 
                recomp=self.recomp,
                total_stages=total_stages,
                layer_idxs=self.layer_idxs,
                comp_power=self.comp_power,
            )
            self.workloads[mid][WorkloadType.B] = igw
            if self.recomp:
                rfw = Workload(
                    schedule_method = self.schedule_method,
                    bwd_split=self.bwd_split,
                    device_id=self.did,
                    stage_id=self.sid,
                    microbatch_id=mid,
                    wtype=WorkloadType.R,
                    duration=get_workload_duration(
                    bwd_split=self.bwd_split,
                        mid=mid,
                        sid=self.sid,
                        layer_wise=self.layerwise,
                        layer_num=self.layer_num,
                        stage_num=self.stage_num,
                        wtype=WorkloadType.R,
                        recomp=self.recomp,
                        layer_idxs=self.layer_idxs,
                        comp_power=self.comp_power,
                    ), 
                    recomp=self.recomp,
                    total_stages=total_stages,
                    layer_idxs=self.layer_idxs,
                    comp_power=self.comp_power,
                )
                self.workloads[mid][WorkloadType.R] = rfw

            if self.bwd_split:
                pgw = Workload(
                    schedule_method = self.schedule_method,
                    bwd_split=self.bwd_split,
                    device_id=self.did,
                    stage_id=self.sid,
                    microbatch_id=mid,
                    wtype=WorkloadType.W,
                    duration=get_workload_duration(
                    bwd_split=self.bwd_split,
                        mid=mid,
                        sid=self.sid,
                        layer_wise=self.layerwise,
                        layer_num=self.layer_num,
                        stage_num=self.stage_num,
                        wtype=WorkloadType.W,
                        recomp=self.recomp,
                        layer_idxs=self.layer_idxs,
                        comp_power=self.comp_power,
                    ),
                    recomp=self.recomp,
                    total_stages=total_stages,
                    layer_idxs=self.layer_idxs,
                    comp_power=self.comp_power,
                )
                if self.stage_type == StageType.CE: # Cross-entropy layer has no W for parameter training
                    continue
                self.workloads[mid][WorkloadType.W] = pgw

    def update_constraints_within_stage(self, time, constraint: Workload) -> Workload:
        c_did = constraint.did
        c_sid = constraint.sid
        c_mid = constraint.mid
        c_wlt = constraint.wtype
        cstr = WorkloadConstraint(
            device_id=c_did,
            stage_id=c_sid, 
            microbatch_id=c_mid, 
            workload_type=c_wlt
        )
        if c_wlt == WorkloadType.F:
            if self.sid == c_sid + 1:
                self.workloads[c_mid][c_wlt].update_constraints(time, cstr)
                return self.workloads[c_mid][c_wlt]
            elif self.sid == c_sid and self.sid == constraint.total_stages - 1:
                self.workloads[c_mid][WorkloadType.B].update_constraints(time, cstr)
                return self.workloads[c_mid][WorkloadType.B]
        elif c_wlt == WorkloadType.B:
            if self.sid == c_sid - 1:
                self.workloads[c_mid][c_wlt].update_constraints(time, cstr)
                return self.workloads[c_mid][c_wlt]
            elif self.sid == c_sid and self.bwd_split:
                self.workloads[c_mid][WorkloadType.W].update_constraints(time, cstr)
                return self.workloads[c_mid][WorkloadType.W]
        elif c_wlt == WorkloadType.R:
            if self.sid == c_sid:
                self.workloads[c_mid][WorkloadType.B].update_constraints(time, cstr)
                return self.workloads[c_mid][WorkloadType.B]
        elif c_wlt == WorkloadType.W:
            if self.sid == c_sid - 1:
                self.workloads[c_mid][WorkloadType.B].update_constraints(time, cstr)
                return self.workloads[c_mid][WorkloadType.B]
        else:
            return None

        # for mid in self.workloads:
        #     for wlt in self.workloads[mid]:
        #         self.workloads[mid][wlt].update_constraints(
        #             time,
        #             WorkloadConstraint(
        #                 device_id=constraint.did,
        #                 stage_id=constraint.sid, 
        #                 microbatch_id=constraint.mid, 
        #                 workload_type=constraint.wtype
        #             )
        #         ) 

    def update_memory_usage(self, workload:Workload, sim = False):
        begin_memory = self.memory_usage
        begin_peak_memory = self.peak_memory_usage
        begin_emb_memory_gradient_usage = self.emb_memory_gradient_usage
        begin_grad_memory_usage = self.grad_memory_usage
        layers_per_stage = self.layer_num
        peak_memory = self.memory_usage

        if workload.wtype == WorkloadType.F:
            self.memory_usage += (Activation.FULL * (1 - self.recomp) + Activation.INPUT * self.recomp) * layers_per_stage
            if self.sid == 0: # Including emb layer
                self.memory_usage += Activation.EMB
            peak_memory = self.memory_usage
            if self.sid == self.stage_num - 1:
                self.memory_usage += Activation.HEAD
                self.memory_usage += Activation.LOSS / 2
                peak_memory = self.memory_usage + Activation.LOSS / 2 # Need copy a FP32 logits

        elif workload.wtype == WorkloadType.R:
            self.memory_usage += (Activation.FULL - Activation.INPUT) * layers_per_stage
            peak_memory = max(peak_memory, self.memory_usage)

        elif workload.wtype == WorkloadType.W:
            if self.grad_memory_usage == 0:
                self.grad_memory_usage = self.para_num * gpc["FP16"] / gpc["TP_SIZE"] # Para gradients
                if self.sid == 0: # Gradient is already stored
                    assert self.emb_memory_gradient_usage > 0, "W should after B."
                    self.grad_memory_usage -= self.emb_memory_gradient_usage
                self.memory_usage += self.grad_memory_usage # Input gradient of layers
                peak_memory = self.memory_usage
            else:
                if self.sid != 0: # emb will be stored to the end
                    peak_memory += self.grad_memory_usage
            if self.sid == self.stage_num - 1:
                self.memory_usage -= Activation.LOSS / 4 # Input gradient of head
            self.memory_usage -= (Activation.FULL * ACT_W_RATIO) * layers_per_stage
            self.memory_usage -= Gradient.INPUT * layers_per_stage
            
        else:
            if self.bwd_split:
                if workload.wtype == WorkloadType.B:
                    self.memory_usage += Gradient.INPUT * layers_per_stage # Input gradient of layers
                    peak_memory = self.memory_usage
                    if self.sid == 0 and self.emb_memory_gradient_usage == 0: # Including emb layer
                        self.emb_memory_gradient_usage = Gradient.HEAD_INPUT
                        self.memory_usage += self.emb_memory_gradient_usage
                        peak_memory = max(peak_memory, self.memory_usage)
                    if self.sid == 0:
                        self.memory_usage -= Activation.EMB
                    if self.sid == self.stage_num - 1:
                        self.memory_usage -= Activation.HEAD * ACT_HEAD_B
                        self.memory_usage -= Activation.LOSS / 2
                        self.memory_usage += Activation.LOSS / 4 # Input gradient of head
                    self.memory_usage -= Activation.FULL * ACT_B_RATIO * layers_per_stage
                    if self.sid == 0:
                        peak_memory = max(self.memory_usage + self.emb_memory_gradient_usage, peak_memory)
            else:
                if workload.wtype == WorkloadType.B:
                    self.memory_usage -= Activation.FULL * layers_per_stage
                    if self.grad_memory_usage == 0:
                        self.grad_memory_usage = self.para_num * gpc["FP16"] / gpc["TP_SIZE"] # Para gradients
                        self.memory_usage += self.grad_memory_usage
                        peak_memory = self.memory_usage
                    if self.sid == self.stage_num - 1:
                        self.memory_usage -= Activation.HEAD * ACT_HEAD_W
                        self.memory_usage -= Activation.LOSS
        self.peak_memory_usage = peak_memory
        if sim:
            peak_memory_delta = self.peak_memory_usage - begin_memory
            memory_delta = self.memory_usage - begin_memory
            self.memory_usage = begin_memory
            self.peak_memory_usage = begin_peak_memory
            self.emb_memory_gradient_usage = begin_emb_memory_gradient_usage
            self.grad_memory_usage = begin_grad_memory_usage
            return peak_memory_delta, memory_delta

    def execute_workload(self, time, mid=None, workload_type=None)->Workload:
        if mid is not None and workload_type is not None and workload_type in self.workloads[mid]:
            w = self.workloads[mid][workload_type]
            if w.execute(time=time):
                return w
        else:
            if self.stage_type == StageType.EMBD and workload_type in (WorkloadType.B, WorkloadType.W):
                return None
            elif self.stage_type == StageType.CE and workload_type in (WorkloadType.W, ):
                return None
            # print("Lack of workload info.")
        return None

    def __repr__(self) -> str:
        return (f"StageClass(stage_id={self.sid}, "
                f"memory_usage={self.memory_usage})")
