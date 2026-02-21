from .Workload import *
import copy

@dataclass
class StageType:
    EMBD = 1
    LAYER = 2
    HEAD = 3
    CE = 4
    LAYERS = 5

class Stage:

    def __init__(self, stage_idx: int, total_stage_num: int, layer_num: int, device_idx:int, schedule_method, training_config: TrainingConfig , para_num:int, stage_type: StageType, mid_offset:int, layer_idx_start: int, layerwise:bool = False, recomp: bool = False, split_recomp: bool = False, comp_power: float = 1, placement = None):
        self.sid: int = stage_idx
        self.did: int = device_idx
        self.schedule_method = schedule_method
        self.total_stage_num  = total_stage_num
        self.layer_num  = layer_num # Number of layers in this stage, only used when layerwise is False. If layerwise is True, it must be 1.

        self.tc         = training_config
        self.bwd_split  = self.tc.bwd_split
        self.nmb        = self.tc.micro_batch_num
        self.layer_f_times = self.tc.layer_f_times
        self.layer_b_times = self.tc.layer_b_times
        self.layer_w_times = self.tc.layer_w_times
        self.tp_size = self.tc.tp_size
        self.pp_size = self.tc.pp_size
        self.total_layer_num = self.tc.layer_num
        self.vocab_parallel = self.tc.vocab_parallel
        self.fp16 = self.tc.fp16
        self.fp32 = self.tc.fp32
        self.placement = placement

        self.mid_offset: int = mid_offset
        self.para_num: int = para_num / 1024**3
        self.model_memory_usage = self.para_num * self.fp16 / self.tp_size
        self.grad_memory_usage = 0
        self.emb_memory_gradient_usage = 0
        self.opt_memory_usage = self.para_num * 3 * self.fp32 / self.tp_size / gpc["ZERO_SIZE"]
        self.memory_generated_by_mb : int = [0] * self.nmb
        self.memory_usage: int = self.model_memory_usage + self.grad_memory_usage + self.opt_memory_usage
        self.peak_memory_usage: int = self.model_memory_usage + self.grad_memory_usage + self.opt_memory_usage
        self.workloads: dict[int, dict[WorkloadType, Workload]] = {}  
        self.stage_type: StageType = stage_type
        self.recomp = recomp
        self.split_recomp = split_recomp
        self.layerwise = layerwise
        self.layer_idx_start = layer_idx_start
        self.layer_idxs = list(range(layer_idx_start, layer_idx_start + self.layer_num))
        self.comp_power = comp_power
        if layerwise: 
            assert self.layer_num == 1, f"LAYERWISE require 1 layer per stage but got {self.layer_num}"
        self._add_workload()
    
    def get_workload_duration(self, wtype:WorkloadType)->float:
        if wtype in (WorkloadType.F, WorkloadType.R):
            duration = sum([self.layer_f_times[idx] for idx in self.layer_idxs])
        elif wtype == WorkloadType.B:
            duration = sum([self.layer_b_times[idx] for idx in self.layer_idxs])
            if self.recomp:
                if not self.split_recomp:
                    duration += sum([self.layer_f_times[idx] for idx in self.layer_idxs])
        elif wtype == WorkloadType.W:
            duration = sum([self.layer_w_times[idx] for idx in self.layer_idxs])
        else:
            raise ValueError(f"Wrong workload type: {wtype}.")
        return int(duration / self.comp_power)

    def _add_workload(self) -> None:
        total_stages = self.total_layer_num + 3 if self.layerwise else self.total_stage_num
        if self.vocab_parallel:
            total_stages = self.total_stage_num + 1
        for mid in range(self.mid_offset, self.mid_offset + self.nmb):
            self.workloads[mid] = {}
            fpw = Workload(
                schedule_method = self.schedule_method,
                bwd_split=self.bwd_split,
                device_idx=self.did,
                stage_idx=self.sid,
                microbatch_idx=mid,
                wtype=WorkloadType.F,
                duration=self.get_workload_duration(wtype=WorkloadType.F),
                recomp=self.recomp,
                split_recomp=self.split_recomp,
                total_stage_num=total_stages,
                layer_idxs=self.layer_idxs,
                comp_power=self.comp_power,
                vocab_parallel=self.vocab_parallel,
                placement=self.placement,
            )
            self.workloads[mid][WorkloadType.F] = fpw

            igw = Workload(
                schedule_method = self.schedule_method,
                bwd_split=self.bwd_split,
                device_idx=self.did,
                stage_idx=self.sid,
                microbatch_idx=mid,
                wtype=WorkloadType.B,
                duration=self.get_workload_duration(wtype=WorkloadType.B),
                recomp=self.recomp,
                split_recomp=self.split_recomp,
                total_stage_num=total_stages,
                layer_idxs=self.layer_idxs,
                comp_power=self.comp_power,
                vocab_parallel=self.vocab_parallel,                
                placement=self.placement,
            )
            self.workloads[mid][WorkloadType.B] = igw

            if self.recomp:
                if self.split_recomp:
                    rfw = Workload(
                        schedule_method = self.schedule_method,
                        bwd_split=self.bwd_split,
                        device_idx=self.did,
                        stage_idx=self.sid,
                        microbatch_idx=mid,
                        wtype=WorkloadType.R,
                        duration=self.get_workload_duration(wtype=WorkloadType.R,),
                        recomp=self.recomp,
                        split_recomp=self.split_recomp,
                        total_stage_num=total_stages,
                        layer_idxs=self.layer_idxs,
                        comp_power=self.comp_power,
                        vocab_parallel=self.vocab_parallel,
                        placement=self.placement,
                    )
                    self.workloads[mid][WorkloadType.R] = rfw

            if self.bwd_split:
                pgw = Workload(
                    schedule_method = self.schedule_method,
                    bwd_split=self.bwd_split,
                    device_idx=self.did,
                    stage_idx=self.sid,
                    microbatch_idx=mid,
                    wtype=WorkloadType.W,
                    duration=self.get_workload_duration(wtype=WorkloadType.W,),
                    split_recomp=self.split_recomp,
                    recomp=self.recomp,
                    total_stage_num=total_stages,
                    layer_idxs=self.layer_idxs,
                    comp_power=self.comp_power,
                    vocab_parallel=self.vocab_parallel,
                    placement=self.placement,
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
            elif self.sid == c_sid and self.sid == constraint.total_stage_num - 1:
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
            if self.sid == self.total_stage_num - 1:
                self.memory_usage += Activation.HEAD
                self.memory_usage += Activation.LOSS / 2
                peak_memory = self.memory_usage + Activation.LOSS / 2 # Need copy a FP32 logits

        elif workload.wtype == WorkloadType.R:
            self.memory_usage += (Activation.FULL - Activation.INPUT) * layers_per_stage
            peak_memory = max(peak_memory, self.memory_usage)

        elif workload.wtype == WorkloadType.W:
            if self.grad_memory_usage == 0:
                self.grad_memory_usage = self.para_num * self.fp16 / self.tp_size # Para gradients
                if self.sid == 0: # Gradient is already stored
                    assert self.emb_memory_gradient_usage > 0, "W should after B."
                    self.grad_memory_usage -= self.emb_memory_gradient_usage
                self.memory_usage += self.grad_memory_usage # Input gradient of layers
                peak_memory = self.memory_usage
            else:
                if self.sid != 0: # emb will be stored to the end
                    peak_memory += self.grad_memory_usage
            if self.sid == self.total_stage_num - 1:
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
                    if self.sid == self.total_stage_num - 1:
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
                        self.grad_memory_usage = self.para_num * self.fp16 / self.tp_size # Para gradients
                        self.memory_usage += self.grad_memory_usage
                        peak_memory = self.memory_usage
                    if self.sid == self.total_stage_num - 1:
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
