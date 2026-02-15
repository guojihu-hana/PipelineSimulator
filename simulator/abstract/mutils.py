from ..config import *

class ExecuteStrategy:
    def __init__(self, overlap_aware, save_memory, constrain_warmup, swith_workload_type):
        self.overlap_aware = overlap_aware
        self.save_memory = save_memory
        self.constrain_warmup = constrain_warmup
        self.switch_workload_type = swith_workload_type

class TrainingConfig:
    def __init__(self, pp_size, tp_size, dp_size, 
                bwd_split, vocab_parallel, overlap_aware, save_memory, constrain_warmup, swith_workload_type,
                layer_num, chunk_num, device_num, micro_batch_num, micro_batch_size,
                batch_size = None, layer_f_times=None, layer_b_times=None, layer_w_times=None,
                layer_f_memories=None, layer_b_memories=None):
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.bwd_split = bwd_split
        self.vocab_parallel = vocab_parallel
        self.overlap_aware = overlap_aware
        self.save_memory = save_memory
        self.constrain_warmup = constrain_warmup
        self.switch_workload_type = swith_workload_type
        self.layer_num = layer_num
        self.chunk_num = chunk_num
        self.device_num = device_num
        self.micro_batch_num = micro_batch_num
        self.micro_batch_size = micro_batch_size
        self.batch_size = batch_size if batch_size else micro_batch_num * micro_batch_size
        self.layer_f_times = layer_f_times if layer_f_times else [10 for _ in range(layer_num)]
        self.layer_b_times = layer_b_times if layer_b_times else [10 for _ in range(layer_num)]
        self.layer_w_times = layer_w_times if layer_w_times else [0 for _ in range(layer_num)]
        self.layer_f_memories = layer_f_memories
        self.layer_b_memories = layer_b_memories
        self.fp16 = 2
        self.fp32 = 4 # bytes

def dict_to_2d_list(nested_dict):
    rows = sorted(nested_dict.keys())
    cols = sorted(set().union(*(d.keys() for d in nested_dict.values())))
    return [[nested_dict[row].get(col, None) for col in cols] for row in rows]



