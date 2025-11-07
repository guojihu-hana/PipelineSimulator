import random
from dataclasses import dataclass
from simulator.abstract.variables import *
from simulator.model_config import *
# --------------------- Solver config ---------------------
BASE_SOLUTION = True
RUN_MODE = RunMode.LAYERWISE_GUROBI_SOLVE
RUN_MODE = RunMode.GUROBI_SOLVE
RUN_MODE = RunMode.CHIMERA
RUN_MODE = RunMode.SIM_SOLVE

MICRO_BATCH_TIME = [1 + random.randint(0, 1) for _ in range(MICRO_BATCH_NUM)]
MICRO_BATCH_TIME = [1 if _ % 2 else 2 for _ in range(MICRO_BATCH_NUM)]

SOLVING_TIME_LIMIT = 60 * 30
SCHEDULE_METHOD = Schedule.Layerwise
SCHEDULE_METHOD = Schedule.STANDARD_1F1B
# SCHEDULE_METHOD = Schedule.ReCycle
# SCHEDULE_METHOD = Schedule.STANDARD_INTERLEAVED
# SCHEDULE_METHOD = Schedule.STANDARD_ZBH
# SCHEDULE_METHOD = Schedule.Mist
SCHEDULE_METHOD = Schedule.OctoPipe

# SCHEDULE_METHOD = Schedule.ZBV
# SCHEDULE_METHOD = Schedule.STANDARD_AFAB
STAGE_PLACEMENT = Placement.INTERLEAVED
# STAGE_PLACEMENT = Placement.SEARCHED
# STAGE_PLACEMENT = Placement.WAVELIKE
SPLIT_BACKPROP = True
if SCHEDULE_METHOD == Schedule.STANDARD_INTERLEAVED:
    STAGE_PLACEMENT = Placement.INTERLEAVED
    CHUNK_NUM = LAYER_NUM // DEVICE_NUM
    SPLIT_BACKPROP = False
if SCHEDULE_METHOD in (Schedule.STANDARD_ZBH, Schedule.STANDARD_1F1B, Schedule.STANDARD_AFAB):
    CHUNK_NUM = 1
# --------------------- Solver config ---------------------
Hierarchical = True
test_upp = True if SCHEDULE_METHOD == Schedule.OctoPipe else False
HETER_DEVICE = True
HETER_RATIO = 1
COMP_POWER = 2
OVERLAP_AWARE_SCHEDULE = True if not HETER_DEVICE else False
OVERLAP_AWARE_SCHEDULE = True
# --------------------- Simulator config ---------------------
FIND_OPTIMAL_RECOMP = False
TIME_LIMIT = 150000
HEAD_DP = False if test_upp else False
# [1, 2, 100, None]
OVERLAP_DEGREE = None
MEMORY_CONSTRAIN = 0.9
MEMORY_REDUCATION = 0.0
IDEAL_SITUATION = False

# Gemma
EMB_F_TIME = 0
EMB_B_TIME = 0
EMB_W_TIME = 0
HEAD_F_TIME = 0
HEAD_B_TIME = 0
HEAD_W_TIME = 0
CE_F_TIME = 0
CE_B_TIME = 0
CE_W_TIME = 0
F_TIME = 0
B_TIME = 0
W_TIME = 0
COMM_TIME = 0


if SCHEDULE_METHOD in (Schedule.STANDARD_ZBH, Schedule.ZBV):
    SPLIT_BACKPROP = True
    if SCHEDULE_METHOD == Schedule.ZBV:
        STAGE_PLACEMENT = Placement.WAVELIKE

if SPLIT_BACKPROP:
    EMB_B_TIME = 0
    EMB_W_TIME = 0
    HEAD_W_TIME = HEAD_B_TIME // 2
    HEAD_B_TIME = HEAD_B_TIME // 2
    CE_B_TIME = 0
    CE_W_TIME = 0
    W_TIME = B_TIME // 2
    B_TIME = B_TIME // 2

if IDEAL_SITUATION:
    EMB_F_TIME = 0
    EMB_B_TIME = 0
    EMB_W_TIME = 0
    HEAD_F_TIME = 0
    HEAD_B_TIME = 0
    HEAD_W_TIME = 0
    CE_F_TIME = 0
    CE_B_TIME = 0
    CE_W_TIME = 0

LAYERWISE = False
RECOMP = False
AUTO_RECOMP_SEARCH = False
RUN_SCHEDULE = False
RUN_STANDARD_ZBV = False
if not RUN_SCHEDULE and RUN_STANDARD_ZBV:
    print("Overlooking non-transformer layers")
    EMB_F_TIME = 0
    EMB_B_TIME = 0
    EMB_W_TIME = 0
    HEAD_F_TIME = 0
    HEAD_B_TIME = 0
    HEAD_W_TIME = 0
    CE_F_TIME = 0
    CE_B_TIME = 0
    CE_W_TIME = 0
    F_TIME = 12
    B_TIME = 12
    W_TIME = 12

if DEEPSEEK + GEMMA + NEMOTRONH > 1:
    print(f"DeepSeek:{DEEPSEEK}, Gemma:{GEMMA}, NemotronH:{NEMOTRONH}")

SAVE_MEMORY = False
CONSTRAIN_WARMUP = False
SWITCH_WORKLOAD_TYPE = True

F_TIME = 10
F_TIMES = [F_TIME] * LAYER_NUM
B_TIMES = [F_TIME] * LAYER_NUM
W_TIMES = [F_TIME] * LAYER_NUM

if not IDEAL_SITUATION:
    if GEMMA:
        try:
            from data.profiled_data import profiled_data
            ratios = profiled_data["GEMMA"][HIDDEN_SIZE][SEQ_LEN][VOCAB_SIZE]
            [tf_tf, tb_tf, tw_tf, _, _, _, hf_tf, hb_tf, hw_tf] = [round(r, 1) for r in ratios]
            B_TIMES = [t*(tb_tf+tw_tf) for i,t in enumerate(F_TIMES)]
            HEAD_F_TIME = F_TIME * hf_tf
            HEAD_B_TIME = F_TIME * (hb_tf + hw_tf)
            if SPLIT_BACKPROP:
                B_TIMES = [t*tb_tf for i,t in enumerate(F_TIMES)]
                W_TIMES = [t*tw_tf for i,t in enumerate(F_TIMES)]
                HEAD_B_TIME = F_TIME * hb_tf
                HEAD_W_TIME = F_TIME * hw_tf
        except:
            print("----- No profiled data! Use predefined ratios. -----")

    if DEEPSEEK:
        try:
            from data.profiled_data import profiled_data
            ratios = profiled_data["DEEPSEEK"][HIDDEN_SIZE][SEQ_LEN][VOCAB_SIZE]
            [tf_tf, tb_tf, tw_tf, mf_tf, mb_tf, mw_tf, hf_tf, hb_tf, hw_tf] = [round(r, 1) for r in ratios]
            B_TIMES = [t*(mb_tf+mw_tf) if i >= LAYER_NUM//PP_SIZE - 1  else t * (tb_tf+tw_tf) for i,t in enumerate(F_TIMES)]
            HEAD_F_TIME = F_TIME * hw_tf
            HEAD_B_TIME = F_TIME * (hb_tf+hw_tf)
            if SPLIT_BACKPROP:
                if tw_tf == 0:
                    tw_tf = 0.2
                    tb_tf -= tw_tf
                B_TIMES = [t*mb_tf if i >= LAYER_NUM//PP_SIZE - 1  else t * tb_tf for i,t in enumerate(F_TIMES)]
                W_TIMES = [t*mw_tf if i >= LAYER_NUM//PP_SIZE - 1  else t * tw_tf for i,t in enumerate(F_TIMES)]
                HEAD_B_TIME = F_TIME * hb_tf
                HEAD_W_TIME = F_TIME * hw_tf
            F_TIMES = [t*mf_tf if i >= LAYER_NUM//PP_SIZE - 1 else t for i,t in enumerate(F_TIMES)]
        except:
            print("----- No profiled data! Use predefined ratios. -----")

    if NEMOTRONH:
        diff = 3 * N_SCALE
        try:
            from data.profiled_data import profiled_data
            ratios = profiled_data["NEMOTRONH"][HIDDEN_SIZE][SEQ_LEN][VOCAB_SIZE]
            [tf_mf, tb_mf, tw_mf, mf_mf, mb_mf, mw_mf, hf_mf, hb_mf, hw_mf] = [round(r, 1) for r in ratios]
            print(hf_mf,hb_mf,hw_mf)
            B_TIMES = [t*(tb_mf+tw_mf) if (i+1)%diff==0  else t * mb_mf for i,t in enumerate(F_TIMES)]
            HEAD_F_TIME = F_TIME * hf_mf
            HEAD_B_TIME = F_TIME * (hb_mf + hw_mf)
            if SPLIT_BACKPROP:
                B_TIMES = [t*tb_mf if (i+1)%diff==0  else t * (mb_mf-0.1) for i,t in enumerate(F_TIMES)]
                W_TIMES = [t*tw_mf if (i+1)%diff==0  else t * 0.1 for i,t in enumerate(F_TIMES)]
                HEAD_B_TIME = F_TIME * hb_mf
                HEAD_W_TIME = F_TIME * hw_mf
            F_TIMES = [t*tf_mf if (i+1)%diff==0 else t for i,t in enumerate(F_TIMES)]
        except:
            print("----- No profiled data! Use predefined ratios. -----")


    if VARYLEN:
        diff = 12
        from data.profiled_data import profiled_data
        ratios = profiled_data["NEMOTRONH"][HIDDEN_SIZE][SEQ_LEN][VOCAB_SIZE]
        [tf_mf, tb_mf, tw_mf, mf_mf, mb_mf, mw_mf, hf_mf, hb_mf, hw_mf] = [round(r, 1) for r in ratios]
        B_TIMES = [t*(tb_mf+tw_mf) if (i+1)%diff==0  else t * mb_mf for i,t in enumerate(F_TIMES)]
        HEAD_F_TIME = F_TIME * hf_mf
        HEAD_B_TIME = F_TIME * (hb_mf + hw_mf)
        if SPLIT_BACKPROP:
            B_TIMES = [t*tb_mf if (i+1)%diff==0  else t * (mb_mf-0.1) for i,t in enumerate(F_TIMES)]
            W_TIMES = [t*tw_mf if (i+1)%diff==0  else t * 0.1 for i,t in enumerate(F_TIMES)]
            HEAD_B_TIME = F_TIME * hb_mf
            HEAD_W_TIME = F_TIME * hw_mf
        F_TIMES = [t*tf_mf if (i+1)%diff==0 else t for i,t in enumerate(F_TIMES)]
        print("------ Test vary length sequence. -----")

SCHEDULE_UNIT = MICRO_BATCH_NUM // 1
REVERSE_LAST_STAGES = False
REVERSE_FIRST_STAGES = False
# Run standard ZBV ---------------------
# SCHEDULE_METHOD = Schedule.ZBV
# RUN_SCHEDULE = False
# RUN_STANDARD_ZBV = True
# Run standard ZBV ---------------------
DENSITY_MAX = 1
DENSITY_MIN = 1
# --------------------- Simulator config ---------------------


# Memory overhead calculation
GPU_MAX_MEM = 80 * G / G
FP32 = 4 # 4 Bytes
FP16 = 2 # 2 Bytes
MIX_TRAINING = True
DATA_TYPE: int = FP16 if MIX_TRAINING else FP32
b = MICRO_BATCH_SIZE
s = SEQ_LEN
h = HIDDEN_SIZE
a = NUM_ATTENTION_HEAD
l = LAYER_NUM
v = VOCAB_SIZE
i = INTER_SIZE

LAYER_PARA_NUM = 4 * h * h + 3 * h * i + 2 * h if MODEL_TYPE in ("LLAMA", "Qwen") else 12 * h * h + 13 * h 
HEAD_PARA_NUM = v * h
PARAMETER_NUM = LAYER_PARA_NUM * LAYER_NUM + HEAD_PARA_NUM

LAYER_MEMORY = DATA_TYPE * LAYER_PARA_NUM / G
HEAD_MEMORY = DATA_TYPE * HEAD_PARA_NUM / G

OPTIMIZER_MEMORY = PARAMETER_NUM * FP32 * 3 / G # Optimizer status * 2, gradients * 1, model parameters * 1
MAX_ACTIVATION_TIMES_OF_STAGE_NUM = 1

@dataclass
class Parameter:
    EMB: int = b * v * h
    HEAD: int = b * v * h
    LAYER: int = LAYER_PARA_NUM

@dataclass
class StateMemory:
    EMB: int = DATA_TYPE * Parameter.EMB / G / TP_SIZE
    HEAD: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    LAYER: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE
    # Optimizer M + V, gradients * 1, model * 1
    OPTIMIZER: int = FP32 * 4 * (Parameter.LAYER * l + Parameter.EMB + Parameter.HEAD) / G / (TP_SIZE * PP_SIZE) / ZERO_SIZE

ACT_OPT_COE = 0.18298 # adjust by profiling results
ACT_B_RATIO = 0.5669
ACT_W_RATIO = 1 - ACT_B_RATIO
ACT_HEAD_B = 2/3
ACT_HEAD_W = 1 - ACT_HEAD_B
@dataclass
class Activation:
    INPUT: int = (2*b*s*h) / G / TP_SIZE
    FULL: int = (34*b*s*h + 5*b*s*s*a) * ACT_OPT_COE / G / TP_SIZE
    LOSS: int = (2*FP32*b*s*v) / G / TP_SIZE
    HEAD: int = (3*FP16*b*s*h) / G / TP_SIZE
    EMB: int = (FP16*b*s*h) / G / TP_SIZE

GRAD_COE = 0.225
# 1.5->0.6
@dataclass
class Gradient:
    INPUT: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE * GRAD_COE
    PARAMETER: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE
    HEAD_INPUT: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    HEAD_PARA: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    # HEAD_INPUT: int = 0
    # HEAD_PARA: int = 0



# --------------------- Painter Config ---------------------
PIXEL_BASE = 1
PP_HEIGHT = 25
PP_ALIGN = 5
SHOW_WORKLOAD_TEXT = True
if CHUNK_NUM > PP_SIZE:
    SHOW_WORKLOAD_TEXT = False
# --------------------- Painter Config ---------------------

# --------------------- Save File Config ---------------------
SAVE_RES_TO_FILE = True
SCH_FILE_PATH = f"schedule_results/schedules/heter{HETER_DEVICE}/vs{VOCAB_SIZE}_l{LAYER_NUM}_s{SEQ_LEN}_h{HIDDEN_SIZE}/mb{MICRO_BATCH_NUM}_pp{PP_SIZE}_tp{TP_SIZE}_zr{ZERO_SIZE}_c{CHUNK_NUM}/{SCHEDULE_METHOD.name}_{STAGE_PLACEMENT.name}_w{SPLIT_BACKPROP}_l{LAYERWISE}_o{OVERLAP_DEGREE}.txt"
PLA_FILE_PATH = f"schedule_results/placements/heter{HETER_DEVICE}/vs{VOCAB_SIZE}_l{LAYER_NUM}_s{SEQ_LEN}_h{HIDDEN_SIZE}/mb{MICRO_BATCH_NUM}_pp{PP_SIZE}_tp{TP_SIZE}_zr{ZERO_SIZE}_c{CHUNK_NUM}/{SCHEDULE_METHOD.name}_{STAGE_PLACEMENT.name}_w{SPLIT_BACKPROP}_l{LAYERWISE}_o{OVERLAP_DEGREE}.txt"
TEMP_PLA_PATH = f"schedule_results/placement.txt"
TEMP_RES_PATH = f"schedule_results/result.txt"

STAGE_NUM = int(DEVICE_NUM * CHUNK_NUM)
assert STAGE_NUM <= LAYER_NUM, f"Stage ({STAGE_NUM}) should be less than Layer ({LAYER_NUM}). "

WORKLOAD_TYPE_NUM = 3
if not SPLIT_BACKPROP:
    B_TIME += W_TIME
    WORKLOAD_TYPE_NUM = 2

MAX_ACTIVATION_COUNTS = int(STAGE_NUM * 2)
MAX_ACT = 1
PROFILE_GENERATION = True