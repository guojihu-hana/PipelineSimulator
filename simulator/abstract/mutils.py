from ..config import *

def dict_to_2d_list(nested_dict):
    rows = sorted(nested_dict.keys())
    cols = sorted(set().union(*(d.keys() for d in nested_dict.values())))
    return [[nested_dict[row].get(col, None) for col in cols] for row in rows]

if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE:
    assert SCHEDULE_METHOD == Schedule.Layerwise, "SCHEDULE_METHOD should be Layerwise"

STAGE_SEARCH_METHOD = StageSearchOrder.Random

GLOBAL_TIME = 0

def UPDATE_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME+=1

def DECREASE_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME-=1

def GET_TIME():
    global GLOBAL_TIME
    return GLOBAL_TIME

def RESET_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME = 0


def is_head_layer(sid, total_layer_num=LAYER_NUM, layerwise=LAYERWISE):
    if layerwise and total_layer_num + 1 == sid:
        return True
    return False



