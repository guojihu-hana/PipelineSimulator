from ..config import *
def get_mist_predefined_partition_placement(seq_len, device_num, layer_num):
    if GEMMA:
        mist_layer_assignments = {
            32 : { 4 : [8, 9, 9, 6], },
            64 : { 8 : [3, 10, 10, 10, 10, 10, 10, 1], },
            128 : { 16 : [1, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 2], },
        }
    if DEEPSEEK:
        mist_layer_assignments = {
            16 : { 4 : [5, 4, 4, 3], },
            32 : { 8 : [6, 4, 4, 4, 4, 4, 4, 2], },
            64 : { 8 : [7, 9, 9, 9, 9, 9, 9, 3], },
        }
    if NEMOTRONH:
        mist_layer_assignments = {
            28 : { 4 : [8, 8, 8, 4], },
            56 : { 8 : [6, 7, 8, 7, 8, 7, 8, 5], },
            112 : {
                8 : [13, 14, 16, 16, 14, 14, 16, 9],
                16 : [6, 7, 7, 7, 9, 7, 7, 9, 7, 7, 7, 7, 9, 7, 7, 2],
            },
        }
    layer_assignment = mist_layer_assignments[layer_num][device_num]
    assert sum(layer_assignment) == layer_num, f"Mist {sum(layer_assignment)} != {layer_num}"
    return layer_assignment

def get_octopipe_predefined_partition_placement(seq_len, device_num, layer_num):
    layer_assignment = []
    if GEMMA:
        if seq_len == 2*K:
            if device_num == 4:
                layer_assignment=[9,9,8,6]
            if device_num == 8:
                if layer_num == 64:
                    # Mist 256
                    layer_assignment=[5, 9, 9, 9, 9, 9, 9, 5]
                    layer_assignment=[9, 9, 9, 8, 8, 8, 8, 5]
                    # Mist 512
                    layer_assignment=[5, 9, 9, 9, 9, 9, 9, 5]
                    layer_assignment=[9, 9, 9, 8, 8, 8, 8, 5]
                elif layer_num == 128:
                    layer_assignment=[14, 17, 17, 17, 17, 17, 17, 12]
                else:
                    layer_assignment = [layer_num//device_num for i in range(device_num)]
            if device_num == 16:
                # Mist
                layer_assignment=[1, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5]
                layer_assignment=[9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 1]
        if seq_len == 4*K:
            if device_num == 4:
                layer_assignment=[10,9,8,5]
            if device_num == 8:
                if layer_num == 64:
                    layer_assignment=[9, 9, 9, 9, 8, 8, 8, 4]
                elif layer_num == 128:
                    layer_assignment=[18, 17, 17, 17, 17, 17, 17, 8]
                else:
                    layer_assignment = [layer_num//device_num for i in range(device_num)]
            
            if device_num == 16:
                # Mist
                layer_assignment=[1, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5]
                layer_assignment=[9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 1]
    if DEEPSEEK:
        if device_num == 4:
            layer_assignment=[5,4,4,3]
            if seq_len == 4*K:
                layer_assignment=[6,4,4,2]
        if device_num == 8:
            if layer_num == 64:
                layer_assignment=[12,8,8,8,8,8,8,4]
            elif layer_num == 32:
                layer_assignment=[6,3,4,4,4,4,4,3]
            else:
                layer_assignment=[6,4,4,4,4,4,4,2]
    if NEMOTRONH:
        if device_num == 4:
            layer_assignment=[7,7,7,7]
            if seq_len == 2*K:
                layer_assignment=[8,7,8,5]
            if seq_len == 4*K:
                layer_assignment=[8,8,8,4]
            else:
                layer_assignment=[8,8,8,4]
        if device_num == 8:
            if layer_num == 56:
                if seq_len == 2*K:
                    layer_assignment=[9,8,7,8,7,7,7,3]
                if seq_len == 4*K:
                    layer_assignment=[9,8,7,8,8,7,7,2]
                if seq_len == 8*K:
                    layer_assignment=[8,8,8,8,8,8,7,1]
            if layer_num == 112:
                layer_assignment=[16, 16, 15, 15, 16, 16, 15, 3]
        if device_num == 16:
            if seq_len == 2*K:
                layer_assignment=[8, 6, 7, 6, 8, 7, 7, 8, 7, 8, 8, 7, 8, 8, 8, 1]
            if seq_len == 4*K:
                layer_assignment=[8, 6, 7, 6, 8, 7, 7, 8, 7, 8, 8, 7, 8, 8, 8, 1]    
    if VARYLEN:
        if seq_len == 1*K:
            layer_assignment=[14, 14, 15, 15, 14, 14, 15, 11]
        if seq_len == 2*K:
            layer_assignment=[14, 15, 15, 15, 14, 14, 15, 10]
        if seq_len == 4*K:
            layer_assignment=[13, 14, 16, 16, 14, 15, 16, 8]
        if seq_len == 8*K:
            layer_assignment=[16, 16, 15, 15, 16, 16, 14, 4]
            layer_assignment=[15, 16, 16, 15, 16, 16, 15, 3]
        if seq_len == 16*K:
            layer_assignment=[13, 17, 17, 15, 17, 16, 16, 1]
        if seq_len == 32*K:
            if device_num == 8:
                layer_assignment=[14, 14, 17, 17, 17, 16, 16, 1]
                layer_assignment=[16, 16, 16, 16, 16, 15, 16, 1]
            if device_num == 4:
                layer_assignment=[31, 33, 33, 15]
    assert sum(layer_assignment) == layer_num, f"{sum(layer_assignment)} != {layer_num}"
    return layer_assignment
