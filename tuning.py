from re import I
import time
import random
import heapq
from functools import wraps
from collections import defaultdict


time_line = []

def time_recorder_decorator(time_line_list):
    """
    创建一个装饰器，用于记录被装饰函数的执行时间，并将其追加到指定的列表中。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 记录函数执行前的时间
            start_time = time.time()

            # 执行原始函数
            result = func(*args, **kwargs)

            # 记录函数执行后的时间
            end_time = time.time()

            # 计算花费的时间
            elapsed_time = end_time - start_time

            # 将花费的时间追加到列表中
            if time_line_list:
                time_line_list.append(elapsed_time + time_line_list[-1])
            else:
                time_line_list.append(elapsed_time)

            # 返回原始函数的执行结果
            return result
        return wrapper
    return decorator

def check_placement_convergence(placement_history, repeat_count=2):
    """
    placement_history: 列表，每项为序列化后的 placement
    repeat_count: 连续重复次数判定收敛
    返回 (converged: bool)
    """
    n = len(placement_history)
    if n < repeat_count:
        return False

    # 最近 repeat_count 次是否完全相同
    last_places = placement_history[-repeat_count:]
    if all(p == last_places[0] for p in last_places):
        return True

    return False

def check_partition_convergence(history, window=10):
    """
    history: 列表 每项为 (iteration, time_value, partition)
    window: 用最近 window 次迭代判断收敛
    返回 (converged, best_partition, best_time)
    """

    n = len(history)
    if n < window:
        return False, None, None

    recent = history[-window:]

    # 找到最近 window 中的最优
    times = [item[1] for item in recent]
    best_time = min(times)
    best_partition = None
    for it, t, part, place in recent:
        if t == best_time:
            best_partition = part
            break

    # 情况一 partition 模式来回震荡不再变化
    partitions = [tuple(item[2]) for item in recent]
    unique_parts = set(partitions)
    if len(unique_parts) <= 2:
        return True, best_partition, best_time

    # 情况二 全局最优不再改进
    # 这里确保不会对空列表取 min
    earlier = history[:-window]
    if earlier:
        earlier_best_time = min(item[1] for item in earlier)
        # 若最近 window 次的最优时间不比更早的时间好 说明不再改进
        if best_time >= earlier_best_time:
            return True, best_partition, best_time

    return False, None, None

def balanced_transpose(placement, layer_comp_time):
    # 原始行的总开销
    original_cost = [sum(layer_comp_time[l] for l in row) for row in placement]
    num_rows = len(placement)

    lid_time_list = []
    for lid in range(len(layer_comp_time)):
        lid_time_list.append((lid, round(layer_comp_time[lid], 0)))
    lid_time_list.sort(key=lambda x: x[1])

    # 新的空 placement
    new_place = [[] for _ in range(num_rows)]
    new_cost = [0] * num_rows

    # 一个小根堆按 (当前开销, 行号) 排序
    # 这样每次将更重的 layer 放到当前最轻的行
    heap = [(0, i) for i in range(num_rows)]
    heapq.heapify(heap)

    # 按 comp time 从大到小填充
    for lid, time in lid_time_list:
        cost, idx = heapq.heappop(heap)
        new_place[idx].append(lid)
        new_cost[idx] += time
        heapq.heappush(heap, (new_cost[idx], idx))

    # 最终按每行第一个 layer 升序排序保证干净布局
    for row in new_place:
        row.sort()
    new_place.sort(key=lambda row: row[0] if row else 1e9)
    return new_place

def serialize(p):
    return tuple(tuple(row) for row in p)


def tune_placement(model_placement, model_partition, layer_comp_time, device_execution_times, device_bubble_times, history):
    num_devices = len(model_placement)
    bubble_ratio = []
    for exec_t, bubble_t in zip(device_execution_times, device_bubble_times):
        if exec_t == 0:
            ratio = 0.0
        else:
            ratio = bubble_t / exec_t
        bubble_ratio.append(ratio)

    num_devices = len(model_placement)

    # source device 從 bubble 小到大
    source_list = sorted(range(num_devices),
                            key=lambda i: (bubble_ratio[i], device_bubble_times[i]))

    # target device 從 bubble 大到小
    target_list = sorted(range(num_devices),
                            key=lambda i: (bubble_ratio[i], device_bubble_times[i]),
                            reverse=True)

    # 序列化用於檢查重複


    old_state = serialize(model_placement)

    # 對所有 source device 嘗試
    for src in source_list:

        if not model_placement[src]:
            continue

        # 找 source device 中最小開銷 layer
        small_layer = min(model_placement[src], key=lambda l: layer_comp_time[l])
        small_cost = layer_comp_time[small_layer]

        # 對所有 target device 嘗試
        for tgt in target_list:

            if tgt == src:
                continue

            # 在 target device 中找 comp time 更大的 layer
            candidates = [l for l in model_placement[tgt]
                            if layer_comp_time[l] > small_cost]

            if not candidates:
                continue

            # 選擇 comp time 最小的 candidate
            large_layer = min(candidates, key=lambda l: layer_comp_time[l])

            # 嘗試交換
            new_place = [row[:] for row in model_placement]

            new_place[src].remove(small_layer)
            new_place[tgt].remove(large_layer)
            new_place[src].append(large_layer)
            new_place[tgt].append(small_layer)

            # 排序
            for row in new_place:
                row.sort()

            new_state = serialize(new_place)

            # 不可和歷史重複
            if new_state not in history and new_state != old_state:
                return new_state

    # 所有交換方案都無法使用
    return model_placement

def tune_partition_random(partition, placement, layer_comp_time, device_execution_times, device_bubble_times, history, stage=0):
    num_devices = len(partition)
    src = random.randrange(num_devices)
    if partition[src] == 0:
        return partition

    dst = random.randrange(num_devices)
    while dst == src:
        dst = random.randrange(num_devices)

    new_partition = list(partition)
    new_partition[src] -= 1
    new_partition[dst] += 1

    return new_partition

def tune_placement_random(model_placement, model_partition, layer_comp_time, device_execution_times, device_bubble_times, history):
    new_partition = model_partition
    new_placement = model_placement

    num_devices = len(new_partition)

    src = random.randrange(num_devices)
    if len(new_placement[src]) == 0:
        return new_placement
    src_layer = random.choice(new_placement[src])

    dst = random.randrange(num_devices)
    while dst == src:
        dst = random.randrange(num_devices)
    dst_layer = random.choice(new_placement[dst])

    new_placement[src].remove(src_layer)
    new_placement[dst].append(src_layer)

    new_placement[src].append(dst_layer)
    new_placement[dst].remove(dst_layer)

    return new_placement

def tune_partition_dfs(partition, placement, layer_comp_time, device_execution_times, device_bubble_times, history, stage=0):
    num_devices = len(partition)
    src_list = range(num_devices)
    dst_list = range(num_devices)
    for src in src_list:
        for dst in dst_list:
            if dst == src:
                continue
            new_partition = list(partition)
            new_partition[src] -= 1
            new_partition[dst] += 1
            if tuple(new_partition) in history:
                continue
            else:
                return new_partition
    return partition

def tune_placement_dfs(model_placement, model_partition, layer_comp_time, device_execution_times, device_bubble_times, history):
    num_devices = len(model_placement)
    for src in range(num_devices):
        src_layers = model_placement[src]
        for dst in range(num_devices):
            if dst == src:
                continue
            dst_layers = model_placement[dst]
            for src_layer in src_layers:
                for dst_layer in dst_layers:
                    new_placement = list(model_placement)
                    new_placement[src].remove(src_layer)
                    new_placement[dst].append(src_layer)
                    new_placement[src].append(dst_layer)
                    new_placement[dst].remove(dst_layer)
                    if serialize(new_placement) in history:
                        continue
                    else:
                        return new_placement

    return new_placement

def tune_partition(partition, placement, layer_comp_time, device_execution_times, device_bubble_times, history, stage=0):
    n = len(partition)
    indexes = list(range(n))

    if stage == 0:
        workloads = []
        for lids in placement:
            w = 0
            for lid in lids:
                w += layer_comp_time[lid]
            workloads.append(w)

        sorted_idx = sorted(indexes, key=lambda i: workloads[i])
        max_list = sorted_idx[:]
        min_list = sorted_idx[::-1]

    elif stage == 1:
        bubble_ratio = []
        for exec_t, bubble_t in zip(device_execution_times, device_bubble_times):
            if exec_t == 0:
                ratio = 0.0
            else:
                ratio = bubble_t / exec_t
            bubble_ratio.append(ratio)

        sorted_idx = sorted(indexes,
                            key=lambda i: (bubble_ratio[i], device_bubble_times[i]),
                            reverse=True)
        max_list = sorted_idx[:]
        min_list = sorted_idx[::-1]

    # Skip the same cases in history
    for max_idx in max_list:
        for min_idx in min_list:

            if max_idx == min_idx:
                continue

            if partition[min_idx] <= 1:
                continue

            # new partition
            new_partition = list(partition)
            new_partition[min_idx] -= 1
            new_partition[max_idx] += 1

            # check history
            if tuple(new_partition) not in history:
                return new_partition

    return partition
