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

def solve_placement_min_pp_comp_time(
    model_partition,
    model_placement,
    layer_comp_time,
    *,
    beam_width: int = 512,
    top_n: int = 1,
    normalize_equal_cost_blocks: bool = True,
):
    """
    Diverse candidate placement generator.

    Generates a large, diverse set of candidate placements using multiple
    strategies so that the caller can evaluate each via full simulation and
    pick the best actual makespan.

    Strategies:
      A) Interleaved (stride-D) — known to work well for pipeline scheduling.
      B) Interleaved + random perturbations (1–5 random swaps).
      C) Random placements that satisfy the adjacency constraint.
      D) Constructive beam search (assign stages one-by-one to least-loaded device).

    Adjacency constraint: stage k and stage k±1 must be on different devices.

    Returns:
      If top_n == 1: (placement, pp_comp_times, stage_comp_times)
      If top_n > 1:  (placements_list, pp_comp_times_list, stage_comp_times)
    """
    import random as _rng

    num_pp = len(model_placement)
    num_stages = len(model_partition)
    if num_stages > 1 and num_pp < 2:
        raise ValueError("Need at least 2 devices when stage_num > 1.")

    prefix = [0]
    for p in model_partition:
        prefix.append(prefix[-1] + int(p))
    assert prefix[-1] == len(layer_comp_time), (
        f"Sum(partition) ({prefix[-1]}) must equal len(layer_comp_time) ({len(layer_comp_time)})"
    )

    stage_comp_times = []
    for k in range(num_stages):
        stage_comp_times.append(sum(layer_comp_time[prefix[k]:prefix[k + 1]]))
    stage_comp_times = [round(t, 2) for t in stage_comp_times]

    def _adj_ok(assigned):
        for k in range(1, num_stages):
            if assigned[k] == assigned[k - 1]:
                return False
        return True

    def _load(assigned):
        load = [0.0] * num_pp
        for sid, did in enumerate(assigned):
            load[did] += stage_comp_times[sid]
        return load

    def _variance(vals):
        m = sum(vals) / len(vals)
        return sum((v - m) ** 2 for v in vals) / len(vals)

    def _place_from_assigned(assigned):
        p = [[] for _ in range(num_pp)]
        for sid, did in enumerate(assigned):
            p[did].append(sid)
        for row in p:
            row.sort()
        return p

    # --- Candidate collection ---
    seen_sigs = set()
    all_candidates = []  # list of (assigned, load)

    def _add(assigned):
        sig = tuple(assigned)
        if sig in seen_sigs:
            return
        seen_sigs.add(sig)
        all_candidates.append((list(assigned), _load(assigned)))

    # Strategy A: Interleaved (stride-D)
    interleaved = [s % num_pp for s in range(num_stages)]
    if _adj_ok(interleaved):
        _add(interleaved)

    # Strategy B: Interleaved + random perturbations
    n_perturb = min(beam_width // 2, 300)
    for _ in range(n_perturb):
        a = interleaved[:]
        n_swaps = _rng.randint(1, min(5, num_stages // 2))
        for __ in range(n_swaps):
            s1, s2 = _rng.sample(range(num_stages), 2)
            a[s1], a[s2] = a[s2], a[s1]
        if _adj_ok(a):
            _add(a)

    # Strategy C: Random adjacency-satisfying placements
    n_random = min(beam_width // 2, 300)
    for _ in range(n_random):
        a = [-1] * num_stages
        ok = True
        for s in range(num_stages):
            devs = list(range(num_pp))
            _rng.shuffle(devs)
            placed = False
            for d in devs:
                if s > 0 and a[s - 1] == d:
                    continue
                a[s] = d
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and _adj_ok(a):
            _add(a)

    # Strategy D: Constructive beam search (assign in stage order to least-loaded device)
    cbeam = [([0.0] * num_pp, [-1] * num_stages)]
    for sid in range(num_stages):
        cost = stage_comp_times[sid]
        next_beam = []
        for load, assigned in cbeam:
            for d in range(num_pp):
                if sid > 0 and assigned[sid - 1] == d:
                    continue
                new_load = load[:]
                new_load[d] += cost
                new_assigned = assigned[:]
                new_assigned[sid] = d
                score = (_variance(new_load), max(new_load), new_load[d], d)
                next_beam.append((score, new_load, new_assigned))
        next_beam.sort(key=lambda x: x[0])
        cbeam = [(l, a) for _, l, a in next_beam[:min(32, beam_width // 4)]]
    for _, assigned in cbeam:
        if _adj_ok(assigned):
            _add(assigned)

    # Strategy E: Interleaved variants with different stride patterns
    for stride in range(1, num_pp + 1):
        a = [-1] * num_stages
        for s in range(num_stages):
            a[s] = (s * stride) % num_pp
        if _adj_ok(a):
            _add(a)
    # Reversed interleaved
    a_rev = [interleaved[num_stages - 1 - s] for s in range(num_stages)]
    if _adj_ok(a_rev):
        _add(a_rev)

    if not all_candidates:
        raise RuntimeError("No feasible placement found.")

    # --- Deduplicate by load signature and return top-N ---
    all_candidates.sort(key=lambda s: (_variance(s[1]), max(s[1])))

    top_results = []
    seen_load_sigs = set()
    for assigned, load in all_candidates:
        place = _place_from_assigned(assigned)
        if normalize_equal_cost_blocks:
            place = normalize_equal_cost_blocks_round_robin(place, stage_comp_times)
        top_results.append((place, list(load)))
        if len(top_results) >= top_n:
            break

    if top_n == 1:
        return top_results[0][0], top_results[0][1], stage_comp_times

    top_placements = [r[0] for r in top_results]
    top_pp_comp_times = [r[1] for r in top_results]
    return top_placements, top_pp_comp_times, stage_comp_times


def enforce_monotone_device_ids_by_stage_id(
    placement,
    stage_comp_times,
    pp_comp_times,
    *,
    tol: float = 1e-6,
):
    """
    Step-2 postprocess:
    Re-arrange stage assignment so that as stage id increases, its device id is non-decreasing,
    i.e. stages form contiguous blocks for device 0,1,...,D-1 in stage-id order.

    Constraint: per-device total comp time must remain unchanged (pp_comp_times).

    This is done by scanning stages in increasing stage id and cutting D contiguous blocks whose
    sums match the target per-device loads exactly (within tol).

    Returns:
      new_placement (List[List[int]])
    Raises:
      ValueError if exact matching is impossible under the monotone constraint.
    """
    num_pp = len(placement)
    num_stages = len(stage_comp_times)
    if len(pp_comp_times) != num_pp:
        raise ValueError("pp_comp_times length must equal number of devices")

    targets = [float(x) for x in pp_comp_times]
    stages = list(range(num_stages))

    new_placement = [[] for _ in range(num_pp)]
    idx = 0
    for did in range(num_pp):
        tgt = targets[did]
        s = 0.0
        # assign at least one stage if remaining stages == remaining devices
        while idx < num_stages and (s + stage_comp_times[idx] <= tgt + tol or (num_stages - idx) == (num_pp - did)):
            new_placement[did].append(idx)
            s += float(stage_comp_times[idx])
            idx += 1
            if abs(s - tgt) <= tol:
                break
        if abs(s - tgt) > tol:
            raise ValueError(
                f"Cannot match target load for device {did}: got {s}, target {tgt}. "
                f"Monotone-by-stage-id constraint makes it infeasible."
            )

    if idx != num_stages:
        raise ValueError("Not all stages were assigned under monotone constraint.")

    return new_placement


def normalize_equal_cost_blocks_round_robin(
    placement,
    stage_comp_times,
    *,
    tol: float = 1e-6,
):
    """
    Postprocess placement for the case you described:

    For any *consecutive* stage-id block where stage_comp_times are equal (within tol),
    enforce round-robin device assignment by stage id order:
      stage base+0 -> device 0
      stage base+1 -> device 1
      ...
      stage base+D-1 -> device D-1
    for each full chunk of size D inside that equal-cost block.

    This preserves per-device total comp time for those stages as long as within each
    full chunk we still assign exactly one stage to each device (true by construction),
    and the stages in the chunk have equal cost.

    Notes:
    - Only applies to full chunks of size D. Remainder (<D) stages at the end of a block
      are left as-is.
    - This DOES allow device ids to "wrap" (e.g., stage 3 on device 3, stage 4 on device 0),
      which matches your expected 0,1,2,3 pattern.
    """
    num_pp = len(placement)
    num_stages = len(stage_comp_times)

    # Build current stage->device mapping.
    stage_to_dev = [-1] * num_stages
    for did, sids in enumerate(placement):
        for sid in sids:
            stage_to_dev[sid] = did
    if any(d == -1 for d in stage_to_dev):
        raise ValueError("Placement does not cover all stages (stage_to_dev has -1).")

    # Find consecutive equal-cost blocks by stage id.
    blocks = []
    i = 0
    while i < num_stages:
        j = i + 1
        while j < num_stages and abs(float(stage_comp_times[j]) - float(stage_comp_times[i])) <= tol:
            j += 1
        blocks.append((i, j))  # [i, j)
        i = j

    # Reassign within each block, chunked by device count.
    for start, end in blocks:
        length = end - start
        full = (length // num_pp) * num_pp
        for base in range(start, start + full, num_pp):
            for offset in range(num_pp):
                sid = base + offset
                target_dev = offset
                cur_dev = stage_to_dev[sid]
                if cur_dev == target_dev:
                    continue
                # Move sid from cur_dev to target_dev (swap by moving).
                placement[cur_dev].remove(sid)
                placement[target_dev].append(sid)
                stage_to_dev[sid] = target_dev

    # Normalize: sort stage ids within each device.
    for row in placement:
        row.sort()
    return placement

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

if __name__ == "__main__":
    placement = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    partition = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    layer_comp_time = [12.971404966230345, 10.245337226926678, 27.587792641586727, 26.872286634312736, 27.45064462194539, 26.55433409442805, 26.50246579930036, 26.858190957946007, 26.875891235139633, 26.745782079720737, 26.764864541063403, 26.65297163345597, 27.35851009626581, 27.06289459177942, 26.178063635573245, 26.423338039354846, 26.7423149856052, 26.58582664288656, 26.144167055686317, 26.51176076224356, 26.575061039792168, 26.638643577243343, 26.065844745346993, 26.282502118084167, 27.730233389620828, 28.49078457584285, 27.202737509602247, 38.96261173787743]
    new_placement, pp_comp_times, stage_comp_times = solve_placement_min_pp_comp_time(partition, placement, layer_comp_time)
    print(new_placement)
    print(pp_comp_times)
    print(stage_comp_times)