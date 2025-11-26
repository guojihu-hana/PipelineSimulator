import heapq
import random
from typing import List, Tuple, Dict
import math

def greedy_lpt_assign(times: List[float],
                      mems: List[float],
                      D: int,
                      mem_limits: List[float]) -> Tuple[List[List[int]], List[float], List[float]]:
    """
    L 个 layer 无序可重排 情况下的初始分配
    times and mems 长度为 L
    D 为机器数
    mem_limits 长度为 D
    返回 assignments machine -> list of layer indices, loads per machine, mem usage per machine
    """
    L = len(times)
    assert L == len(mems)
    assert D == len(mem_limits)

    # 初始化每台机器的当前计算负载和显存使用
    loads = [0.0] * D
    mem_used = [0.0] * D
    assigns: List[List[int]] = [[] for _ in range(D)]

    # 按时间从大到小排序 layer
    idxs = sorted(range(L), key=lambda i: times[i], reverse=True)

    # 使用一个最小堆按 (load, machine_id) 存放机器
    heap = [(0.0, i) for i in range(D)]
    heapq.heapify(heap)

    for i in idxs:
        t = times[i]
        m = mems[i]

        placed = False
        # 尝试从最轻负载开始找到第一个显存可放下的机器
        # 为避免破坏堆顺序, 我们弹出若干项到临时列表, 找到后再推回
        temp = []
        while heap:
            load, mid = heapq.heappop(heap)
            if mem_used[mid] + m <= mem_limits[mid]:
                # 可放
                assigns[mid].append(i)
                loads[mid] += t
                mem_used[mid] += m
                heapq.heappush(heap, (loads[mid], mid))
                placed = True
                break
            else:
                temp.append((load, mid))
        # 推回临时保存的机器
        for item in temp:
            heapq.heappush(heap, item)

        if not placed:
            # 没找到显存足够的机器, 尝试强制放到某台仍能容纳的机器
            # 这一步实际与上一步等价, 但放在这里是为了清晰处理不可行情况
            feasible = [mid for mid in range(D) if mem_used[mid] + m <= mem_limits[mid]]
            if not feasible:
                raise ValueError(f"No machine can fit layer {i} with mem {m}")
            # 在可行机器中选择当前负载最小的
            best = min(feasible, key=lambda mid: loads[mid])
            assigns[best].append(i)
            loads[best] += t
            mem_used[best] += m
            # 需要刷新堆中的该机器项, 最简单方式是重建堆
            heap = [(loads[j], j) for j in range(D)]
            heapq.heapify(heap)

    return assigns, loads, mem_used


def compute_variance(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def local_search_improve(assigns: List[List[int]],
                         times: List[float],
                         mems: List[float],
                         mem_limits: List[float],
                         max_iters: int = 500) -> Tuple[List[List[int]], List[float], List[float]]:
    """
    在初始解上进行局部改进
    两种尝试:
      1) 单向移动 一个 layer 从机器 a 移到机器 b
      2) 交换 机器 a 的 layer i 和 机器 b 的 layer j
    接受任何能降低负载方差且满足显存约束的改动
    """
    D = len(assigns)
    loads = [sum(times[idx] for idx in assigns[d]) for d in range(D)]
    mem_used = [sum(mems[idx] for idx in assigns[d]) for d in range(D)]

    best_var = compute_variance(loads)
    improved = True
    it = 0

    # 为了速度, 随机化遍历顺序
    rng = random.Random(12345)

    while improved and it < max_iters:
        improved = False
        it += 1

        # 先尝试单向移动
        machine_order = list(range(D))
        rng.shuffle(machine_order)
        for a in machine_order:
            if not assigns[a]:
                continue
            # 对 a 的 layer 尝试移动到别的机器
            layer_indices = assigns[a][:]
            rng.shuffle(layer_indices)
            for layer in layer_indices:
                t = times[layer]
                m = mems[layer]
                # 尝试目标机器
                for b in machine_order:
                    if b == a:
                        continue
                    if mem_used[b] + m > mem_limits[b]:
                        continue
                    # 进行移动后的负载
                    new_loads = loads[:]
                    new_loads[a] -= t
                    new_loads[b] += t
                    new_var = compute_variance(new_loads)
                    if new_var + 1e-12 < best_var:
                        # 接受移动
                        assigns[a].remove(layer)
                        assigns[b].append(layer)
                        loads = new_loads
                        mem_used[a] -= m
                        mem_used[b] += m
                        best_var = new_var
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # 再尝试交换
        machine_pairs = [(i, j) for i in range(D) for j in range(i+1, D)]
        rng.shuffle(machine_pairs)
        for a, b in machine_pairs:
            if not assigns[a] or not assigns[b]:
                continue
            # 随机挑若干候选减少计算
            candidates_a = assigns[a][:min(6, len(assigns[a]))]
            candidates_b = assigns[b][:min(6, len(assigns[b]))]
            rng.shuffle(candidates_a)
            rng.shuffle(candidates_b)
            done_swap = False
            for la in candidates_a:
                for lb in candidates_b:
                    ta, ma = times[la], mems[la]
                    tb, mb = times[lb], mems[lb]
                    # 交换后的内存使用
                    new_mem_a = mem_used[a] - ma + mb
                    new_mem_b = mem_used[b] - mb + ma
                    if new_mem_a > mem_limits[a] or new_mem_b > mem_limits[b]:
                        continue
                    new_loads = loads[:]
                    new_loads[a] = loads[a] - ta + tb
                    new_loads[b] = loads[b] - tb + ta
                    new_var = compute_variance(new_loads)
                    if new_var + 1e-12 < best_var:
                        # 接受交换
                        assigns[a].remove(la)
                        assigns[a].append(lb)
                        assigns[b].remove(lb)
                        assigns[b].append(la)
                        loads = new_loads
                        mem_used[a] = new_mem_a
                        mem_used[b] = new_mem_b
                        best_var = new_var
                        improved = True
                        done_swap = True
                        break
                if done_swap:
                    break
            if done_swap:
                break

    return assigns, loads, mem_used


def solve_unordered(times: List[float],
                    mems: List[float],
                    D: int,
                    mem_limits: List[float],
                    max_iters: int = 500) -> Dict:
    """
    主函数包装
    返回结果字典 包括 assignments loads mem_used variance
    """
    assigns, loads, mem_used = greedy_lpt_assign(times, mems, D, mem_limits)
    assigns, loads, mem_used = local_search_improve(assigns, times, mems, mem_limits, max_iters=max_iters)
    var = compute_variance(loads)
    assigns = sort_lists(assigns)
    return {
        "assignments": assigns,
        "loads": loads,
        "mem_used": mem_used,
        "variance": var
    }

def sort_lists(lists:list):
    return sorted([sorted(arr) for arr in lists], key=lambda x : x[0])

if __name__ == "__main__":
    random.seed(1)
    L = 120
    D = 16
    times = [random.uniform(1, 100) for _ in range(L)]
    mems = [random.uniform(1, 10) for _ in range(L)]
    mem_limits = [144 for _ in range(D)]

    result = solve_unordered(times, mems, D, mem_limits, max_iters=800)
    
    for d in range(D):
        print(f"machine {d}: layers {result['assignments'][d]} load {result['loads'][d]:.2f} mem {result['mem_used'][d]:.2f}/{mem_limits[d]:.2f}")
    print("Final variance:", result["variance"])
