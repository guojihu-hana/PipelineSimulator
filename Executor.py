from simulator.abstract.Pipeline import PipelineScheduler
from simulator.abstract.mutils import TrainingConfig, dict_to_2d_list
from simulator.abstract.Device import Device
from simulator.abstract.variables import Schedule, WorkloadType
from simulator.abstract.context import global_context as gpc
from simulator.painter import MultiPipelinePainter as MPP
from tuning import (
    time_line, time_recorder_decorator,
    check_placement_convergence, check_partition_convergence,
    balanced_transpose, serialize,
    tune_placement, tune_partition_random, tune_placement_random,
    tune_partition_dfs, tune_placement_dfs, tune_partition,
    solve_placement_min_pp_comp_time,
    fast_octopipe_estimate,
    solve_placement_guided,
)
import random
import math
import heapq


class Executor:

    def __init__(self, schedule_method, tc: TrainingConfig, model_name: str = None, nmb_per_dp: list = None, device_comp_power: list[list] = None, device_mem: list[list] = None) -> None:
        self.time           = 0
        self.schedule_method = schedule_method
        self.model_name     = model_name,
        self.tc             = tc
        self.bwd_split      = tc.bwd_split
        self.finish_flag    = False
        self.dp_size        = tc.dp_size
        self.chunk_num      = tc.chunk_num
        self.device_num     = tc.device_num
        self.layer_num      = tc.layer_num
        self.stage_num      = self.chunk_num * self.device_num
        self.device_comp_power  = device_comp_power if device_comp_power else [[gpc["COMP_POWER"] for _ in range(self.device_num)] for _ in range(self.dp_size)]
        self.device_mem         = device_mem if device_mem else [[gpc["GPU_MAX_MEM"] for _ in range(self.device_num)] for _ in range(self.dp_size)]
        self.nmb_per_dp     = [tc.micro_batch_num for _ in range(self.dp_size)] if not nmb_per_dp else nmb_per_dp
        self.mid_offsets    = [0] + [sum(self.nmb_per_dp[:i]) for i in range(1, self.dp_size+1)]
        self.micro_batch_size = tc.micro_batch_size
        self.batch_size = sum(self.nmb_per_dp) * self.micro_batch_size
        self.best_pipelines = None

    def _init_pipelines(self, placement = None, partition=None):
        self.pipelines      = [
            PipelineScheduler(
                pipeline_idx=dp_idx,
                schedule_method=self.schedule_method,
                training_config=self.tc,
                placement=placement,
                partition=partition,
                mid_offset=self.mid_offsets[dp_idx],
                comp_power=self.device_comp_power[dp_idx],
                max_mem=self.device_mem[dp_idx],
                executor=self
            ) for dp_idx in range(self.dp_size)
        ]

    def update_time(self):
        self.time += 1

    def reset_time(self):
        self.time = 0

    def get_time(self):
        return self.time

    def get_total_workload_count(self):
        count = 0
        for pipeline in self.pipelines:
            count += pipeline.total_workload
        return count

    def update_constraints_across_dp(self, time):
        for pipeline in self.pipelines:
            for device in pipeline.devices:
                if device.current_workload and time >= device.current_workload.end_time:
                    finished_mid = device.current_workload.mid
                    for p in self.pipelines:
                        for d in p.devices:
                            if d.did == device.did or p.pid == pipeline.pid:
                                continue # only update constraints on other devices or pipelines
                            if finished_mid in d.held_mids:
                                d.update_constraints_within_device(time, constraint=device.current_workload)

    def partition_to_placement(self, partition):
        placement = []
        start = 0
        for size in partition:
            end = start + size
            placement.append(list(range(start, end)))
            start = end
        return placement

    def get_layer_comp_times(self, vocab_parallel=False):
        layer_comp_time = [f + b + w if not gpc["RECOMP"] else f + f + b + w for f, b, w in zip(self.tc.layer_f_times, self.tc.layer_b_times, self.tc.layer_w_times)]
        if vocab_parallel:
            layer_comp_time = layer_comp_time[:-1]
        return  layer_comp_time

    @time_recorder_decorator(time_line)
    def one_step_tuning(self, time_limit, placement=None, partition=None, verbose=False):
        self._init_pipelines(placement=placement, partition=partition)
        self.reset_time()
        self.finish_flag = False
        while self.get_time() <= time_limit and not self.finish_flag:
            finish_count = 0
            for pipeline in self.pipelines:
                pipeline.check_workload_status(time=self.time)
                self.update_constraints_across_dp(time=self.time)
                pipeline.execute_workload(time=self.time)
                pipeline.check_device_status(time=self.time)
                finish_count += pipeline.num_finished_microbatch
            self.finish_flag = True if finish_count == self.get_total_workload_count() else False
            self.update_time()

        if verbose:
            print(f"Time: {self.get_time()}, Finish: {self.finish_flag}")
            for pipeline in self.pipelines:
                pipeline.print_device_utilization(self.get_time())

        for pipeline in self.pipelines:
            pipeline.save_partition()

    def _simulate_placement(self, placement, partition, time_limit):
        """Run one simulation and return the makespan (max device finish time)."""
        self._init_pipelines(placement=placement, partition=partition)
        self.reset_time()
        self.finish_flag = False
        while self.get_time() <= time_limit and not self.finish_flag:
            fc = 0
            for pipeline in self.pipelines:
                pipeline.check_workload_status(time=self.time)
                self.update_constraints_across_dp(time=self.time)
                pipeline.execute_workload(time=self.time)
                pipeline.check_device_status(time=self.time)
                fc += pipeline.num_finished_microbatch
            self.finish_flag = fc == self.get_total_workload_count()
            self.update_time()
        return max(max(p.get_device_execution_time()) for p in self.pipelines)

    def iterative_tuning(self, iter_limit=100, time_limit=gpc["TIME_LIMIT"],
                         placement=None, partition=None, verbose=False,
                         beam_width=1024, top_n=200, local_search_rounds=3,
                         sim_k=64, ls_neighbor_cap=512):
        """
        Three-phase placement optimiser for OctoPipe scheduling (single-threaded).

        Phase 1 — Fast estimator-guided candidate generation
        Phase 2 — Simulation evaluation of top-``sim_k`` candidates
        Phase 3 — ILS with two-tier local search (fast estimator pre-filter
                  then real simulation), perturbation kicks, and random
                  restarts.

        ``ls_neighbor_cap`` limits how many swap/transfer neighbours are
        scored with the fast estimator per local-search round (large models
        can have >1k neighbours); beyond the cap, a random subset is used.
        Pass ``None`` for no cap (slowest, exhaustive neighbour scan).
        """
        import time as _time

        placement = [[lid for lid in range(did * self.layer_num // self.device_num,
                       (did + 1) * self.layer_num // self.device_num)]
                      for did in range(self.device_num)]
        partition = [1 for lid in range(self.layer_num)]
        num_stages = len(partition)
        num_pp = self.device_num
        num_mb = self.tc.micro_batch_num

        layer_f = list(self.tc.layer_f_times)
        layer_b = list(self.tc.layer_b_times)

        # ----------------------------------------------------------
        # Phase 1: fast estimator-guided candidate generation
        # ----------------------------------------------------------
        t0 = _time.time()
        guided_plas, est_times, stage_f, stage_b = solve_placement_guided(
            partition, placement, layer_f, layer_b,
            num_mb=num_mb, beam_width=beam_width, top_n=top_n,
        )
        print(f"[Phase 1] {len(guided_plas)} candidates, "
              f"best est={est_times[0]}  ({_time.time()-t0:.1f}s)")

        # ----------------------------------------------------------
        # Phase 2: evaluate top candidates with real simulator
        # ----------------------------------------------------------
        t0 = _time.time()
        sim_k = min(sim_k, len(guided_plas))
        eval_results = []
        for pla in guided_plas[:sim_k]:
            t = self._simulate_placement(pla, partition, time_limit)
            eval_results.append((t, pla))
        eval_results.sort(key=lambda x: x[0])

        best_time = eval_results[0][0]
        best_placement = [row[:] for row in eval_results[0][1]]
        print(f"[Phase 2] top-{sim_k} → best T={best_time}  "
              f"({_time.time()-t0:.1f}s)")

        # Adaptive simulation cap: no need to fully simulate placements
        # that are clearly worse than the current best.
        sim_cap = int(best_time * 2)

        # ----------------------------------------------------------
        # Phase 3: ILS
        # ----------------------------------------------------------
        t0 = _time.time()
        # One tuple conversion for the hot fast-estimator loop (Python path).
        _sf = tuple(int(x) for x in stage_f)
        _sb = tuple(int(x) for x in stage_b)

        def _a_from_p(pla):
            a = [-1] * num_stages
            for d, row in enumerate(pla):
                for s in row:
                    a[s] = d
            return a

        def _p_from_a(a):
            p = [[] for _ in range(num_pp)]
            for s, d in enumerate(a):
                p[d].append(s)
            return p

        def _adj_ok(a):
            return all(a[k] != a[k - 1] for k in range(1, num_stages))

        def _random_adj():
            a = [-1] * num_stages
            for s in range(num_stages):
                devs = list(range(num_pp))
                random.shuffle(devs)
                for d in devs:
                    if s > 0 and a[s - 1] == d:
                        continue
                    a[s] = d
                    break
            return a

        def _perturb(a, n):
            b = a[:]
            for _ in range(n * 3):
                s1, s2 = random.sample(range(num_stages), 2)
                b[s1], b[s2] = b[s2], b[s1]
                if _adj_ok(b):
                    n -= 1
                    if n <= 0:
                        return b
                else:
                    b[s1], b[s2] = b[s2], b[s1]
            return b if _adj_ok(b) else a

        def _sim(a):
            """Simulate with adaptive cap."""
            return self._simulate_placement(_p_from_a(a), partition, sim_cap)

        def _fe(a):
            return fast_octopipe_estimate(a, _sf, _sb, num_mb, num_pp)

        def _local_search(start_a, start_t, rounds=local_search_rounds):
            cur_a, cur_t = start_a, start_t
            for _ in range(rounds):
                nbr_assigns = []

                for s1 in range(num_stages):
                    for s2 in range(s1 + 1, num_stages):
                        if cur_a[s1] == cur_a[s2]:
                            continue
                        na = cur_a[:]
                        na[s1], na[s2] = na[s2], na[s1]
                        if not _adj_ok(na):
                            continue
                        nbr_assigns.append(na)

                for s in range(num_stages):
                    for d in range(num_pp):
                        if d == cur_a[s]:
                            continue
                        na = cur_a[:]
                        na[s] = d
                        if not _adj_ok(na):
                            continue
                        nbr_assigns.append(na)

                if not nbr_assigns:
                    break

                if (ls_neighbor_cap is not None
                        and len(nbr_assigns) > ls_neighbor_cap):
                    nbr_assigns = random.sample(nbr_assigns, ls_neighbor_cap)

                estimates = [_fe(a) for a in nbr_assigns]
                top_k = max(3, min(15, len(estimates) // 10))
                top_idx = heapq.nsmallest(top_k, range(len(estimates)),
                                          key=lambda i: estimates[i])

                best_found_t = cur_t
                best_found_a = None
                for i in top_idx:
                    t = self._simulate_placement(
                        _p_from_a(nbr_assigns[i]), partition, sim_cap)
                    if t < best_found_t:
                        best_found_t = t
                        best_found_a = nbr_assigns[i]

                if best_found_a is None:
                    break
                cur_a, cur_t = best_found_a, best_found_t

            return cur_a, cur_t

        # ILS from best Phase-2 candidate
        ils_a = _a_from_p(best_placement)
        ils_a, ils_t = _local_search(ils_a, best_time)
        if ils_t < best_time:
            best_time = ils_t
            best_placement = _p_from_a(ils_a)
            sim_cap = int(best_time * 2)
        print(f"[Phase 3] After local search: T={best_time}")

        # Perturbation kicks
        no_improve = 0
        for kick in range(local_search_rounds * 5):
            n = random.randint(3, max(5, num_stages // 4))
            pa = _perturb(ils_a, n)
            pt = _sim(pa)
            na, nt = _local_search(pa, pt)
            if nt < best_time:
                best_time = nt
                best_placement = _p_from_a(na)
                ils_a, ils_t = na, nt
                sim_cap = int(best_time * 2)
                no_improve = 0
                print(f"[Phase 3] Kick {kick}: T={best_time}")
            else:
                no_improve += 1
                if no_improve >= 5:
                    break

        # Random restarts from different basins
        for ri in range(min(5, local_search_rounds)):
            ra = _random_adj()
            rt = _sim(ra)
            oa, ot = _local_search(ra, rt)
            if ot < best_time:
                best_time = ot
                best_placement = _p_from_a(oa)
                sim_cap = int(best_time * 2)
                print(f"[Phase 3] Random {ri}: T={best_time}")
            for kick in range(3):
                n = random.randint(3, max(5, num_stages // 4))
                pa = _perturb(oa, n)
                pt = _sim(pa)
                na, nt = _local_search(pa, pt)
                if nt < best_time:
                    best_time = nt
                    best_placement = _p_from_a(na)
                    sim_cap = int(best_time * 2)
                    print(f"[Phase 3] Random {ri} kick {kick}: "
                          f"T={best_time}")

        print(f"[Tuning] Final best T={best_time}  "
              f"({_time.time()-t0:.1f}s)")

        # Re-run with best placement to save artifacts.
        if best_placement is not None:
            self.one_step_tuning(time_limit=time_limit, placement=best_placement,
                                 partition=partition, verbose=verbose)

    def iterative_tuning_old(self, iter_limit=100, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=True, show_success=True, show_partition=True, show_placement=True, tune_strategy=0, placement=None, partition=None):
        it = 0
        placement = [[lid for lid in range(did * self.layer_num//self.device_num, (did + 1) * self.layer_num//self.device_num)] for did in range(self.device_num)]
        partition = [1 for lid in range(self.layer_num)]

        history = []
        partition_history = []
        placement_history = []
        best_time = 999999999999
        best_partition = None
        layer_comp_time = self.get_layer_comp_times(vocab_parallel=self.tc.vocab_parallel)
        window_size = 10
        partition_tuning_stage = 0
        partition_check_window = window_size
        worst_time = -1
        while it < iter_limit:
            self.one_step_tuning(time_limit=time_limit, placement=placement, partition=partition)
            if self.finish_flag:
                for pipeline in self.pipelines:
                    device_execution_times = pipeline.get_device_execution_time()
                    device_bubble_times = pipeline.get_device_bubble_time()
                    model_partition = pipeline.get_model_partition()
                    model_placement = self.partition_to_placement(model_partition)
                    pipeline_finish_time = max(device_execution_times)
                    worst_time = max(worst_time, pipeline_finish_time)
                    history.append((it, pipeline_finish_time, model_partition, model_placement))
                    partition_history.append(tuple(model_partition))
                    if best_time > pipeline_finish_time:
                        best_time = pipeline_finish_time
                        best_partition = model_partition
                    print(f"Iteration:{it}, Time:{pipeline_finish_time}, Bubbles:{round(sum(device_bubble_times)/pipeline_finish_time/len(device_bubble_times), 2):.2f}, Partition:{best_partition}, Tried:{model_partition}")
                    if tune_strategy == 0:
                        model_partition = tune_partition(best_partition, model_placement, layer_comp_time, device_execution_times, device_bubble_times, history=partition_history, stage=partition_tuning_stage)
                    elif tune_strategy == 1:
                        model_partition = tune_partition_random(best_partition, model_placement, layer_comp_time, device_execution_times, device_bubble_times, history=partition_history, stage=partition_tuning_stage)
                    elif tune_strategy == 2:
                        model_partition = tune_partition_dfs(best_partition, model_placement, layer_comp_time, device_execution_times, device_bubble_times, history=partition_history, stage=partition_tuning_stage)
            else:
                print("Fail")

            it += 1
            converged = False
            if model_partition is None:
                converged = True

            converged, _, _ = check_partition_convergence(history, window=partition_check_window)

            if converged:
                if partition_tuning_stage == 1:
                    break
                elif partition_tuning_stage == 0:
                    partition_tuning_stage += 1
                    partition_check_window += window_size

        best_placement = self.partition_to_placement(partition=best_partition)
        placement = balanced_transpose(placement=placement, layer_comp_time=layer_comp_time)
        partition = [1 for _ in range(self.layer_num)]
        while it < iter_limit:

            self.one_step_tuning(time_limit=time_limit, placement=placement, partition=partition)

            if self.finish_flag:
                for pipeline in self.pipelines:
                    device_execution_times = pipeline.get_device_execution_time()
                    device_bubble_times = pipeline.get_device_bubble_time()
                    model_partition = pipeline.get_model_partition()
                    model_placement = pipeline.get_model_placement()
                    if self.tc.vocab_parallel:
                        for row in model_placement:
                            row.pop()
                    pipeline_finish_time = max(device_execution_times)
                    worst_time = max(worst_time, pipeline_finish_time)
                    history.append((it, pipeline_finish_time, model_partition, model_placement))
                    placement_history.append(serialize(model_placement))
                    if best_time > pipeline_finish_time:
                        best_time = pipeline_finish_time
                        best_placement = model_placement
                    # print(f"Iteration:{it}, Time:{pipeline_finish_time}, Bubbles:{round(sum(device_bubble_times)/pipeline_finish_time/len(device_bubble_times), 2):.2f}, Placement:{best_placement}, Tried:{model_placement}")
                    print(f"Iteration:{it}, Time:{pipeline_finish_time}, Bubbles:{round(sum(device_bubble_times)/pipeline_finish_time/len(device_bubble_times), 2):.2f}")
                    if tune_strategy == 0:
                        placement = tune_placement(best_placement, partition, layer_comp_time, device_execution_times, device_bubble_times, history=placement_history)
                    elif tune_strategy == 1:
                        placement = tune_placement_random(best_placement, partition, layer_comp_time, device_execution_times, device_bubble_times, history=placement_history)
                    elif tune_strategy == 2:
                        placement = tune_placement_dfs(best_placement, partition, layer_comp_time, device_execution_times, device_bubble_times, history=placement_history)
            else:
                print("Fail")

            converged = check_placement_convergence(placement_history=placement_history)
            if converged:
                break
            it += 1

        best_placement = sorted(best_placement, key=lambda x : x[0])
        best_placement = [[lid for lid in range(did, self.layer_num, self.device_num)] for did in range(self.device_num)]
        print(best_placement)
        self.one_step_tuning(time_limit=time_limit, placement=best_placement, partition=partition)
        for pipeline in self.pipelines:
            pipeline.print_partition_placement()
        print(f"Time {worst_time} -> {best_time}")

    def run_all_dp(self, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=False, show_success=True, show_partition=False, show_placement=False):
        self.reset_time()
        partition = [7, 8, 8, 8, 8, 8, 8, 1]
        placement = [[0], [1], [2], [3], [4], [5], [6], [7]]
        self._init_pipelines(partition=partition,placement=placement)
        workloads = {}
        while self.get_time() <= time_limit and not self.finish_flag:
            finish_count = 0
            for pipeline in self.pipelines:
                pipeline.check_workload_status(time=self.time)
                self.update_constraints_across_dp(time=self.time)
                pipeline.execute_workload(time=self.time)
                pipeline.check_device_status(time=self.time)
                finish_count += pipeline.num_finished_microbatch
                if self.dp_size > 1 and self.schedule_method in (Schedule.OctoPipe, Schedule.ReCycle):
                    if self.get_time() == 0:
                        if pipeline.pid == 0:
                            # Pop all
                            # workloads = pipeline.pop_workload(mid_group=list(range(pipeline.mid_offset, pipeline.mid_offset + pipeline.nmb)), did_group=[2])
                            # Pop partial
                            workloads = pipeline.pop_workload(mid_group=list(range(pipeline.mid_offset, pipeline.mid_offset + pipeline.nmb - 3)), did_group=[2])
                    if self.get_time() == 0:
                        if pipeline.pid == 1:
                            pipeline.insert_workload(workloads=workloads,did_group=[2])
            self.finish_flag = True if finish_count == self.get_total_workload_count() else False
            self.update_time()
        if show_success:
            if self.finish_flag:
                if show_partition and show_placement:
                    for pipeline in self.pipelines:
                        pipeline.print_partition_placement()
                print("Success")
            else:
                print("Fail")
        if show_utilization:
            for pipeline in self.pipelines:
                pipeline.print_device_utilization(self.get_time())

    def draw(self) -> None:
        res_all_dp = {}
        res_all_dp["res"]={}
        res_all_dp["painter_conf"]={}
        all_dp_f_times = {}
        all_dp_b_times = {}
        all_dp_w_times = {}
        for dp_idx in range(self.dp_size):
            pipeline = self.pipelines[dp_idx]
            f_times, b_times, w_times = pipeline.get_workloadload_duration()
            for sid in f_times:
                if sid not in all_dp_f_times:
                    all_dp_f_times[sid] = {}
                    all_dp_b_times[sid] = {}
                    all_dp_w_times[sid] = {}

                for mid in f_times[sid]:
                    all_dp_f_times[sid][mid] = f_times[sid][mid]
                    all_dp_b_times[sid][mid] = b_times[sid][mid]
                    all_dp_w_times[sid][mid] = w_times[sid][mid]

            res = {}
            for key in pipeline.results:
                if key.startswith(("f_","b_","w_","r_")):
                    res[key] = pipeline.results[key]
            painter_conf = {
                "device_num": pipeline.device_num,
                "devices": pipeline.placement,
                "stage_num": pipeline.stage_num,
                "pp_height": gpc["PP_HEIGHT"],
                "pp_align": gpc["PP_ALIGN"],
                "pixel_base": gpc["PIXEL_BASE"],
                "nmb": pipeline.nmb,
                "mid_offset": pipeline.mid_offset,
                "comm_length": [gpc["COMM_TIME"] for _ in range(pipeline.stage_num)],
            }
            res_all_dp["res"][dp_idx]=res
            res_all_dp["painter_conf"][dp_idx]=painter_conf
        for dp_idx in range(self.dp_size):
            res_all_dp["painter_conf"][dp_idx]["f_times"] = dict_to_2d_list(all_dp_f_times)
            res_all_dp["painter_conf"][dp_idx]["b_times"] = dict_to_2d_list(all_dp_b_times)
            res_all_dp["painter_conf"][dp_idx]["w_times"] = dict_to_2d_list(all_dp_w_times)
        MPP(res_all_dp["painter_conf"]).draw(res_all_dp["res"])
