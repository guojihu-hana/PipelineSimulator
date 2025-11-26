from enum import Enum
from enum import IntEnum

class RecomputeType(Enum):
    FULL = 1
    SELECTIVE = 2

# class WorkloadType(Enum):
#     F = "F"
#     B = "B"
#     W = "W"
#     R = "R"

class WorkloadType(IntEnum):
    F = 1
    B = 2
    W = 3
    R = 4

class Placement(Enum):
    WAVELIKE = 1
    VSHAPE = 2
    INTERLEAVED = 3
    NAIVE = 4
    RECURRENT = 5
    CROSS = 6
    SEARCHED = 7

class Schedule(Enum):
    GREEDY_v1 = 1
    INTERLEAVED = 2
    ONE_F_ONE_B = 3
    ZBV = 4
    ZBH1 = 5
    BFW = 6
    GREEDY_v2 = 7
    Layerwise = 8
    Chimera = 9
    OctoPipe = 15
    Mist = 16
    ReCycle = 17
    
    STANDARD_1F1B = 10
    STANDARD_INTERLEAVED = 11
    STANDARD_ZBH = 12
    STANDARD_ZBV = 13
    STANDARD_AFAB = 14

class StageSearchOrder(Enum):
    Random = "Random"
    IncDec = "IncDec"

class RunMode(Enum):
    SEARCH_SCHEDULE = "search_schedule"
    Z3_SOLVE = "z3"
    GUROBI_SOLVE = "gurobi"
    SIM_SOLVE = "sim"
    LAYERWISE_GUROBI_SOLVE = "layer"
    CHIMERA = "chimera"

import heapq

class OrderedQueue:
    def __init__(self, type_order):
        """
        type_order: list，例如 [WorkloadType.F, WorkloadType.B, WorkloadType.W]
        """
        self._heap = []
        self._counter = 0
        self.last_type_order = type_order
        self.type_priority = {t: i for i, t in enumerate(type_order)}

    def _key(self, workload):
        type_rank = self.type_priority.get(workload.wtype, 999)
        return (type_rank, workload.mid, workload.sid, self._counter)

    def push(self, workload):
        key = self._key(workload)
        heapq.heappush(self._heap, (key, workload))
        self._counter += 1

    def pop(self):
        if not self._heap:
            return None
        _, workload = heapq.heappop(self._heap)
        return workload

    def peek(self):
        if not self._heap:
            return None
        return self._heap[0][1]

    def __len__(self):
        return len(self._heap)

    def set_type_order(self, type_order):
        """
        动态更新 type 排序顺序并重排堆
        """
        if type_order == self.last_type_order:
            return
        self.last_type_order = type_order
        self.type_priority = {t: i for i, t in enumerate(type_order)}
        # 重新构造堆
        new_heap = []
        for _, workload in self._heap:
            key = self._key(workload)
            new_heap.append((key, workload))
        heapq.heapify(new_heap)
        self._heap = new_heap

class WorkloadConstraint:

    def __init__(self, 
                 device_id: int,
                 microbatch_id: int, 
                 stage_id: int, 
                 workload_type: WorkloadType,
                ) -> None:
        self.did = device_id
        self.mid: int = microbatch_id  # 微批次编号
        self.sid: int = stage_id              # 阶段编号
        self.workload_type: WorkloadType = workload_type  # 工作负载类型

        self._hash = hash((self.mid, self.sid, self.workload_type))

    def __eq__(self, other):
        if not isinstance(other, WorkloadConstraint):
            return NotImplemented
        return (
            self.mid == other.mid 
            and self.sid == other.sid
            and self.workload_type == other.workload_type
        )
    
    def __hash__(self):
        return self._hash
    
    def __repr__(self):
        return (f"did={self.did}, mid={self.mid}, sid={self.sid}, wtype={self.workload_type.name})")