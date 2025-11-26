## 🚀 AdaPtis：针对异构模型的自适应流水线并行

流水线并行是大模型训练的重要并行策略之一。例如当模型大小超过单卡（NPU、TPU、GPU等计算卡）显存容量时，需要将模型切分放置在不同的卡上才能进行训练，此时就可以采用流水线并行。

## 🧩 Pipeline Parallelism：流水线并行
流水线并行可以分为三个部分
1. **模型切分（Model Partition）**

   * 决定模型被切分为几个块（Stage），每块由一系列连续的模型层（Layer）组成。
2. **模型放置（Model Placement）**

   * 决定每一块应放置在哪个 GPU 上，例如顺序的放置模式下，第一块放置在第一个 GPU，第二块放置在第二个 GPU，以此类推，第 N 块放置在第 N 个 GPU 上。
3. **任务调度（Workload Scheduling）**

   * 决定每个 Stage 上所有 Micro-batch 应该进行的前向计算（Forward Pass）、反向计算（Backward Pass）的计算顺序。例如 4 个 Micro-batch，用 F1 表示第一个 Micro-batch 的前向计算，Stage 1 上的计算顺序可以为 F1-F2-F3-F4-B1-B2-B3-B4。

在模型切分和放置在指定的 GPU 之后，每个 GPU 按照预先指定好的任务调度方式开始进行 Micro-batch 对应的前向和反向计算。不同的 GPU 之间通过 P2P 通信来传输中间结果，例如 Stage 1 包含了第 1、2 层 Layer，被放置在了 GPU 1 上，Stage 2 包含了第 3、4 层 Layer，被放置在了 GPU 2 上，那么任意 Micro-batch N，的前向计算需要先在 GPU 1 上进行，然后通过 P2P 通信将结果传输给 GPU 2 进行下一步的前向计算，直到 N 在所有的 Layer 上都进行了前向计算，则可以开始进行反向计算。

理想情况下，我们期望流水线中的所有 GPU 尽可能处于繁忙状态。然而现实却是由于模型被切分放置在不同的 GPU 上，所以天然存在设备空闲时间，即设备等待前序设备进行中间结果的传递才能进行下一步的计算，也叫做 **Pipeline Bubble（流水线气泡）**。针对流水线并行的优化需要尽可能减少 Pipeline Bubble。

## 🧩 Heterogeneous Models：异构模型
理想情况下，一个全由 Transformer Layer 构成的模型可以被均匀的切分，此时空泡最少。直观理解是流水线的执行时间取决于流水线中最慢的设备，而每一个设备上分配到的 Layer 的数量决定了计算时间的下限，假设每个 Layer 的计算时间相同，则为每个设备分配相同数量的 Layer 可以最小化流水线的执行时间下限。然而当模型存在异构时，模型难以均匀切分，比如：
1. **Embedding | Head Layer :**
模型一般都会在第一层之前加入一层 Embedding Layer，在最后一层之后加入一层 Head Layer。而这两种 Layer 往往和 Transformer Layer 的计算时间不同。
2. **MLP Layer :**
一些模型中存在不同的 MLP 层，比如 Transformer Layer 中的 MLP 层是 FFN，而在稀疏模型中，FFN 被替换为 MoE。若一个模型在不同的 Layer 中存在不同类型的 MLP Layer，则这些层的计算时间不同，例如 DeepSeek 的前 k 层是 FFN，其余层是 MoE。
3. **Attention Layer :**
传统的 Transformer 模型使用 Self-Attention。最近的一些新模型，开始混用 Linear-Attention 和 Self-Attention，例如 Qwen-Next、Nemotron-H、Minimax-01。当不同的层使用了不同的注意力机制时，其计算时间往往也会出现差异，例如线性注意力机制在长序列情况下的计算时间要比自注意力机制短得多。

当模型中出现各种各样的结构或机制时，模型就变得不再均匀。对于非均匀的模型而言，传统的流水线方法难以处理产生的 Pipeline Bubble。

## 💡 为什么传统流水线做不到？

目前主流的大模型训练系统（比如 Megatron-LM、DeepSpeed）都采用 **固定的流水线并行策略**：

* 每个 GPU 负责一段模型；
* 每个阶段用固定的调度（如 1F1B）去跑微批次。

这在模型结构统一的情况下性能尚可。但随着模型架构越来越趋向异构，有的 Stage 太慢，成为其他 Stage 的 Stragglers。
最终导致训练性能的下降，我们观察到传统方法在异构模型上可以出现最大 **40%**
的 Pipeline Bubble。

现有方法只针对 Pipeline 的某一部分进行优化（如 ZB 针对 Workload Scheduling、Mist 针对 Model Partition、Interleaved-1F1B 针对 Model Placement），在异构模型上有时候会产生负优化效果，即使能够在一定程度上缓解 Bubble 问题，却仍然存在的大量的可优化空间。这是因为 Pipeline 的三个部分互相影响，单独优化某一部分时无法兼顾其他两个部分，难以找到性能更佳的组合。

---

## ⚙️ AdaPtis 的关键思路：自适应流水线并行

根据上述，解决异构模型上的 Pipeline Bubble 问题需要
1. **Adaptive and Co-optimized :** 能够自适应调整自身策略的流水线而非对模型架构无感的、一成不变的流水线。
2. **Search Efficient :** 能够高效求解出针对流水线三部分进行 Co-optimize 的算法，因为 Co-optimization 的搜索空间巨大。
3. **Flexible :** 能够高效执行各种流水线策略的灵活的执行引擎。

---

## 🧩 Adaptis 架构

### 1. Pipeline Performance Model

精准的流水线性能模型，能预测每种流水线策略的：

* 计算时间
* 通信延迟
* 显存占用
* 气泡比例
* 计算-通信重叠情况

### 2. Pipeline Generator

聪明的搜索引擎。
它不会暴力枚举，而是采用**逐阶段优化（Phase-by-Phase Tuning）**：
识别瓶颈 → 调整对应阶段 → 验证 → 回滚 → 继续搜索。
像个懂经验的老工程师。

### 3. Unified Pipeline Executor

统一执行层。
能在不同拓扑、不同通信模式下灵活运行，
自动实现计算与通信重叠（overlap），
并防止通信死锁。

---

## 📈 实验结果：

在多个真实模型上（Gemma、DeepSeek、Nemotron-H 等），AdaPtis 相对于 相比 Megatron-LM 的 Interleaved-1F1B 方法实现了：

* 平均加速 **1.42×**
* 最高加速 **2.14×**
* 并且在 8 → 128 GPU 的 Scaling 实现中，效率提升 **530%**