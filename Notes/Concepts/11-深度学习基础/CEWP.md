---
type: concept
aliases: [Cross-Expert Wavefront Pipelining, 跨专家波前流水线, 异步推理]
---

# CEWP（Cross-Expert Wavefront Pipelining）

## 定义

CEWP 是一种针对 [[Mixture-of-Transformers|MoT-based WAM]] 的异步推理优化技术，将视频专家与动作专家的块级执行从串行改为波前流水线并行，仅在动作注意力需要视频 KV 时才同步，消除跨专家串行等待开销，且不改变任何参数或输出。由 [[RealtimeWAM]] 提出。

## 数学形式

每个 Transformer Block $i$ 分解为三个算子：

$$
\text{Block}_i = P_i \rightarrow \text{Attn}_i \rightarrow R_i
$$

其中 $P_i$（QKV 投影）、$\text{Attn}_i$（注意力，动作侧需要视频 KV）、$R_i$（FFN 等其余计算）。

**依赖关系**：仅 $\text{Attn}_a^{(i)}$ 需要等待视频 KV $(K_i^v, V_i^v)$，$P_a^{(i)}$ 完全独立。

## 核心要点

1. **细粒度依赖分析**：不是等整个视频专家完成，而是逐块分析，找到最小等待粒度（仅注意力前同步）。
2. **CUDA 双流实现**：视频流（Stream_v）执行完 $P_v^{(i)}$ 后记录 CUDA Event $e_i$；动作流（Stream_a）立即执行 $P_a^{(i)}$，仅在 $\text{Attn}_a^{(i)}$ 前 wait on $e_i$。
3. **完全等价性**：仅重排执行顺序，不改变任何参数、输入、掩码、算子，数学输出与顺序执行完全一致（严格证明）。
4. **无需重新训练**：等价性保证使 CEWP 可作为后处理优化直接应用于已训练的 MoT-WAM。
5. **与 TACD 叠加**：TACD（1步蒸馏）+ CEWP（异步流水线）+ CUDA Graph 组合，Fast-WAM 从 299.7 ms → 12.2 ms（约 25×）。

## 代表工作

- [[RealtimeWAM]]（Lv et al., 2026）：提出 CEWP，首次将块级异步流水线应用于 MoT-WAM

## 相关概念

- [[CUDA Stream]]
- [[Mixture-of-Transformers]]
- [[TACD]]
- [[RealtimeWAM]]
- [[Fast-WAM]]
