---
type: concept
aliases: [视频专家, Video Branch, 视频分支]
---

# Video Expert（视频专家）

## 定义

Video Expert 是 [[Mixture-of-Transformers|MoT-based WAM]] 中专门处理视觉观测和语言指令的 Transformer 专家分支，仅执行一次前向以产生逐块 Key-Value 缓存，供动作专家跨注意力复用，实现视频表征与动作生成的解耦计算。

## 数学形式

视频专家前向输出（逐块 KV 缓存）：

$$
z(o, l) = \{(K_i^v, V_i^v)\}_{i=1}^{N}
$$

其中 $o$ 为视觉观测，$l$ 为语言指令，$N$ 为 Transformer 块数。动作专家的跨注意力：

$$
\text{CrossAttn}_a^{(i)} = \text{softmax}\left(\frac{Q_a^{(i)} {K_i^v}^\top}{\sqrt{d}}\right) V_i^v
$$

## 核心要点

1. **单次执行复用**：视频专家对每个控制周期只执行一次，动作专家的多步去噪全程复用同一 KV 缓存（蒸馏后单步只需一次）。
2. **MoT 架构核心**：与 Shared-Backbone WAM 相比，视频专家与动作专家参数分离，视频 KV 只需计算一次，避免重复的视频前向传播。
3. **冻结策略**：在蒸馏微调（如 [[TACD]]）时通常冻结视频专家，仅微调动作专家，节省显存（65.98 → 24.73 GiB）并稳定训练。
4. **CEWP 的利用**：[[CEWP]] 利用视频 KV 的逐块产出时序，将动作专家 QKV 投影与视频专家计算并行，实现流水线加速。

## 代表工作

- [[Fast-WAM]]：MoT-based WAM 的代表，视频专家 + 动作专家解耦设计
- [[Faster-WAM]]：更强的 MoT WAM
- [[RealtimeWAM]]：冻结视频专家进行 TACD 蒸馏 + CEWP 异步流水线

## 相关概念

- [[Mixture-of-Transformers]]
- [[World Action Model]]
- [[Action Chunking]]
- [[CEWP]]
