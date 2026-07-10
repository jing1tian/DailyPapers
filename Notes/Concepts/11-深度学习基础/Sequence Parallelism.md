---
type: concept
aliases: [序列并行, Ulysses SP, DeepSpeed Ulysses, SP]
---

# Sequence Parallelism（序列并行）

## 定义
在多 GPU 上沿 sequence 维度切分长序列 token，使单 GPU 内存仅需承载全序列的 $1/N$ 分，通过 all-to-all 通信在注意力计算时交换 token 分片与 head 分片，实现超长上下文训练而无需单卡存放全序列。

## 核心要点
1. **DeepSpeed Ulysses 机制**：Transformer 块前将打包序列沿 token 维度分片；注意力计算前 all-to-all 将 token 分片转置为 head 分片；计算后再 all-to-all 恢复 token 分片
2. **序列并行 + 专家并行组合**：视频扩散 MoE 需要同时处理 1M+ token 序列和 100+ 专家的 token 路由，二者可在不同并行维度正交组合
3. **与 DP/FSDP 组合**：4D 并行（DP + FSDP + SP + EP）允许各维度独立配置
4. **应用场景**：对视频扩散尤为重要——480p × 多帧视频序列在 float16 下轻松超过单卡内存上限

## 代表工作
- DeepSpeed Ulysses (Jacobs et al., arXiv 2309.14509)
- [[LingBot-Video]]: 训练基础设施中集成 Ulysses SP 以支持 1M+ token 视频序列

## 相关概念
- [[FSDP2]]
- [[Sparse MoE]]
- [[DiT]]
