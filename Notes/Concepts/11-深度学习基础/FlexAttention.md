---
type: concept
aliases: [FlexAttention, flex_attention, 灵活注意力]
---

# FlexAttention

## 定义

PyTorch 提供的可编程注意力算子，允许用户通过自定义 `score_mod` / `mask_mod` 函数灵活定义任意注意力掩码与分数修改逻辑，同时保留接近手写 CUDA kernel（如 [[FlashAttention]]）的内存与计算效率，避免为每种新掩码模式单独编写算子。

## 核心要点

1. **可编程掩码**：通过传入掩码函数（而非显式构造 $N\times N$ 掩码矩阵）实现因果掩码、滑动窗口、块状掩码、文档掩码等任意自定义注意力模式
2. **保持 FlashAttention 级别效率**：底层基于分块计算与算子融合（torch.compile），避免显式实例化大型掩码矩阵带来的显存开销
3. **适合复杂训练范式的掩码需求**：当训练范式需要混合多种掩码模式时（如同时支持干净上下文全可见 + 含噪目标因果可见），比手写定制 kernel 更易扩展
4. **与 MagiAttention 等同类工具的关系**：均属于"自定义掩码注意力算子"这一类，可互相替代，具体选择取决于工程实现与性能需求

## 代表工作

- FlexAttention (PyTorch team): PyTorch 官方提供的可编程注意力实现
- [[Causal-rCM]]: 提出用 FlexAttention 或 MagiAttention 等自定义掩码算子实现 Teacher-Forcing 打包因果前向所需的专用注意力掩码 $\bm{M}_{\mathrm{TF}}$，作为替代两段式实现（后者与[[激活检查点]]兼容性差、显存占用更高）的方案

## 相关概念

- [[FlashAttention]]
- [[激活检查点]]
- [[KV 缓存]]
