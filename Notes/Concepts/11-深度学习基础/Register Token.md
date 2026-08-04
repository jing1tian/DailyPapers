---
type: concept
aliases: [注册 token, Register Tokens, 寄存器 token]
---

# Register Token

## 定义

Register Token（注册 token）是 Vision Transformer 中额外插入的若干可学习 token，不与任何图像块或输入位置对应，专门用于在注意力计算中"吸收"全局统计信息，从而避免 artifact（高激活伪影）并提升表示质量。

## 数学形式

在标准 ViT 序列 $[x_\text{cls}, x_1, \ldots, x_N]$ 中插入 $r$ 个注册 token：

$$
[\underbrace{x_\text{cls}}_{1}, \underbrace{r_1, \ldots, r_R}_{\text{register tokens}}, \underbrace{x_1, \ldots, x_N}_{N}]
$$

注册 token 参与全局注意力，输出时丢弃，仅保留图像 patch token 或 CLS token 的表示。

## 核心要点

1. **吸收全局信息**: 注意力中的"平均池化"行为会在某些 token 上积聚全局统计量，导致局部高激活伪影；注册 token 提供专用"槽位"承接此类信息。
2. **改善 patch token 质量**: 释放普通 patch token 的表示能力，使其更专注于局部语义，改善密集预测任务（如分割）中的特征质量。
3. **DiT 中的稳定作用**: 在扩散变换器（DiT）主干中，注册 token 有助于稳定训练动态和注意力分布。

## 代表工作

- [[WorldDiT]]: 44 层 DiT 主干中使用 4 个注册 token 稳定注意力计算
- Vision Transformers Need Registers (Darcet et al., 2023): 原始提出论文

## 相关概念

- [[DiT]]
- [[Vision Transformer]]
- [[Transformer]]
