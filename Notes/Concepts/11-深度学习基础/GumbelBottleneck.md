---
type: concept
aliases: [Gumbel Bottleneck, Write-Protected Discrete Bottleneck, 写保护离散瓶颈]
---

# GumbelBottleneck（写保护离散瓶颈）

## 定义
GumbelBottleneck 是一种 discrete variational bottleneck 架构，使用 Gumbel-Softmax 进行离散化，同时施加"写保护"约束：语言/VLM 的梯度只允许读取（查询）物理符号表示，而不能修改（写入）它们，从而防止语言梯度覆写物理动力学表示。

## 数学形式
$$z = \text{GumbelSoftmax}(e; \tau), \quad \nabla_{z} \mathcal{L}_{lang} \perp \nabla_{z} \mathcal{L}_{phys}$$

即物理损失和语言损失的梯度在 bottleneck 处被隔离，物理表示只通过物理损失更新。

## 核心要点
1. 动机：端到端语言注入（RT-2、Octo 范式）会破坏 world model 的物理符号表示
2. "写保护" = language 梯度不通过 bottleneck 反向传播到物理表示
3. 双引擎（Dual-Engine）架构：语言引擎和物理引擎共享 bottleneck 但梯度隔离
4. 与 Frozen Bottleneck 的区别：frozen 完全冻结，write-protected 只阻断语言方向的梯度

## 代表工作
- [[WP-WM]]: Write-Protected Discrete Bottlenecks for Language-Grounded World Models（2607.08312）——实证证明 Gumbel-Softmax 在语言梯度下面临结构性坍缩困境，提出写保护三层修复方案

## 相关概念
- [[V-JEPA]]
- [[VQ-VAE]]
- [[RT-2]]
