---
type: concept
aliases: [Mamba, Selective SSM, S6]
---

# Mamba

## 定义
在 [[SSM]] 基础上引入输入依赖选择机制（Selective State Space）的序列模型，通过动态调整状态矩阵参数，使模型能有选择地记忆或遗忘历史信息，同时保持线性计算复杂度。

## 数学形式

Mamba 将固定矩阵 $A, B, C$ 替换为输入 $x_t$ 的函数：

$$
B_t = \text{Linear}(x_t), \quad C_t = \text{Linear}(x_t), \quad \Delta_t = \text{Softplus}(\text{Linear}(x_t))
$$

## 核心要点
1. 硬件感知的并行扫描算法（Parallel Scan），推理时线性时间，训练时可并行
2. 相比 S4 等传统 SSM，选择机制使其在语言建模任务上接近 Transformer 性能
3. 常与 Transformer block 交叉使用（Interleaved Mamba），在视频生成中效果良好

## 代表工作
- [[CoME]]: 论文将 Transformer+Interleaved Mamba 作为长程记忆 baseline 之一

## 相关概念
- [[SSM]]
- [[Transformer]]
