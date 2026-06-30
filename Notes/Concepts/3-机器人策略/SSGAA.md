---
type: concept
aliases: [状态空间引导自适应注意力, State-Space Guided Adaptive Attention, SSGAA]
---

# SSGAA（状态空间引导自适应注意力）

## 定义

SSGAA 是 S²-VLA 提出的核心模块，通过动态信念状态生成软门控权重，自适应融合视觉、语义意图和动作序列三条并行注意力分支的输出，实现 VLA 对长时序任务阶段的相位感知。

## 数学形式

门控权重生成：

$$
[g_\text{vis}^{(l)},\, g_\text{ite}^{(l)},\, g_\text{act}^{(l)}]^\top = \text{Softmax}\!\left(\text{MLP}_g^{(l)}(\mathbf{b}_t^{(l)})\right)
$$

多分支融合：

$$
\mathbf{H}^{(l)} = g_\text{vis}^{(l)} \cdot \mathbf{O}_\text{vis}^{(l)} + g_\text{ite}^{(l)} \cdot \mathbf{O}_\text{ite}^{(l)} + g_\text{act}^{(l)} \cdot \mathbf{O}_\text{act}^{(l)}
$$

## 核心要点

1. **三分支并行**: 低层视觉交叉注意力（精确定位）、高层意图交叉注意力（语义规划）、动作序列自注意力（时序一致性）
2. **信念状态驱动**: 门控权重由 [[信念状态]] 动态生成，而非固定权重
3. **单层最优**: 插入 Transformer 中间层（第 12 层）效果最优；多层门控反而引入不稳定性

## 代表工作

- [[S2-VLA]]: SSGAA 的提出论文，在 LIBERO 长时序任务上以 2B 参数超越 7B 模型

## 相关概念

- [[信念状态]]
- [[Cross-Attention|交叉注意力]]
- [[GRU]]
- [[VLA（视觉-语言-动作模型）]]
