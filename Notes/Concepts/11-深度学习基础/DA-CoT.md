---
type: concept
aliases: [Depth-Aware Chain-of-Thought, 深度感知思维链]
---

# DA-CoT (Depth-Aware Chain-of-Thought)

## 定义
在语言和 flow-time conditioning 下对几何信息进行结构化、非自回归推理的模块，用于 VLA 的空间几何感知。

## 数学形式
$$\mathbf{r} = \text{DA-CoT}(f_\text{geom}, l, t) \quad \text{non-autoregressive}$$

## 核心要点
1. 非自回归（parallel）执行几何推理，避免 CoT 的串行推理开销
2. 以深度特征、语言指令和 flow-time 共同条件化
3. 输出结构化几何推理结果用于 action prediction

## 代表工作
- [[GaussVLA]]: DA-CoT 作为几何推理模块

## 相关概念
- [[GST]]
- [[GaussVLA]]
