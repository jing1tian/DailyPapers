---
type: concept
aliases: [VLN Continuous Environment, 视觉语言导航连续环境, Vision-and-Language Navigation in Continuous Environments]
---

# VLN-CE（视觉语言导航连续环境）

## 定义

在连续 3D 环境（基于 Habitat 仿真器）中执行视觉语言导航的 benchmark，包含 R2R（Room-to-Room）和 RxR（Room-Across-Room）两个子集。

## 核心要点

1. **R2R**: 根据自然语言指令从起点导航至目标房间，使用 SR、SPL、NE 评估
2. **RxR**: 多语言版本，强调路径形状相似性，使用 SR、SPL、nDTW 评估
3. 连续动作空间（非离散图导航），更贴近真实导航

## 评估指标

$$
\text{SPL} = \frac{1}{N}\sum_{i=1}^N S_i \cdot \frac{l_i}{\max(p_i, l_i)}, \quad \text{nDTW} = \exp\left(-\frac{\text{DTW}(P,R)}{n \cdot d_t}\right)
$$

## 代表工作

- [[GPT6Astra-EmbodiedEval]]: GPT-6-Astra R2R 10× SR 88%，RxR 10× SR 90%

## 相关概念

- [[LHPR-VLN]]
- [[Embodied AI|具身智能]]
