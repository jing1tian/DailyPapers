---
type: concept
aliases: [High-Fidelity FP8, 高保真 FP8 量化]
---

# HiF8

## 定义
OSP-Next 提出的一种自定义 8-bit 浮点量化格式，在标准 FP8（E4M3/E5M2）基础上引入混合指数位分配策略，针对视频 DiT 模型的激活分布优化量化误差。

## 数学形式

HiF8 混合格式：对不同层/激活区间动态选择 E4M3 或 E5M2 子格式，
$$x_{HiF8} = \arg\min_{f \in \{E4M3, E5M2\}} \|x - Q_f(x)\|_2$$

## 核心要点
1. 针对 DiT attention 和 FFN 激活值的尖峰分布设计
2. 与硬件标准 FP8 相比，需要自定义 kernel 支持，缺乏通用 GPU 加速
3. 仅在 OSP-Next 框架中使用，目前生态支持受限

## 代表工作
- [[OSP-Next]]: HiF8 的提出论文

## 相关概念
- [[PTQ]]
- [[QAT]]
- [[DiT]]
