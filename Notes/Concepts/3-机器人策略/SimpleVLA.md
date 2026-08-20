---
type: concept
aliases: [Simple VLA]
---

# SimpleVLA

## 定义
一种简化的 VLA 基线模型，通常用于强化学习微调实验中作为对照，提供不含复杂结构的 VLA 参考实现。

## 核心要点
1. 轻量化 VLA，去除冗余模块，聚焦基础 perception-action 管线
2. 在 GRPO/RL 微调研究中常作为对比基线（如 Prism-GRPO）
3. 代码通常开源，便于复现

## 代表工作
- [[Prism-GRPO]] (2608.17423): 以 SimpleVLA 作为对比基线评估 Prism-GRPO 的加速效果

## 相关概念
- [[VLA]]
- [[OpenVLA]]
- [[SmolVLA]]
