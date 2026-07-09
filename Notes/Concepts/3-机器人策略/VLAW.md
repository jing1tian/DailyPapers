---
type: concept
aliases: [Vision-Language-Action-World, VLAW modeling]
---

# VLAW (Vision-Language-Action-World)

## 定义
将视觉感知、语言理解、动作生成、世界预测四个目标统一在单一模型框架下的机器人学习范式。

## 核心要点
1. 在 VLA（Vision-Language-Action）基础上增加 World 维度，模型同时做动作预测和世界状态预测
2. World 预测作为辅助目标，提供物理先验和未来规划能力
3. 四个目标可通过交织序列建模（interleaved modeling）在单一自回归框架实现

## 代表工作
- [[WorldBagel]]：基于 BAGEL 的 VLAW 框架，引入 FFAD/FFAT 做连续动作编解码
- [[InternVLA-A1.5]]：通过 MoT 架构统一理解、预见（foresight）、动作三个目标

## 相关概念
- [[VLA]]
- [[World Model]]
- [[Latent Foresight]]
