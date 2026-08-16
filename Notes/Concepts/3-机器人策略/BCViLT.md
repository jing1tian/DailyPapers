---
type: concept
aliases: [BCViLT, Behavior Cloning ViLT]
---

# BCViLT

## 定义
基于 ViLT（Vision-and-Language Transformer）做行为克隆的机器人策略，直接从图像-语言对映射到连续动作。

## 核心要点
1. ViLT 将图像 patch 和语言 token 统一处理，无需独立 visual encoder
2. 在 LIBERO benchmark 上作为轻量 baseline 被 EMS 等方法比较
3. 轻量但对复杂任务泛化能力有限

## 相关概念
- [[ACT]]
- [[OpenVLA]]
- [[EMS]]
