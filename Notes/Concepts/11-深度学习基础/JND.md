---
type: concept
aliases: [Just Noticeable Difference, 知觉可察觉差异]
---

# JND（Just Noticeable Difference）

## 定义
感知心理学概念：人类视觉系统能察觉的最小刺激差异阈值。在 token 压缩中，JND 被借用来衡量哪些视觉 token 变化对任务输出"不可感知"，从而安全丢弃。

## 核心要点
1. 来自人类感知心理学，原用于图像质量评估
2. 在 VLA 的 token 压缩中：若 token 变化量低于 JND 阈值，则动作输出基本不变
3. 比固定压缩率更自适应：根据内容难度动态调整保留 token 数

## 代表工作
- [[JND-VLA]]: 首次将 JND 建模用于 VLA token 压缩，在不牺牲动作精度的前提下降低推理延迟

## 相关概念
- [[VLA]]
- [[KV]]-cache
