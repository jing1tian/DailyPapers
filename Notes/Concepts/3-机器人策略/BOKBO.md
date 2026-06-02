---
type: concept
aliases: [BOKBO, calibrated abstention]
---

# Best of K Bad Options

## 定义
VLA 推理时安全弃权方法：当 K 个候选动作全部不安全时，选择放弃执行而非强行选一个坏的。

## 核心要点
1. 1. 针对 test-time scaling (K 采样) 的安全补充
2. 2. 用分位数回归（CRC）做无分布弃权校准
3. 3. Coverage/abstain 率可调，分布自由保证

## 代表工作
- [[BOKBO (paper)]]

## 相关概念
- [[RoboMonkey]]
- [[VLA]]
