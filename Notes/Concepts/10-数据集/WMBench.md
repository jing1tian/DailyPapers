---
type: concept
aliases: [World Model Benchmark, WMBench]
---

# WMBench（世界模型评估基准）

## 定义
GigaWorld-1 提出的评估框架，专门用于评估世界模型（WM）作为机器人策略评估器的可靠性，衡量 WM 是否能准确预测策略的成功率，从而替代昂贵的真实机器人评估。

## 核心要点
1. 核心指标：WMES（World Model Evaluation Score）——测量 WM 预测成功率与真实成功率的相关性
2. 与 FVD（Fréchet Video Distance）的区别：FVD 只衡量视觉保真度，WMES 直接衡量策略评估准确性
3. 强调物理真实性（physics fidelity）比视觉保真度更重要
4. 数据来源：GigaData（真实机器人数据）+ PhysData（物理模拟数据）

## 代表工作
- [[GigaWorld-1]]：提出 WMBench 框架和 WMES 指标，系统研究 WM 作为 policy evaluator 的条件

## 相关概念
- [[World Model]]
- [[GigaBrain]]
- [[FVD]]
