---
type: concept
aliases: [Experience Backtracking, World Model Backtracking]
---

# Rollback

## 定义
Luo 等（2026，arXiv 2610.36068，"Rollback: Experience Backtracking in World Models for Policy Improvement"）提出的世界模型回溯机制，在执行中途检测到潜在失败信号时回退到历史安全检查点重新尝试，而非从失败推进到头再重置。

## 核心要点
1. 世界模型维护状态历史栈
2. 当 value/reward 估计下降超过阈值时触发回溯
3. 从最近"安全检查点"重新展开执行

## 代表工作
- Luo et al. (2026): 原始论文，含真实机器人实验

## 相关概念
- [[World Model]]
- [[RISE]]
