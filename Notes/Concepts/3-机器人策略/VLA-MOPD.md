---
type: concept
aliases: [Multi-task On-Policy Distillation for VLA]
---

# VLA-MOPD

## 定义
用于多任务 VLA 策略合并的在线策略蒸馏方法，解决多专项 VLA 合并为通用策略时的灾难性遗忘问题。

## 核心要点
1. 先用 SFT 收集各专项 VLA 数据做初始化
2. On-Policy Distillation：学生在线采样，同时接受多 teacher 软标签监督
3. 比纯 behavior cloning 在 distribution shift 下更稳定
4. 在 [[LIBERO]] benchmark 上验证，有真实实验

## 代表工作
- [[VLA-MOPD]]: 原始论文（USTC/Central South Univ，2611.12286）

## 相关概念
- [[VLA]]
- [[LIBERO]]
- [[ODE]]
