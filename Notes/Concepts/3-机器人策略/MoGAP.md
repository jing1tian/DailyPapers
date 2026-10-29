---
type: concept
aliases: [Motion-Guided Asymmetric Prediction]
---

# MoGAP

## 定义
（2026，arXiv 2610.36624，"Adapting Video Models for Robot Control via Motion-guided Asymmetric Prediction"）提出的视觉预测任务设计方案，用 optical flow 区分运动显著区域与背景区域，前者高精度预测，后者低精度预测，聚焦控制相关表示学习。

## 核心要点
1. 引入区域不对称预测：运动区域高精度，背景区域低精度
2. 用 optical flow 自动区分两类区域，无需人工标注
3. 与 Trace-WAM 同方向（运动相关区域重点监督），但实现路径不同

## 代表工作
- 原始论文（2026），基于 GR-1 和 RoboCasa 评估

## 相关概念
- [[World Model]]
- [[Trace-WAM]]
- [[GR-1]]
