---
type: concept
aliases: [AR Forcing, AR-Forcing]
---

# Autoregressive Forcing

## 定义
修复扩散式世界模型训练-推理 distribution shift 的方法：训练时引入自回归强制，让模型见过自己的预测输出作为输入。

## 核心要点
1. 1. 训练时并行监督 vs 推理时自回归展开导致分布偏移
2. 2. 类似 scheduled sampling，在训练中注入自身预测
3. 3. 在导航 WM 中验证长程稳定性提升

## 代表工作
- (待补充)

## 相关概念
- [[World Model]]
- [[DiT]]
