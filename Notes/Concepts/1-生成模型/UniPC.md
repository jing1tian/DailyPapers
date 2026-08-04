---
type: concept
aliases: [UniPC, Unified Predictor-Corrector]
---

# UniPC

## 定义
扩散模型的高效 ODE 求解器，采用统一的预测-校正（predictor-corrector）框架，实现高阶精度与低 NFE 的平衡。

## 核心要点
1. 统一了多种高阶 ODE solver 的表达形式
2. 支持无训练加速，可即插即用于已有扩散模型
3. 常用于 WAM/Flow Matching 推理加速

## 代表工作
- [[FBFM]]: 用 UniPC 加速 WAM 推理

## 相关概念
- [[DPM-Solver]]
- [[FlowMatching]]
- [[NFE]]
