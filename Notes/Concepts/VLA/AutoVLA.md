---
type: concept
aliases: []
---

# AutoVLA

## 定义
面向自动驾驶的端到端 Vision-Language-Action 模型，在单一自回归框架内统一场景推理（CoT 文本）和轨迹动作生成，常作为驾驶类 VLA baseline 出现。

## 核心要点
1. 自回归地交替输出语言推理 token 与轨迹动作 token，推理与决策共享同一模型
2. 常被驾驶 VLA 后续工作（如对比仿真/真实数据对齐的方法）拿来做 baseline
3. 与纯模块化（感知-预测-规划分离）方案相比，强调端到端可解释性

## 代表工作
- 待补充

## 相关概念
- [[OpenVLA]]
