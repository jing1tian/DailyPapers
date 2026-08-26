---
type: concept
aliases: [typed spatial readout, spatial grounding interface]
---

# Typed Spatial Grounding

## 定义
为 VLA 模型的空间 grounding 定义类型约束接口（如 point-in-box、point-in-mask 等），将空间读出从自回归文本坐标分离出来，使 grounding 接口更稳定、可验证。

## 核心要点
1. 传统 VLA 用文本坐标（如"(0.3, 0.7)"）表示空间位置，分布迁移时脆弱
2. Typed readout 强制类型约束，减少 hallucination 和格式错误
3. 接口与动作执行解耦，更易模块化替换

## 代表工作
- [[Pointing-VLA]]: 在 Embodied-R1 上构建 typed hidden-state spatial readout

## 相关概念
- [[Spatial Grounding]]
- [[VLA]]
- [[RoboPoint]]
