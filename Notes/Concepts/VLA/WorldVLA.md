---
type: concept
aliases: [World Model VLA, WorldVLA]
---

# WorldVLA

## 定义
将 world model 与 VLA 策略深度集成的框架，通过在动作推理前运行世界模型前向预测，验证候选动作序列的安全性和合理性，再执行通过验证的动作。

## 核心要点
1. 动作生成模块（VLA）和世界模型验证模块相互配合
2. 世界模型预测未来状态，用于 OOD 检测和安全验证
3. Pre-VLA 使用 WorldVLA 框架做 runtime verification 评估
4. 是 WAM（World Action Model）范式的具体实现之一

## 代表工作
- [[WorldVLA]]：被 Pre-VLA 作为参考框架引用
- [[Pre-VLA]]：Preemptive Runtime Verification，使用 WorldVLA 评估

## 相关概念
- [[VLA]]
- [[World Model]]
- [[World Action Model]]
