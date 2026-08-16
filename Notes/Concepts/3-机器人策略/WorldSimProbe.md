---
type: concept
aliases: [World Simulator Probe, Simulator Faithfulness Benchmark]
---

# WorldSimProbe

## 定义
一个诊断 Action-Conditioned World Model（ACWM）仿真器保真度的 benchmark，测试 world model 是否精确响应动作条件（而不只是视觉质量好）。

## 核心要点
1. **Observable Simulator Contract**：定义 ACWM 应满足的动作-运动对应契约
2. **多维度评测**：可行运动覆盖率（action realization）、接触语义保真度（interaction-primitive fidelity）、OOD 泛化
3. 测试平台：RoboTwin、ManiSkill、LIBERO
4. 关键发现：Cosmos-3 在 false-interaction grounding 上显著失效（视觉好但物理错）

## 核心洞见
视觉质量高 ≠ 物理保真度高。用 ACWM 做 data augmentation 或 planning 之前，必须检验其动作响应的准确性。

## 代表工作
- WorldSimProbe: Diagnosing Simulator Faithfulness in Action-Conditioned World Models (2026)

## 相关概念
- [[RoboTwin]]
- [[Ctrl-World]]
- [[DreamDojo]]
