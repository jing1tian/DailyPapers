---
type: concept
aliases: [Habitat Sim, habitat-sim]
---

# Habitat-Sim

## 定义
Meta AI Research 开发的高性能 3D 仿真平台，专为 embodied AI 任务（导航、QA、操作）设计。

## 核心要点
1. 支持真实扫描场景（HM3D、ReplicaCAD 等），视觉保真度高
2. 渲染速度快（headless GPU），适合大规模 RL 训练
3. 与 [[SoundSpaces]] 结合支持多感知（视觉 + 音频 + 嗅觉）
4. Python API 对接 Gym 接口，常见的导航/操作 baseline 平台

## 代表工作
- [[OdorSpaces]]: 在 Habitat-Sim 上增加 physics-based olfaction 仿真

## 相关概念
- [[SoundSpaces]]
- [[AI2-THOR]]
- [[OmniGibson]]
