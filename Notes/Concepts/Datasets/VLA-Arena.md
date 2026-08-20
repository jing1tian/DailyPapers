---
type: concept
aliases: [VLA Arena, VLA-Arena Benchmark]
---

# VLA-Arena

## 定义

一个专门评估 [[Vision-Language-Action Model|VLA]] 模型在**多维度挑战场景**下泛化能力的仿真基准，涵盖安全性、干扰物鲁棒性、外推能力和长时序规划四个维度。

## 核心要点

1. **四个评估维度**:
   - **Safety（安全性）**: 模型能否在危险场景中避免不安全动作
   - **Distractors（干扰物）**: 面对与任务无关的干扰物体时的鲁棒性
   - **Extrapolation（外推）**: 对训练分布外的物体/场景的泛化
   - **Long Horizon（长时序）**: 需要多步规划的复杂任务完成率
2. **与 LIBERO 的区别**: LIBERO 侧重基础操控技能；VLA-Arena 侧重鲁棒性和复杂场景泛化，更接近真实部署需求。
3. **LoopVLA 结果**: 48.7% 平均成功率，在长时序维度（76.0% vs 59.0%）上大幅超越 Qwen3OFT 基线，体现循环精炼对复杂规划的优势。

## 相关概念

- [[Vision-Language-Action Model]]
- [[LIBERO]]
- [[LoopVLA]]
