---
type: concept
aliases: [VLABench Benchmark]
created: 2026-06-04
---

# VLABench

## 定义

一个评估 VLA 模型泛化能力的仿真基准，专注于 Out-of-Distribution（OOD）泛化，涵盖 5 类泛化场景。

## 核心要点

1. **5 类泛化轨道**:
   - **In-distribution (In-dist.)**: 训练分布内场景
   - **Cross-Category (Cross-Cat.)**: 跨物体类别迁移
   - **Commonsense**: 需要常识推理的间接指令
   - **Instruction**: 指令形式多样化泛化
   - **Texture**: 物体纹理变化泛化
2. **三项指标**: Success Rate (SR)、Progress Score (PS)、Intention Score (IS)
3. **设计理念**: 揭示 VLA 模型在真实世界部署中的泛化边界，而非仅评估任务完成率

## 代表工作（在此 Benchmark 上测试）

- [[ERVLA]]: Avg SR 53.2%，超越 π₀.₅（48.1%）
- [[Pi05]]: Avg SR 48.1%，Avg PS 62.3%

## 相关概念

- [[VLA]]
- [[LIBERO-Plus]]
