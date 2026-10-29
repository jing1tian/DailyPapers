---
type: concept
aliases: [RoboCasa GR1, RoboCasa-GR1 Benchmark]
---

# RoboCasa-GR1

## 定义

RoboCasa-GR1 是基于 [[RoboCasa]] 仿真环境构建的**人形机器人操作**评测基准，使用 GR1 双臂人形机器人执行厨房场景操作任务，提供少样本和全数据两种评测设置，以及域外泛化（OOD）测试。

## 核心要点

1. **机器人平台**: GR1 双臂人形机器人（Fourier Intelligence 傅利叶智能）
2. **任务场景**: 厨房操作任务（抓取、摆放、开关抽屉等）
3. **评测设置**:
   - **少样本（Few-shot）**: 数据量受限场景，测试数据高效学习能力
   - **全数据（Full-data）**: 完整演示数据训练
   - **OOD 泛化**: 分布外场景的泛化成功率
4. **基准难度**: 比 LIBERO 更具挑战性，强调少样本和 OOD 能力

## 代表工作

- [[TraceWAM]]: 少样本 63.6% / 全数据 80.3%，OOD 平均 58.6%，是截至 2026-10 无预训练方法的 SOTA
- [[LAWA]]: 带 ego-centric 增强的全数据 80.8%（有预训练）
- GR00T N1.6：全数据 47.6%

## 相关概念

- [[RoboCasa]]: 底层仿真环境
- [[LIBERO]]: 另一主要操作评测基准
- [[World Action Model]]: 在此基准上评测的主要方法类型
