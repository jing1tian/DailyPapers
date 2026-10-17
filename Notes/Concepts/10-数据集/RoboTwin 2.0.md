---
type: concept
aliases: [RoboTwin2, RoboTwin 2.0 Benchmark]
---

# RoboTwin 2.0

## 定义
一个双臂机器人操作仿真基准（benchmark），提供 Clean（标准场景）和 Random（场景随机化）两种评估设置，用于测试策略的鲁棒性和泛化能力。

## 核心要点
1. **双臂操作**: 专注于需要双臂协作的精细操作任务（如 mug、switch、seal、mouse、scan 等）
2. **双场景设置**:
   - **Clean**: 标准固定场景，评估基本成功率
   - **Random**: 场景随机化（物体位置、外观等），评估泛化能力
3. **Clean2Random**: 训练集仅用 Clean 数据，测试时在 Random 场景评估，考察 OOD 泛化

## 代表工作
- [[VeriVLA]]: 在 5 个精细操作任务上评估，VeriVLA 将 Clean 成功率从 66.4% 提升至 73.6%，Clean2Random 从 62.8% 提升至 69.6%

## 相关概念
- [[LIBERO]]
- [[Sim-to-Real Gap]]
- [[域随机化]]
