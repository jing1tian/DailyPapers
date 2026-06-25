---
type: concept
aliases: [Composable 3D Value Maps]
---

# VoxPoser

## 定义
用大语言模型 + 视觉语言模型从自然语言指令中提取约束，组合生成 3D voxel affordance/约束值图（value map），再用运动规划器在值图上求解轨迹的机器人操作框架。

## 核心要点
1. LLM 负责把指令拆解为可组合的空间约束（affordance、避障区域等），无需任务特定训练
2. VLM 负责将语言约束 grounding 到 3D 场景坐标
3. 输出的是规划器可直接消费的代价/约束场，而非端到端动作

## 代表工作
- 待补充

## 相关概念
- [[OpenVLA]]
