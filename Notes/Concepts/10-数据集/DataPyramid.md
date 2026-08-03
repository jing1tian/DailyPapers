---
type: concept
aliases: [Data Pyramid, 数据金字塔]
---

# DataPyramid

## 定义
为 embodied 操作任务提出的分层数据框架，将训练数据按质量和获取成本组织成金字塔：底层仿真数据（量大、成本低）、中层视频观察数据（无动作标注）、顶层真实机器人演示（质量最高、量少）。

## 数学形式
$$\mathcal{D} = \mathcal{D}_{\text{sim}} \cup \mathcal{D}_{\text{video}} \cup \mathcal{D}_{\text{real}}, \quad |\mathcal{D}_{\text{sim}}| \gg |\mathcal{D}_{\text{video}}| \gg |\mathcal{D}_{\text{real}}|}$$

## 核心要点
1. 底层：仿真器生成数据（如 [[ManiSkill3]]），量大但存在 sim-to-real gap
2. 中层：互联网视频或 ego-video，提供视觉先验但缺少动作标注
3. 顶层：真实机器人演示（如 [[AgiBot World]]、[[RoboMIND]]），质量最高
4. 各层数据按比例混合，显式建模数据质量层次对策略泛化的影响

## 代表工作
- [[DataPyramid]]: "Data Pyramid for Embodied Manipulation" (2607.24744)

## 相关概念
- [[WAM]]
- [[GigaWorld1]]
- [[ManiSkill3]]
