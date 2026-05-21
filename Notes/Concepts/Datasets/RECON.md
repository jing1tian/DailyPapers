---
type: concept
aliases: [RECON Dataset, RECON Benchmark]
---

# RECON

## 定义
用于评估户外机器人导航规划能力的真实世界数据集与 Benchmark，包含 100 条户外导航轨迹，以绝对轨迹误差（ATE）和相对位姿误差（RPE）作为规划精度指标。

## 核心要点
1. 场景：真实户外环境（校园、街道等），非仿真
2. 任务：给定目标图像，规划从起点到目标的导航路径
3. 评估指标：ATE（Absolute Trajectory Error，绝对轨迹误差）、RPE（Relative Pose Error）
4. 常与 GNM、NOMAD 等导航基线比较

## 代表工作
- [[CoME]]: 在 RECON 上 ATE 从 NWM 的 1.13 降至 0.96，RPE 从 0.35 降至 0.28

## 相关概念
- [[NWM]]
- [[世界模型]]
