---
type: concept
aliases: [Navigation World Model, NWM]
---

# NWM（Navigation World Model）

## 定义
Bar et al. (2024) 提出的基于扩散模型的导航世界模型，将相机位姿嵌入条件信号，在户外导航场景下生成与位姿一致的未来视频帧，用于目标导航规划。

## 核心要点
1. 以相机位姿序列作为条件，生成位姿对齐的视频预测
2. 用于 model-based 导航规划：在世界模型中模拟多条轨迹，选择最优路径
3. 在 RECON benchmark 上 ATE=1.13，是扩散世界模型导航的强基线
4. 局限：单模型无显式长期记忆，场景一致性随距离增加而下降

## 代表工作
- [[CoME]]: 以 NWM 为基础并通过 STM/LTM 专家增强，ATE 降至 0.96

## 相关概念
- [[世界模型]]
- [[扩散世界模型]]
- [[RECON]]
