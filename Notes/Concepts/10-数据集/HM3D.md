---
type: concept
aliases: [Habitat-Matterport 3D, HM3D Dataset]
---

# HM3D

## 定义

Habitat-Matterport 3D（HM3D）是基于真实室内场景扫描的大规模 3D 导航数据集，包含 1,000 个高保真三维室内环境，配套 Habitat 仿真平台使用。

## 核心要点

1. **规模**: 1,000 个真实室内场景（房屋、公寓、办公楼等）
2. **来源**: Matterport 三维扫描，高保真几何与纹理
3. **用途**: 视觉导航（ObjectNav、ImageGoal）、仿真训练
4. **测地线距离**: 场景带有网格信息，可计算精确测地线距离
5. **NavWAM 使用**: 185,000 条仿真轨迹用于阶段 1 预训练

## 代表工作

- [[NavWAM]]: 以 HM3D 仿真轨迹预训练世界动作模型

## 相关概念

- [[导航世界模型]]
- [[目标条件策略]]
- [[视觉预测]]
