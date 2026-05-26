---
type: concept
aliases: [Act3D, Act-3D, 3D-point-based manipulation]
---

# Act3D

## 定义
基于 3D 点云的机器人操作策略方法，通过在 3D 空间中显式推理来预测末端执行器动作，属于 V-3D-A（Vision-to-3D-to-Action）范式的代表工作。

## 核心要点

1. 从 RGB-D 图像重建点云，在 3D 空间中推理物体位置和操作目标点
2. 利用 3D 几何先验提升对物体位置的空间理解能力
3. 相比纯 2D 方法，具有更好的视角泛化和空间推理能力
4. 局限：仅建模静态 3D 场景，不捕捉时序动态，无法预测运动

## 代表工作

- Act3D 原始论文: 在 RLBench 上取得 53.1% 平均成功率
- [[GAF]]: 将 Act3D 作为 V-3D-A 范式基线进行对比（GAF 以 +7.3% 成功率超越）

## 相关概念

- [[ManiGaussian]]
- [[扩散策略]]
- [[Gaussian Action Field]]
- [[3D Gaussian Splatting]]
