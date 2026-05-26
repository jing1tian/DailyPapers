---
type: concept
aliases: [FoundationPose, Foundation Pose Estimation]
---

# FoundationPose

## 定义
NVIDIA 提出的通用 6DoF 物体姿态估计与追踪框架，支持模型已知和无模型两种模式，基于大规模合成数据预训练，在新物体上无需微调即可运行。

## 数学形式
姿态估计通过 render-and-compare：
$$\hat{T} = \arg\min_T \| I_{obs} - \mathcal{R}(M, T) \|$$

## 核心要点
1. 双模式：给定 3D CAD 模型（精确）或只给参考图像（泛化）
2. 大规模合成渲染数据预训练，在真实场景 zero-shot 泛化
3. 姿态精化用 render-and-compare + 迭代优化
4. 追踪模式利用时序连续性，比逐帧估计更快更稳
5. 在 YCB-V、BOP challenge 上达到 SOTA

## 代表工作
- [[GEM-4D]]: 使用 FoundationPose 做 pose 初始化
- [[ComPose]]: 与 FoundationPose 对比

## 相关概念
- [[MegaPose]]
- [[DexYCB]]
- [[6-3D视觉]]
