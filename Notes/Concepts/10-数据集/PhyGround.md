---
type: concept
aliases: [Physics-Grounded Evaluation Benchmark]
---

# PhyGround

## 定义
PhyGround：一个用于评估视频生成模型物理合理性的 benchmark，专注于测量模型是否真正理解场景的物理演化（而非仅凭视觉连续性欺骗评估）。

## 数学形式
N/A（评估 benchmark）

## 核心要点
1. 测量指标超越视觉质量（FID/PSNR），关注物理一致性和场景演化预测
2. 在 FCP（Looking Ahead）论文中被用于评估自回归视频模型是否真正学到了物理演化
3. 与 WorldModelBench 等形成世界模型评估体系

## 代表工作
- [[FCP]]: 在 PhyGround 上验证 Future Chunk Prediction 目标提升了物理合理性

## 相关概念
- [[WorldModelBench]]: 另一世界模型评估基准
- [[FCP]]: 相关方法
